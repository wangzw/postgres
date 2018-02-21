/*-------------------------------------------------------------------------
 *
 * llvmQual.c
 *	  Just-in-time compiler for expressions
 *
 * Copyright (c) 2016, Institute for System Programming of the Russian Academy of Sciences
 *
 *
 * IDENTIFICATION
 *	  src/backend/executor/llvmQual.c
 *
 *-------------------------------------------------------------------------
 */
/*
 *	 INTERFACE ROUTINES
 *		ExecCompileExprLLVM - compile expression with LLVM JIT
 *		IsExprSupportedLLVM - check if expression is supported by LLVM JIT
 *
 *	 NOTES
 *		ExecCompileExprLLVM traverses expression tree it is called to generate
 *		code for and generates optimized code for each supported node (see
 *		GenerateExpr routine) during a depth-first traversal. Unsupported
 *		nodes are handled by the base case, which consists in generating
 *		calls to corresponding `evalfunc` functions.
 *
 *		For this reason, IsExprSupportedLLVM checks only the top-level node of
 *		an expression, meaning that any expression tree is supported as long
 *		as its root node is supported. However, some nodes are deliberately
 *		marked as not supported in IsExprSupportedLLVM although they
 *		technically are. One (and only, so far) example of such a node is Var:
 *		expressions consisting of just a single Var are not worth
 *		the compiling effort.
 *
 *		If compilation succeeds, top-level `evalfunc` pointer is swapped
 *		to point to generated code, which effectively means that it will be
 *		called whenever ExecEvalExpr is called on the expression.
 */

#include "postgres.h"

#include "access/htup_details.h"
#include "executor/executor.h"
#include "executor/llvm_backend_wrapper.h"
#include "nodes/nodeFuncs.h"
#include "nodes/print.h"
#include "optimizer/planmain.h"
#include "utils/array.h"
#include "utils/lsyscache.h"

#include <llvm-c/Analysis.h>
#include <llvm-c/Core.h>
#include <llvm-c/ExecutionEngine.h>
#include <llvm-c/Transforms/IPO.h>
#include <llvm-c/Transforms/PassManagerBuilder.h>
#include <llvm-c/Transforms/Scalar.h>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>


typedef struct LLVMTupleAttr {
	LLVMValueRef value;
	LLVMValueRef isNull;
	LLVMValueRef isDone;
} LLVMTupleAttr;

#define INIT_LLVMTUPLEATTR \
		{LLVMConstNull(LLVMInt64Type()), \
		LLVMConstNull(LLVMInt8Type()), \
		LLVMConstInt(LLVMInt32Type(), ExprSingleResult, false)}


/*
 * Field numbers.
 */
#define EXPRCONTEXT_SCANTUPLE_FIELDNO 1
#define EXPRCONTEXT_INNERTUPLE_FIELDNO 2
#define EXPRCONTEXT_OUTERTUPLE_FIELDNO 3
#define FUNCTIONCALLINFODATA_FLINFO_FIELDNO 0
#define FUNCTIONCALLINFODATA_CONTEXT_FIELDNO 1
#define FUNCTIONCALLINFODATA_RESULTINFO_FIELDNO 2
#define FUNCTIONCALLINFODATA_FNCOLLATION_FIELDNO 3
#define FUNCTIONCALLINFODATA_ISNULL_FIELDNO 4
#define FUNCTIONCALLINFODATA_NARGS_FIELDNO 5
#define FUNCTIONCALLINFODATA_ARG_FIELDNO 6
#define FUNCTIONCALLINFODATA_ARGNULL_FIELDNO 7
#define HEAPTUPLEDATA_DATA_FIELDNO 3
#define RETURNSETINFO_TYPE_FIELDNO 0
#define RETURNSETINFO_ECONTEXT_FIELDNO 1
#define RETURNSETINFO_EXPECTEDDESC_FIELDNO 2
#define RETURNSETINFO_ALLOWEDMODES_FIELDNO 3
#define RETURNSETINFO_RETURNMODE_FIELDNO 4
#define RETURNSETINFO_ISDONE_FIELDNO 5
#define RETURNSETINFO_SETRESULT_FIELDNO 6
#define RETURNSETINFO_SETDESC_FIELDNO 7
#define TUPLETABLESLOT_TUPLE_FIELDNO 5
#define TUPLETABLESLOT_TUPLEDESCRIPTOR_FIELDNO 6
#define TUPLETABLESLOT_VALUES_FIELDNO 10
#define TUPLETABLESLOT_ISNULL_FIELDNO 11


/*
 * RuntimeContext - struct holding run-time values loaded for each expression
 * prior to execution
 */
typedef struct RuntimeContext {
	LLVMValueRef isNullPtr, isDonePtr;  /* temporary vars  */
	LLVMValueRef fcinfo;  /* FunctionCallInfo - used for function calls */
	LLVMValueRef econtext;  /* dynamic ExprContext */

	/*
	 * TupleTableSlots loaded from ExprContext.
	 */
	LLVMValueRef scanSlot, scanSlotValues, scanSlotIsNull, scanSlotTuple,
		scanSlotTupleDesc;
	LLVMValueRef innerSlot, innerSlotValues, innerSlotIsNull, innerSlotTuple,
		innerSlotTupleDesc;
	LLVMValueRef outerSlot, outerSlotValues, outerSlotIsNull, outerSlotTuple,
		outerSlotTupleDesc;
} RuntimeContext;


static LLVMValueRef
ConstPointer(LLVMTypeRef pointer_type, void *pointer)
{
	return LLVMConstIntToPtr(
		LLVMConstInt(LLVMInt64Type(), (uintptr_t) pointer, false),
		pointer_type);
}


static char *
AddLLVMPrefix(const char *name)
{
	char *name_buf = (char *) palloc(1024 * sizeof(char));

	snprintf(name_buf, 1024, "llvm_%s", name);
	return name_buf;
}


static LLVMValueRef
LLVMAddGlobalWithPrefix(LLVMModuleRef mod,
					   LLVMTypeRef type, const char *name)
{
	char *llvm_name = AddLLVMPrefix(name);
	LLVMValueRef global = LLVMAddGlobal(mod, type, llvm_name);

	pfree(llvm_name);
	return global;
}


static LLVMValueRef
LLVMAddFunctionWithPrefix(LLVMModuleRef mod,
					   const char *name, LLVMTypeRef type)
{
	char *llvm_name = AddLLVMPrefix(name);
	LLVMValueRef func = LLVMAddFunction(mod, llvm_name, type);

	pfree(llvm_name);
	return func;
}


static LLVMValueRef
AddLLVMIntrinsic(LLVMBuilderRef builder, const char *name, LLVMTypeRef type)
{
	LLVMModuleRef mod = LLVMGetGlobalParent(
		LLVMGetBasicBlockParent(LLVMGetInsertBlock(builder)));
	LLVMValueRef function = LLVMGetNamedFunction(mod, name);

	if (!function)
	{
		function = LLVMAddFunction(mod, name, type);
	}

	return function;
}


/* LLVM intrinsic */
static void
GenerateMemSet(LLVMBuilderRef builder, size_t alignment, LLVMValueRef dest,
			   LLVMValueRef val, LLVMValueRef len)
{
	LLVMTypeRef memset_arg_types[] = {
		LLVMPointerType(LLVMInt8Type(), 0),  /* <dest> */
		LLVMInt8Type(),  /* <val> */
		LLVMInt64Type(),  /* <len> */
		LLVMInt32Type(),  /* <align> */
		LLVMInt1Type()  /* <isvolatile> */
	};
	LLVMTypeRef memset_type = LLVMFunctionType(
		LLVMVoidType(), memset_arg_types, lengthof(memset_arg_types), false);
	LLVMValueRef memset_f = AddLLVMIntrinsic(
		builder, "llvm.memset.p0i8.i64", memset_type);
	LLVMValueRef args[5];

	args[0] = LLVMBuildPointerCast(
		builder, dest, LLVMPointerType(LLVMInt8Type(), 0),
		LLVMGetValueName(dest));
	args[1] = LLVMBuildTrunc(
		builder, val, LLVMInt8Type(), LLVMGetValueName(val));
	args[2] = LLVMBuildZExt(
		builder, len, LLVMInt64Type(), LLVMGetValueName(len));
	args[3] = LLVMConstInt(LLVMInt32Type(), alignment, false);
	args[4] = LLVMConstNull(LLVMInt1Type());
	LLVMBuildCall(builder, memset_f, args, lengthof(args), "");
}


static LLVMTypeRef
BackendStructTypeFunc(LLVMTypeRef (*define_func)(LLVMModuleRef))
{
	LLVMModuleRef mod = LLVMModuleCreateWithName("");
	LLVMTypeRef type = define_func(mod);
	LLVMDisposeModule(mod);
	return type;
}

#define BackendStructType(name) BackendStructTypeFunc(define_struct_##name)
#define BackendUnionType(name) BackendStructTypeFunc(define_union_##name)


static LLVMTypeRef
ExprStateEvalFuncType(void)
{
	LLVMTypeRef arg_types[] = {
		LLVMPointerType(BackendStructType(ExprState), 0),
		LLVMPointerType(BackendStructType(ExprContext), 0),
		LLVMPointerType(LLVMInt8Type(), 0),
		LLVMPointerType(LLVMInt32Type(), 0)
	};
	LLVMTypeRef function_type = LLVMFunctionType(
		LLVMInt64Type(), arg_types, lengthof(arg_types), false);

	StaticAssertStmt(sizeof(bool) == sizeof(int8), "bool is 8-bit");
	StaticAssertStmt(sizeof(ExprDoneCond) == sizeof(int32),
					 "ExprDoneCond is 32-bit");
	StaticAssertStmt(sizeof(Datum) == sizeof(int64), "Datum is 32-bit");

	return function_type;
}


static LLVMValueRef
GenerateCallBackend(LLVMBuilderRef builder,
					LLVMValueRef (*define_func)(LLVMModuleRef),
					LLVMValueRef *args, int num_args)
{
	int i;
	LLVMValueRef ret;

	LLVMBasicBlockRef this_bb = LLVMGetInsertBlock(builder);
	LLVMValueRef this_function = LLVMGetBasicBlockParent(this_bb);
	LLVMModuleRef mod = LLVMGetGlobalParent(this_function);
	LLVMValueRef func = define_func(mod);
	LLVMTypeRef args_types[FUNC_MAX_ARGS];
	LLVMValueRef args_fixed[FUNC_MAX_ARGS];

	Assert(num_args ==
		   LLVMCountParamTypes(LLVMGetElementType(LLVMTypeOf(func))));

	LLVMGetParamTypes(LLVMGetElementType(LLVMTypeOf(func)), args_types);

	for (i = 0; i < num_args; i++)
	{
		args_fixed[i] = LLVMBuildBitCast(
			builder, args[i], args_types[i], LLVMGetValueName(args[i]));
	}

	/* Cannot assign a name to void values */
	ret = LLVMBuildCall(builder, func, args_fixed, num_args, "");
	LLVMSetInstructionCallConv(ret, LLVMGetFunctionCallConv(func));
	return ret;
}


/*
 * Initialize llvm FunctionCallInfo structure.
 */
static LLVMValueRef
GenerateInitFCInfo(LLVMBuilderRef builder, FunctionCallInfo fcinfo,
					LLVMValueRef fcinfo_llvm)
{
	LLVMValueRef context_ptr, resultinfo_ptr, fncollation_ptr,
		nargs_ptr, argnulls;

	context_ptr =
		LLVMBuildStructGEP(builder, fcinfo_llvm,
						   FUNCTIONCALLINFODATA_CONTEXT_FIELDNO,
						   "context_ptr");
	resultinfo_ptr =
		LLVMBuildStructGEP(builder, fcinfo_llvm,
						   FUNCTIONCALLINFODATA_RESULTINFO_FIELDNO,
						   "resultinfo_ptr");
	fncollation_ptr =
		LLVMBuildStructGEP(builder, fcinfo_llvm,
						   FUNCTIONCALLINFODATA_FNCOLLATION_FIELDNO,
						   "fncollation_ptr");
	nargs_ptr =
		LLVMBuildStructGEP(builder, fcinfo_llvm,
						   FUNCTIONCALLINFODATA_NARGS_FIELDNO,
						   "nargs_ptr");

	LLVMBuildStore(builder,
				   ConstPointer(
					   LLVMGetElementType(LLVMTypeOf(context_ptr)),
					   fcinfo->context),
				   context_ptr);
	LLVMBuildStore(builder,
				   ConstPointer(
					   LLVMGetElementType(LLVMTypeOf(resultinfo_ptr)),
					   fcinfo->resultinfo),
				   resultinfo_ptr);
	LLVMBuildStore(builder, LLVMConstInt(
			LLVMInt32Type(), fcinfo->fncollation, false), fncollation_ptr);
	LLVMBuildStore(builder, LLVMConstInt(
			LLVMInt16Type(), fcinfo->nargs, false), nargs_ptr);

	/*
	 * Zero-initialize `argnull`.
	 */
	StaticAssertStmt(sizeof(bool) == sizeof(int8_t), "bool is 8-bit");
	argnulls = LLVMBuildStructGEP(builder, fcinfo_llvm,
								  FUNCTIONCALLINFODATA_ARG_FIELDNO,
								  "argnull_ptr");
	argnulls = LLVMBuildStructGEP(builder, argnulls, 0, "argnull_ptr");
	GenerateMemSet(builder, 1, argnulls,
				   LLVMConstNull(LLVMInt8Type()),
				   LLVMConstInt(LLVMInt32Type(), fcinfo->nargs, false));

	return fcinfo_llvm;
}


/*
 * Allocate llvm FunctionCallInfo locally.
 */
static LLVMValueRef
GenerateAllocFCInfo(LLVMBuilderRef builder)
{
	LLVMTypeRef fcinfo_type;
	LLVMValueRef fcinfo_llvm;

	fcinfo_type = BackendStructType(FunctionCallInfoData);

	fcinfo_llvm = LLVMBuildAlloca(builder, fcinfo_type, "fcinfo");

	return fcinfo_llvm;
}


static LLVMValueRef
define_llvm_pg_function(LLVMBuilderRef builder, FmgrInfo *flinfo)
{
	LLVMBasicBlockRef this_bb = LLVMGetInsertBlock(builder);
	LLVMValueRef this_function = LLVMGetBasicBlockParent(this_bb);
	LLVMModuleRef mod = LLVMGetGlobalParent(this_function);

	LLVMValueRef functionRef = define_llvm_function(flinfo->fn_oid, mod);

	/*
	 * If there is no "LLVM equivalent" of postgres backend function,
	 * then try to call postgres function directly.
	 */
	if (functionRef == NULL)
	{
		LLVMTypeRef fcinfo_type, function_type_llvm;

		fcinfo_type = LLVMPointerType(
				BackendStructType(FunctionCallInfoData), 0);
		function_type_llvm = LLVMFunctionType(
				LLVMInt64Type(), &fcinfo_type, 1, false);

		functionRef = ConstPointer(
			LLVMPointerType(function_type_llvm, 0), flinfo->fn_addr);
	}

	Assert(functionRef != NULL);

	return functionRef;
}


/*
 * Same as GenerateFunctionCall*Coll, but with support of NULL for
 * arguments and result.
 */
static LLVMTupleAttr
GenerateFunctionCallNCollNull(LLVMBuilderRef builder, FunctionCallInfo fcinfo,
							  LLVMValueRef fcinfo_llvm, LLVMTupleAttr *attr,
							  bool retSet, LLVMValueRef hasSetArg)
{

	LLVMTupleAttr result = INIT_LLVMTUPLEATTR;

	LLVMValueRef isNull_ptr;

	LLVMValueRef functionRef = define_llvm_pg_function(
		builder, fcinfo->flinfo);
	LLVMValueRef flinfo_ptr = LLVMBuildStructGEP(
		builder, fcinfo_llvm, FUNCTIONCALLINFODATA_FLINFO_FIELDNO,
		"flinfo_ptr");
	LLVMValueRef args = LLVMBuildStructGEP(
		builder, fcinfo_llvm, FUNCTIONCALLINFODATA_ARG_FIELDNO, "args");
	LLVMValueRef argnulls = LLVMBuildStructGEP(
		builder, fcinfo_llvm, FUNCTIONCALLINFODATA_ARGNULL_FIELDNO,
		"argnulls");
	LLVMTypeRef fcinfo_type;
	LLVMValueRef resultinfo_ptr, rsinfo_isDone_ptr;
	int arg_index;

	for (arg_index = 0; arg_index < fcinfo->nargs; ++arg_index)
	{
		LLVMValueRef arg_ptr = LLVMBuildStructGEP(
				builder, args, arg_index, "arg_ptr");
		LLVMValueRef argnull_ptr = LLVMBuildStructGEP(
				builder, argnulls, arg_index, "argnull_ptr");

		LLVMBuildStore(builder, attr[arg_index].value, arg_ptr);
		LLVMBuildStore(builder, attr[arg_index].isNull, argnull_ptr);
	}

	isNull_ptr = LLVMBuildStructGEP(
		builder, fcinfo_llvm, FUNCTIONCALLINFODATA_ISNULL_FIELDNO,
		"isNull_ptr");
	LLVMBuildStore(builder, LLVMConstNull(LLVMInt8Type()), isNull_ptr);

	LLVMBuildStore(builder,
				   ConstPointer(
					   LLVMGetElementType(LLVMTypeOf(flinfo_ptr)),
					   fcinfo->flinfo),
				   flinfo_ptr);

	if (LLVMIsAFunction(functionRef))
	{
		LLVMRemoveFunctionAttr(functionRef, LLVMNoInlineAttribute);
		LLVMAddFunctionAttr(functionRef, LLVMAlwaysInlineAttribute);
	}

	if (retSet)
	{
		resultinfo_ptr =
			LLVMBuildStructGEP(builder, fcinfo_llvm,
							   FUNCTIONCALLINFODATA_RESULTINFO_FIELDNO,
							   "resultinfo_ptr");
		resultinfo_ptr = LLVMBuildBitCast(builder,
			LLVMBuildLoad(builder, resultinfo_ptr, "resultinfo"),
			LLVMPointerType(BackendStructType(ReturnSetInfo), 0),
			"resultinfo");
		rsinfo_isDone_ptr =
			LLVMBuildStructGEP(builder, resultinfo_ptr,
							   RETURNSETINFO_ISDONE_FIELDNO, "&isDone");
	}

	LLVMGetParamTypes(LLVMGetElementType(LLVMTypeOf(functionRef)),
			&fcinfo_type);
	fcinfo_llvm = LLVMBuildBitCast(builder, fcinfo_llvm, fcinfo_type,
								   LLVMGetValueName(fcinfo_llvm));

	result.value = LLVMBuildCall(
		builder, functionRef, &fcinfo_llvm, 1,
		get_func_name(fcinfo->flinfo->fn_oid));
	result.isNull = LLVMBuildLoad(builder, isNull_ptr, "isNull");

	if (retSet)
		result.isDone = LLVMBuildLoad(builder, rsinfo_isDone_ptr, "isDone");
	else
	{
		result.isDone = LLVMConstInt(LLVMInt32Type(), ExprSingleResult, false);
	}

	return result;
}


static LLVMValueRef
FCInfoLLVMAddRetSet(LLVMBuilderRef builder, ExprContext* econtext,
					TupleDesc expectedDesc, LLVMValueRef fcinfo_llvm)
{
	LLVMValueRef resultinfo_ptr =
		LLVMBuildStructGEP(builder, fcinfo_llvm,
						   FUNCTIONCALLINFODATA_RESULTINFO_FIELDNO,
						   "resultinfo_ptr");
	LLVMTypeRef rsinfoType = BackendStructType(ReturnSetInfo);
	LLVMValueRef rsinfo_ptr = LLVMBuildAlloca(builder, rsinfoType, "rsinfo");
	LLVMValueRef ret = rsinfo_ptr;
	LLVMValueRef rsinfo_type_ptr =
		LLVMBuildStructGEP(builder, rsinfo_ptr,
						   RETURNSETINFO_TYPE_FIELDNO,
						   "&rsinfo->type");
	LLVMValueRef rsinfo_econtext_ptr =
		LLVMBuildStructGEP(builder, rsinfo_ptr,
						   RETURNSETINFO_ECONTEXT_FIELDNO,
						   "&rsinfo->econtext");
	LLVMValueRef rsinfo_expectedDesc_ptr =
		LLVMBuildStructGEP(builder, rsinfo_ptr,
						   RETURNSETINFO_EXPECTEDDESC_FIELDNO,
						   "&rsinfo->expectedDesc");
	LLVMValueRef rsinfo_allowedModes_ptr =
		LLVMBuildStructGEP(builder, rsinfo_ptr,
						   RETURNSETINFO_ALLOWEDMODES_FIELDNO,
						   "&rsinfo->allowedModes");
	LLVMValueRef rsinfo_returnMode_ptr =
		LLVMBuildStructGEP(builder, rsinfo_ptr,
						   RETURNSETINFO_RETURNMODE_FIELDNO,
						   "&rsinfo->returnMode");
	LLVMValueRef rsinfo_isDone_ptr =
		LLVMBuildStructGEP(builder, rsinfo_ptr,
						   RETURNSETINFO_ISDONE_FIELDNO,
						   "&rsinfo->isDone");
	LLVMValueRef rsinfo_setResult_ptr =
		LLVMBuildStructGEP(builder, rsinfo_ptr,
						   RETURNSETINFO_SETRESULT_FIELDNO,
						   "&rsinfo->setResult");
	LLVMValueRef rsinfo_setDesc_ptr =
		LLVMBuildStructGEP(builder, rsinfo_ptr,
						   RETURNSETINFO_SETDESC_FIELDNO,
						   "&rsinfo->setDesc");

	LLVMBuildStore(builder, LLVMConstInt(
		LLVMInt32Type(), T_ReturnSetInfo, false), rsinfo_type_ptr);
	LLVMBuildStore(builder, ConstPointer(
		LLVMGetElementType(LLVMTypeOf(rsinfo_econtext_ptr)), econtext),
		rsinfo_econtext_ptr);
	LLVMBuildStore(builder, ConstPointer(
		LLVMGetElementType(LLVMTypeOf(rsinfo_expectedDesc_ptr)),
		expectedDesc), rsinfo_expectedDesc_ptr);
	LLVMBuildStore(builder, LLVMConstInt(
		LLVMInt32Type(), (int) (SFRM_ValuePerCall | SFRM_Materialize), false),
		rsinfo_allowedModes_ptr);
	LLVMBuildStore(builder, LLVMConstInt(
		LLVMInt32Type(), SFRM_ValuePerCall, false), rsinfo_returnMode_ptr);
	LLVMBuildStore(builder, LLVMConstInt(
				LLVMInt32Type(), ExprSingleResult, false), rsinfo_isDone_ptr);
	LLVMBuildStore(builder, ConstPointer(
		LLVMGetElementType(LLVMTypeOf(rsinfo_setResult_ptr)), NULL),
		rsinfo_setResult_ptr);
	LLVMBuildStore(builder, ConstPointer(
		LLVMGetElementType(LLVMTypeOf(rsinfo_setDesc_ptr)), NULL),
		rsinfo_setDesc_ptr);
	rsinfo_ptr = LLVMBuildBitCast(builder, rsinfo_ptr,
		LLVMGetElementType(LLVMTypeOf(resultinfo_ptr)), "rsinfo");
	LLVMBuildStore(builder, rsinfo_ptr, resultinfo_ptr);
	return ret;
}


bool
IsExprSupportedLLVM(Expr *node)
{
	switch (nodeTag(node))
	{
		case T_Const:
		case T_RelabelType:
		case T_RowExpr:
		case T_OpExpr:
		case T_FuncExpr:
		case T_Aggref:
		case T_BoolExpr:
		case T_BooleanTest:
		case T_TargetEntry:
		case T_List:
			return true;

		case T_Var:
		{
			if (((Var *) node)->varattno == InvalidAttrNumber)
				return false;

			return true;
		}

		case T_CaseExpr:
		{
			CaseExpr   *caseexpr = (CaseExpr *) node;

			if (caseexpr->arg)
				return false;

			return true;
		}

		case T_NullTest:
		{
			NullTest *ntest = (NullTest *) node;

			if (ntest->argisrow)
				return false;

			return true;
		}

		case T_ScalarArrayOpExpr:
		{
			ScalarArrayOpExpr *opexpr = (ScalarArrayOpExpr *) node;

			if (!IsA(lsecond(opexpr->args), Const))
				return false;

			return true;
		}

		default:
			return false;
	}
}


static LLVMTupleAttr
GenerateDefaultExpr(LLVMBuilderRef builder,
		ExprState *exprstate,
		RuntimeContext *rtcontext)
{
	LLVMValueRef evalfunc_ptr = ConstPointer(
			LLVMPointerType(
				LLVMPointerType(ExprStateEvalFuncType(), 0), 0),
			&exprstate->evalfunc);
	LLVMValueRef evalfunc = LLVMBuildLoad(
			builder, evalfunc_ptr, "evalfunc");
	LLVMValueRef args[] = {
		ConstPointer(
			LLVMPointerType(BackendStructType(ExprState), 0),
			exprstate),
		rtcontext->econtext,
		rtcontext->isNullPtr,
		rtcontext->isDonePtr
	};

	LLVMTupleAttr result = INIT_LLVMTUPLEATTR;
	result.value = LLVMBuildCall(
			builder, evalfunc, args, lengthof(args), "value");
	result.isNull = LLVMBuildLoad(
			builder, rtcontext->isNullPtr, "isNull");
	result.isDone = LLVMBuildLoad(
			builder, rtcontext->isDonePtr, "isDone");
	return result;
}


/*
 * AttributeStats - struct holding number of attributes per each tuple slot
 * and whether any system attributes are accessed
 */
typedef struct AttributeStats
{
	int numScanAttrs, numInnerAttrs, numOuterAttrs;
	bool hasScanSysAttr, hasInnerSysAttr, hasOuterSysAttr;
} AttributeStats;


static bool
GetAttributeStatsForExpressionWalker(Node *node, AttributeStats *stats)
{
	if (!node)
	{
		return false;
	}

	switch (nodeTag(node))
	{
		case T_Var:
		{
			Var *variable = (Var *) node;
			AttrNumber attno = variable->varattno;
			int *numAttrs;
			bool *hasSysAttr;

			switch (variable->varno)
			{
				case INNER_VAR:
					numAttrs = &stats->numInnerAttrs;
					hasSysAttr = &stats->hasInnerSysAttr;
					break;

				case OUTER_VAR:
					numAttrs = &stats->numOuterAttrs;
					hasSysAttr = &stats->hasOuterSysAttr;
					break;

				default:
					numAttrs = &stats->numScanAttrs;
					hasSysAttr = &stats->hasScanSysAttr;
			}

			if (attno > 0)
			{
				*numAttrs = Max(*numAttrs, attno);
			}
			else if (attno < 0)
			{
				*hasSysAttr = true;
			}

			break;
		}

		case T_Aggref:
			return false;

		default: break;
	}

	return expression_tree_walker(
		node, GetAttributeStatsForExpressionWalker, stats);
}


/*
 * GetAttributeStatsForExpression - create AttributeStats struct for
 * expression
 */
static AttributeStats
GetAttributeStatsForExpression(Expr *expr)
{
	AttributeStats stats = {};
	GetAttributeStatsForExpressionWalker((Node *) expr, &stats);
	return stats;
}


/*
 * GetSomeAttrs - generate a call to `slot_getsomeattrs`
 */
static void
GetSomeAttrs(LLVMBuilderRef builder, LLVMValueRef slot, int attnum)
{
	LLVMValueRef args[] = {
		slot,
		LLVMConstInt(LLVMInt32Type(), attnum, false)
	};

	GenerateCallBackend(
		builder, define_slot_getsomeattrs, args, lengthof(args));
}


/*
 * GetAttr - generate a call to `slot_getattr`
 */
static LLVMValueRef
GetAttr(LLVMBuilderRef builder, LLVMValueRef slot,
		int attnum, LLVMValueRef isNull)
{
	LLVMValueRef args[] = {
		slot,
		LLVMConstInt(LLVMInt32Type(), attnum, false),
		isNull
	};

	return GenerateCallBackend(
		builder, define_slot_getattr, args, lengthof(args));
}

/*
 * LoadUsedAttrs - load attributes used in the expression
 *
 * For system attributes, this function preloads `tuple` and `tupleDesc` into
 * RuntimeContext.
 *
 * For ordinary attributes, this function preloads `values` and `isNull`
 * arrays into RuntimeContext.
 */
static void
LoadUsedAttrs(LLVMBuilderRef builder, Expr *expr, RuntimeContext *rtcontext)
{
	AttributeStats stats = GetAttributeStatsForExpression(expr);

	if (stats.hasScanSysAttr || stats.numScanAttrs > 0)
	{
		LLVMValueRef scanSlot = LLVMBuildLoad(
			builder,
			LLVMBuildStructGEP(
				builder, rtcontext->econtext, EXPRCONTEXT_SCANTUPLE_FIELDNO,
				"scanSlotPtr"),
			"scanSlot");

		rtcontext->scanSlot = scanSlot;

		if (stats.hasScanSysAttr)
		{
			rtcontext->scanSlotTuple = LLVMBuildLoad(
				builder,
				LLVMBuildStructGEP(
					builder, scanSlot, TUPLETABLESLOT_TUPLE_FIELDNO,
					"scanSlotTuplePtr"),
				"scanSlotTuple");
			rtcontext->scanSlotTupleDesc = LLVMBuildLoad(
				builder,
				LLVMBuildStructGEP(
					builder, scanSlot,
					TUPLETABLESLOT_TUPLEDESCRIPTOR_FIELDNO,
					"scanSlotTupleDescPtr"),
				"scanSlotTupleDesc");
		}

		if (stats.numScanAttrs > 0)
		{
			GetSomeAttrs(builder, scanSlot, stats.numScanAttrs);

			rtcontext->scanSlotValues = LLVMBuildLoad(
				builder,
				LLVMBuildStructGEP(
					builder, scanSlot, TUPLETABLESLOT_VALUES_FIELDNO,
					"scanSlotValuesPtr"),
				"scanSlotValues");
			rtcontext->scanSlotIsNull = LLVMBuildLoad(
				builder,
				LLVMBuildStructGEP(
					builder, scanSlot, TUPLETABLESLOT_ISNULL_FIELDNO,
					"scanSlotIsNullPtr"),
				"scanSlotIsNull");
		}
	}

	if (stats.hasInnerSysAttr || stats.numInnerAttrs > 0)
	{
		LLVMValueRef innerSlot = LLVMBuildLoad(
			builder,
			LLVMBuildStructGEP(
				builder, rtcontext->econtext, EXPRCONTEXT_INNERTUPLE_FIELDNO,
				"innerSlotPtr"),
			"innerSlot");

		rtcontext->innerSlot = innerSlot;

		if (stats.hasInnerSysAttr)
		{
			rtcontext->innerSlotTuple = LLVMBuildLoad(
				builder,
				LLVMBuildStructGEP(
					builder, innerSlot, TUPLETABLESLOT_TUPLE_FIELDNO,
					"innerSlotTuplePtr"),
				"innerSlotTuple");
			rtcontext->innerSlotTupleDesc = LLVMBuildLoad(
				builder,
				LLVMBuildStructGEP(
					builder, innerSlot,
					TUPLETABLESLOT_TUPLEDESCRIPTOR_FIELDNO,
					"innerSlotTupleDescPtr"),
				"innerSlotTupleDesc");
		}

		if (stats.numInnerAttrs > 0)
		{
			GetSomeAttrs(builder, innerSlot, stats.numInnerAttrs);

			rtcontext->innerSlotValues = LLVMBuildLoad(
				builder,
				LLVMBuildStructGEP(
					builder, innerSlot, TUPLETABLESLOT_VALUES_FIELDNO,
					"innerSlotValuesPtr"),
				"innerSlotValues");
			rtcontext->innerSlotIsNull = LLVMBuildLoad(
				builder,
				LLVMBuildStructGEP(
					builder, innerSlot, TUPLETABLESLOT_ISNULL_FIELDNO,
					"innerSlotIsNullPtr"),
				"innerSlotIsNull");
		}
	}

	if (stats.hasOuterSysAttr || stats.numOuterAttrs > 0)
	{
		LLVMValueRef outerSlot = LLVMBuildLoad(
			builder,
			LLVMBuildStructGEP(
				builder, rtcontext->econtext, EXPRCONTEXT_OUTERTUPLE_FIELDNO,
				"outerSlotPtr"),
			"outerSlot");

		rtcontext->outerSlot = outerSlot;

		if (stats.hasOuterSysAttr)
		{
			rtcontext->outerSlotTuple = LLVMBuildLoad(
				builder,
				LLVMBuildStructGEP(
					builder, outerSlot, TUPLETABLESLOT_TUPLE_FIELDNO,
					"outerSlotTuplePtr"),
				"outerSlotTuple");
			rtcontext->outerSlotTupleDesc = LLVMBuildLoad(
				builder,
				LLVMBuildStructGEP(
					builder, outerSlot,
					TUPLETABLESLOT_TUPLEDESCRIPTOR_FIELDNO,
					"outerSlotTupleDPtresc"),
				"outerSlotTupleDesc");
		}

		if (stats.numOuterAttrs > 0)
		{
			GetSomeAttrs(builder, outerSlot, stats.numOuterAttrs);

			rtcontext->outerSlotValues = LLVMBuildLoad(
				builder,
				LLVMBuildStructGEP(
					builder, outerSlot, TUPLETABLESLOT_VALUES_FIELDNO,
					"outerSlotValuesPtr"),
				"outerSlotValues");
			rtcontext->outerSlotIsNull = LLVMBuildLoad(
				builder,
				LLVMBuildStructGEP(
					builder, outerSlot, TUPLETABLESLOT_ISNULL_FIELDNO,
					"outerSlotIsNullPtr"),
				"outerSlotIsNull");
		}
	}
}


static LLVMValueRef
GetSysAttr(LLVMBuilderRef builder, LLVMValueRef tuple, AttrNumber attno,
		   LLVMValueRef tupleDesc, LLVMValueRef isNull)
{
	LLVMValueRef args[] = {
		tuple,
		LLVMConstInt(LLVMInt32Type(), attno, true),
		tupleDesc,
		isNull
	};

	return GenerateCallBackend(
		builder, define_heap_getsysattr, args, lengthof(args));
}


static LLVMTupleAttr
GenerateExpr(LLVMBuilderRef builder,
			 ExprState *exprstate,
			 ExprContext *econtext,
			 RuntimeContext *rtcontext)
{
	if (!IsExprSupportedLLVM(exprstate->expr))
		return GenerateDefaultExpr(builder, exprstate, rtcontext);

	switch (nodeTag(exprstate->expr))
	{
		case T_Const:
		{
			Const  *con = (Const *) exprstate->expr;
			LLVMTupleAttr attr = INIT_LLVMTUPLEATTR;

			attr.isNull = LLVMConstInt(
				LLVMInt8Type(), con->constisnull, false);
			attr.value = con->constisnull
				? LLVMConstNull(LLVMInt64Type())
				: LLVMConstInt(LLVMInt64Type(), con->constvalue, false);
			return attr;
		}

		case T_Var:
		{
			Var *variable = (Var *) exprstate->expr;
			AttrNumber attno = variable->varattno;
			LLVMValueRef slotValues, slotIsNull, tuple, tupleDesc;
			LLVMTupleAttr result = INIT_LLVMTUPLEATTR;

			switch (variable->varno)
			{
				case INNER_VAR:
					slotValues = rtcontext->innerSlotValues;
					slotIsNull = rtcontext->innerSlotIsNull;
					tuple = rtcontext->innerSlotTuple;
					tupleDesc = rtcontext->innerSlotTupleDesc;
					break;

				case OUTER_VAR:
					slotValues = rtcontext->outerSlotValues;
					slotIsNull = rtcontext->outerSlotIsNull;
					tuple = rtcontext->outerSlotTuple;
					tupleDesc = rtcontext->outerSlotTupleDesc;
					break;

				default:
					slotValues = rtcontext->scanSlotValues;
					slotIsNull = rtcontext->scanSlotIsNull;
					tuple = rtcontext->scanSlotTuple;
					tupleDesc = rtcontext->scanSlotTupleDesc;
			}

			if (attno > 0)
			{
				LLVMValueRef varIndex = LLVMConstInt(
					LLVMInt32Type(), attno - 1, false);

				Assert(slotValues && slotIsNull);

				result.value = LLVMBuildLoad(
					builder,
					LLVMBuildGEP(
						builder, slotValues, &varIndex, 1, "var_valuePtr"),
					"var_value");
				result.isNull = LLVMBuildLoad(
					builder,
					LLVMBuildGEP(
						builder, slotIsNull, &varIndex, 1, "var_isNullPtr"),
					"var_isNull");
			}
			else if (attno < 0)
			{
				result.value = GetSysAttr(
					builder, tuple, attno, tupleDesc, rtcontext->isNullPtr);
				result.isNull = LLVMBuildLoad(
					builder, rtcontext->isNullPtr, "var_isNull");
			}
			else
			{
				pg_unreachable();
			}

			return result;
		}

		case T_RelabelType:
		{
			GenericExprState *gstate = (GenericExprState *) exprstate;
			ExprState *argstate = gstate->arg;
			return GenerateExpr(
				builder, argstate, econtext, rtcontext);
		}

		case T_RowExpr:
		{
			RowExprState *rstate = (RowExprState *) exprstate;
			ListCell *arg;
			int natts = rstate->tupdesc->natts;
			int i;

			LLVMValueRef tuple, t_data, heap_form_tuple_args[3], list[natts];
			LLVMValueRef rstate_tupdesc_ptr = ConstPointer(
				LLVMPointerType(LLVMPointerType(LLVMInt8Type(), 0), 0),
				&rstate->tupdesc);
			LLVMTupleAttr result = INIT_LLVMTUPLEATTR;
			LLVMTupleAttr attr[natts];

			LLVMBasicBlockRef this_bb = LLVMGetInsertBlock(builder);
			LLVMValueRef this_function = LLVMGetBasicBlockParent(this_bb);
			LLVMModuleRef mod = LLVMGetGlobalParent(this_function);

			/* Allocate workspace */
			LLVMValueRef values_llvm = LLVMAddGlobalWithPrefix(
				mod, LLVMArrayType(LLVMInt64Type(), natts),
				"RowExpr_values");
			LLVMValueRef isNull_llvm = LLVMAddGlobalWithPrefix(
				mod, LLVMArrayType(LLVMInt8Type(), natts),
				"RowExpr_isNull");

			/* preset to nulls in case rowtype has some later-added columns */
			for (i = 0; i < natts; i++)
			{
				list[i] = LLVMConstInt(LLVMInt8Type(), 1, false);
			}

			LLVMSetInitializer(values_llvm,
							   LLVMConstNull(
								   LLVMArrayType(LLVMInt64Type(), natts)));
			LLVMSetInitializer(isNull_llvm,
							   LLVMConstArray(LLVMInt8Type(), list, natts));
			LLVMSetLinkage(values_llvm, LLVMInternalLinkage);
			LLVMSetLinkage(isNull_llvm, LLVMInternalLinkage);

			/* Evaluate field values */
			i = 0;
			foreach (arg, rstate->args)
			{
				ExprState  *argstate = (ExprState *) lfirst(arg);
				LLVMValueRef value, null;
				LLVMValueRef index[] = {
					LLVMConstInt(LLVMInt32Type(), 0, false),
					LLVMConstInt(LLVMInt32Type(), i, false)
				};

				attr[i] = GenerateExpr(
					builder, argstate, econtext, rtcontext);

				value = LLVMBuildInBoundsGEP(builder, values_llvm,
											 index, 2, "values[i]");
				null = LLVMBuildInBoundsGEP(builder, isNull_llvm,
											index, 2, "isNull[i]");
				LLVMBuildStore(builder, attr[i].value, value);
				LLVMBuildStore(builder, attr[i].isNull, null);

				i++;
			}

			/* heap_form_tuple */
			heap_form_tuple_args[0] = LLVMBuildLoad(
				builder, rstate_tupdesc_ptr, "rstate_tupdesc");
			heap_form_tuple_args[1] = values_llvm;
			heap_form_tuple_args[2] = isNull_llvm;
			tuple = GenerateCallBackend(
				builder, define_heap_form_tuple, heap_form_tuple_args, 3);

			/* HeapTupleGetDatum */
			t_data = LLVMBuildStructGEP(
				builder, tuple, HEAPTUPLEDATA_DATA_FIELDNO, "t_data_ptr");
			t_data = LLVMBuildLoad(builder, t_data, "t_data");
			result.value = GenerateCallBackend(
				builder, define_HeapTupleHeaderGetDatum, &t_data, 1);
			result.isNull = LLVMConstInt(LLVMInt8Type(), 0, false);

			return result;
		}

		case T_OpExpr:
		case T_FuncExpr:
		{
			FuncExprState *fexprstate = (FuncExprState *) exprstate;
			FunctionCallInfo fcinfo = &fexprstate->fcinfo_data;
			Oid funcid = 0;
			Oid inputcollid = 0;
			bool strict, retSet;
			LLVMValueRef hasSetArg_ptr = LLVMBuildAlloca(
					builder, LLVMInt1Type(), "&hasSetArg");
			LLVMValueRef argsAreDone_ptr = LLVMBuildAlloca(
					builder, LLVMInt1Type(), "&argsAreDone");
			LLVMTupleAttr result, func_result, mid_result;
			LLVMTupleAttr attr[list_length(fexprstate->args)];
			LLVMValueRef fcinfo_llvm, funcResultStore, funcResultStore_ptr,
						 cond, frsIsNull, rsinfo_ptr, hasSetArg, argsAreDone;
			LLVMValueRef vars[4];
			ListCell* cell;
			short i;

			LLVMValueRef multi_llvm = LLVMConstInt(
							LLVMInt32Type(), ExprMultipleResult, 0);

			LLVMBasicBlockRef this_bb = LLVMGetInsertBlock(builder);
			LLVMValueRef function = LLVMGetBasicBlockParent(this_bb);
			LLVMBasicBlockRef exit_bb = 0;  /* -Wmaybe-uninitialized */
			LLVMBasicBlockRef tuplestore_bb = 0, tuplestore_tuple_bb = 0,
							  tuplestore_no_tuple_bb = 0, calculate_bb = 0,
							  exit_no_calc_bb = 0;
			if (IsA(exprstate->expr, OpExpr))
			{
				OpExpr *op = (OpExpr *) exprstate->expr;
				funcid = op->opfuncid;
				inputcollid = op->inputcollid;
			}
			else if (IsA(exprstate->expr, FuncExpr))
			{
				FuncExpr *func_expr = (FuncExpr *) exprstate->expr;
				funcid = func_expr->funcid;
				inputcollid = func_expr->inputcollid;
			}
			else
			{
				Assert(false);
			}

			LLVMBuildStore(builder,
				LLVMConstInt(LLVMInt1Type(), false, 0), hasSetArg_ptr);
			LLVMBuildStore(builder,
				LLVMConstInt(LLVMInt1Type(), false, 0), argsAreDone_ptr);
			init_fcache(funcid, inputcollid, fexprstate,
						econtext->ecxt_per_query_memory, true);

			strict = fexprstate->func.fn_strict;
			retSet = fexprstate->func.fn_retset;

			funcResultStore_ptr = ConstPointer(LLVMPointerType(
				LLVMPointerType(BackendStructType(Tuplestorestate), 0), 0),
				&fexprstate->funcResultStore);
			if (retSet)
			{
				tuplestore_bb =
					LLVMAppendBasicBlock(function, "FuncExpr_tuplestore");
				tuplestore_tuple_bb =
					LLVMAppendBasicBlock(function, "FuncExpr_tuplestore_tuple");
				tuplestore_no_tuple_bb =
					LLVMAppendBasicBlock(function, "FuncExpr_tuplestore_no_tuple");
				calculate_bb =
					LLVMAppendBasicBlock(function, "FuncExpr_calculate");
				exit_no_calc_bb =
					LLVMAppendBasicBlock(function, "FuncExpr_exit_no_calc");
				funcResultStore = LLVMBuildLoad(builder, funcResultStore_ptr,
						"funcResultStore");
				frsIsNull = LLVMBuildIsNull(builder, funcResultStore,
						"funcResultStore==NULL");
				LLVMBuildCondBr(builder, frsIsNull, calculate_bb, tuplestore_bb);

				LLVMPositionBuilderAtEnd(builder, exit_no_calc_bb);
				result.value = LLVMBuildPhi(
						builder, LLVMInt64Type(), "value");
				result.isNull = LLVMBuildPhi(
						builder, LLVMInt8Type(), "isNull");
				result.isDone = LLVMBuildPhi(
						builder, LLVMInt32Type(), "isDone");

				LLVMPositionBuilderAtEnd(builder, calculate_bb);
				this_bb = calculate_bb;
			}

			exit_bb = LLVMAppendBasicBlock(function, "FuncExpr_exit");

			LLVMPositionBuilderAtEnd(builder, exit_bb);
			mid_result.value = LLVMBuildPhi(
				builder, LLVMInt64Type(), "value");
			mid_result.isNull = LLVMBuildPhi(
				builder, LLVMInt8Type(), "isNull");
			mid_result.isDone = LLVMBuildPhi(
				builder, LLVMInt32Type(), "isDone");

			LLVMPositionBuilderAtEnd(builder, this_bb);

			i = 0;
			foreach (cell, fexprstate->args)
			{
				ExprState *argstate = lfirst(cell);
				LLVMTupleAttr arg = GenerateExpr(
					builder, argstate, econtext, rtcontext);
				LLVMValueRef hasSetArg_cond, arg_isDone;

				attr[i] = arg;
				i++;
				hasSetArg_cond = LLVMBuildICmp(builder, LLVMIntNE,
					arg.isDone, LLVMConstInt(LLVMInt32Type(), ExprSingleResult, 0),
					"");
				hasSetArg = LLVMBuildLoad(builder, hasSetArg_ptr, "");
				hasSetArg = LLVMBuildSelect(builder, hasSetArg_cond,
					LLVMConstInt(LLVMInt1Type(), 1, 0), hasSetArg,
					"select_hasSetArg");
				LLVMBuildStore(builder, hasSetArg, hasSetArg_ptr);
				arg_isDone = LLVMBuildICmp(builder, LLVMIntEQ,
					arg.isDone, LLVMConstInt(LLVMInt32Type(), ExprEndResult, 0),
					"");
				argsAreDone = LLVMBuildLoad(builder, argsAreDone_ptr, "argsAreDone");
				argsAreDone = LLVMBuildSelect(builder, arg_isDone,
					LLVMConstInt(LLVMInt1Type(), 1, 0), argsAreDone,
					"select_argsAreDone");
				LLVMBuildStore(builder, argsAreDone, argsAreDone_ptr);
				if (strict)
				{
					LLVMBasicBlockRef next_bb = LLVMAppendBasicBlock(
						function, "FuncExpr_next");
					LLVMValueRef null_llvm = LLVMConstNull(LLVMInt64Type());
					LLVMValueRef true_llvm = LLVMConstInt(
							LLVMInt8Type(), 1, false);
					LLVMValueRef single_llvm = LLVMConstInt(
							LLVMInt32Type(), ExprSingleResult, false);
					LLVMValueRef end_llvm = LLVMConstInt(
							LLVMInt32Type(), ExprEndResult, false);
					LLVMValueRef isDone = LLVMBuildICmp(builder, LLVMIntEQ,
						arg.isDone, LLVMConstInt(LLVMInt32Type(), ExprEndResult, 0), "isDone");
					LLVMValueRef isNull, isDone_phi;

					this_bb = LLVMGetInsertBlock(builder);
					LLVMAddIncoming(mid_result.value, &null_llvm, &this_bb, 1);
					LLVMAddIncoming(mid_result.isNull, &true_llvm, &this_bb, 1);
					if (retSet)
						isDone_phi = end_llvm;
					else
						isDone_phi = LLVMBuildSelect(builder, isDone,
							end_llvm, single_llvm, "select_isDone_phi");
					LLVMAddIncoming(mid_result.isDone, &isDone_phi, &this_bb, 1);
					isNull = LLVMBuildIsNotNull(
						builder, arg.isNull, "isNull");
					LLVMBuildCondBr(builder, isNull, exit_bb, next_bb);

					LLVMPositionBuilderAtEnd(builder, next_bb);
					LLVMMoveBasicBlockBefore(next_bb, exit_bb);
					this_bb = next_bb;
				}
				else
					this_bb = LLVMGetInsertBlock(builder);
			}

			/*
			 * If a previous call of the function returned a set result in the form of
			 * a tuplestore, continue reading rows from the tuplestore until it's
			 * empty.
			 */
			if (retSet)
			{
				LLVMValueRef null_llvm = LLVMConstNull(LLVMInt64Type());
				LLVMValueRef true_llvm = LLVMConstInt(
						LLVMInt8Type(), 1, 0);
				LLVMValueRef end_llvm = LLVMConstInt(
						LLVMInt32Type(), ExprEndResult, 0);
				LLVMPositionBuilderAtEnd(builder, tuplestore_bb);
				vars[0] = LLVMBuildLoad(builder, funcResultStore_ptr,
										"funcResultStore");
				vars[1] = LLVMConstInt(LLVMInt8Type(), true, 0);
				vars[2] = LLVMConstInt(LLVMInt8Type(), false, 0);
				vars[3] = ConstPointer(LLVMPointerType(
					LLVMPointerType(BackendStructType(TupleTableSlot), 0), 0),
					&fexprstate->funcResultSlot);
				vars[3] = LLVMBuildLoad(builder, vars[3], "");
				cond = GenerateCallBackend(
					builder, define_tuplestore_gettupleslot, vars, 4);
				cond = LLVMBuildIsNotNull(builder, cond, "");
				LLVMBuildCondBr(builder, cond, tuplestore_tuple_bb,
					tuplestore_no_tuple_bb);

				LLVMPositionBuilderAtEnd(builder, tuplestore_tuple_bb);
				LLVMAddIncoming(result.isDone, &multi_llvm, &tuplestore_tuple_bb, 1);
				if (fexprstate->funcReturnsTuple)
				{
					/* We must return the whole tuple as a Datum. */
					LLVMValueRef false_llvm = LLVMConstInt(
						LLVMInt8Type(), false, 0);
					LLVMValueRef value = GenerateCallBackend(
						builder, define_ExecFetchSlotTupleDatum, &vars[3], 1);
					LLVMAddIncoming(result.value, &value,
									&tuplestore_tuple_bb, 1);
					LLVMAddIncoming(result.isNull, &false_llvm,
									&tuplestore_tuple_bb, 1);
				}
				else
				{
					/* Extract the first column and return it as a scalar. */
					LLVMValueRef funcResultSlot = vars[3];
					LLVMValueRef value = GetAttr(
						builder, funcResultSlot, 1, rtcontext->isNullPtr);
					LLVMValueRef isNull = LLVMBuildLoad(
					builder, rtcontext->isNullPtr, "var_isNull");
					LLVMAddIncoming(result.value, &value, &tuplestore_tuple_bb, 1);
					LLVMAddIncoming(result.isNull, &isNull, &tuplestore_tuple_bb, 1);
				}
				LLVMBuildBr(builder, exit_no_calc_bb);

				/* Exhausted the tuplestore, so clean up */
				LLVMPositionBuilderAtEnd(builder, tuplestore_no_tuple_bb);
				GenerateCallBackend(builder,
					define_tuplestore_end, vars, 1);
				LLVMBuildStore(builder, LLVMConstNull(LLVMTypeOf(funcResultStore)),
					funcResultStore_ptr);
				hasSetArg = LLVMBuildLoad(builder, hasSetArg_ptr, "hasSetArg");
				LLVMAddIncoming(result.value, &null_llvm,
								&tuplestore_no_tuple_bb, 1);
				LLVMAddIncoming(result.isNull, &true_llvm,
								&tuplestore_no_tuple_bb, 1);
				LLVMAddIncoming(result.isDone, &end_llvm,
								&tuplestore_no_tuple_bb, 1);
				LLVMBuildCondBr(builder, hasSetArg, calculate_bb, exit_no_calc_bb);
			}
			LLVMPositionBuilderAtEnd(builder, this_bb);
			fcinfo_llvm = GenerateInitFCInfo(
				builder, fcinfo, rtcontext->fcinfo);

			if (retSet)
			{
				rsinfo_ptr = FCInfoLLVMAddRetSet(
						builder, econtext, fexprstate->funcResultDesc,
						fcinfo_llvm);
			}
			hasSetArg = LLVMBuildLoad(builder, hasSetArg_ptr, "hasSetArg");
			func_result = GenerateFunctionCallNCollNull(
					builder, fcinfo, fcinfo_llvm, attr, retSet, hasSetArg);

			/* SFRM_Materialize implementation */
			if (retSet)
			{
				LLVMValueRef funcResultSlot_ptr =
					ConstPointer(LLVMPointerType(LLVMPointerType(
									BackendStructType(TupleTableSlot), 0), 0),
							&fexprstate->funcResultSlot);
				LLVMValueRef rsinfo_returnMode_ptr =
					LLVMBuildStructGEP(builder, rsinfo_ptr,
									   RETURNSETINFO_RETURNMODE_FIELDNO,
									   "&rsinfo->returnMode");
				LLVMValueRef returnMode = LLVMBuildLoad(
						builder, rsinfo_returnMode_ptr, "rsinfo->returnMode");
				LLVMValueRef cond = LLVMBuildICmp(builder, LLVMIntEQ, returnMode,
						LLVMConstInt(LLVMInt32Type(), SFRM_Materialize, 0), "");
				LLVMValueRef null_llvm = LLVMConstNull(LLVMInt64Type());
				LLVMValueRef true_llvm = LLVMConstInt(
						LLVMInt8Type(), 1, 0);
				LLVMValueRef end_llvm = LLVMConstInt(
						LLVMInt32Type(), ExprEndResult, 0);
				LLVMValueRef rsinfo_setResult_ptr, setResult, slotDesc,
							 rsinfo_setDesc_ptr, setDesc, slot_ptr;

				LLVMBasicBlockRef materialize_bb =
					LLVMAppendBasicBlock(function, "FuncExpr_materialize");
				LLVMBasicBlockRef restart_bb =
					LLVMAppendBasicBlock(function, "FuncExpr_restart");
				LLVMMoveBasicBlockBefore(materialize_bb, exit_bb);
				LLVMMoveBasicBlockBefore(restart_bb, exit_bb);

				LLVMBuildCondBr(builder, cond, materialize_bb, exit_bb);

				LLVMPositionBuilderAtEnd(builder, materialize_bb);
				LLVMAddIncoming(result.value, &null_llvm,
						&materialize_bb, 1);
				LLVMAddIncoming(result.isNull, &true_llvm,
						&materialize_bb, 1);
				LLVMAddIncoming(result.isDone, &end_llvm,
						&materialize_bb, 1);
				rsinfo_setResult_ptr =
					LLVMBuildStructGEP(builder, rsinfo_ptr,
									   RETURNSETINFO_SETRESULT_FIELDNO,
									   "&rsinfo->setResult");
				setResult = LLVMBuildLoad(
						builder, rsinfo_setResult_ptr, "rsinfo->setResult");

				/* if setResult was left null, treat it as empty set */
				cond = LLVMBuildIsNull(builder, setResult, "");
				LLVMBuildCondBr(builder, cond, exit_no_calc_bb, restart_bb);

				/* prepare to return values from the tuplestore */
				LLVMPositionBuilderAtEnd(builder, restart_bb);
				LLVMBuildStore(builder, setResult, funcResultStore_ptr);
				rsinfo_setDesc_ptr =
					LLVMBuildStructGEP(builder, rsinfo_ptr,
									   RETURNSETINFO_SETDESC_FIELDNO,
									   "&rsinfo->setDesc");
				setDesc = LLVMBuildLoad(
						builder, rsinfo_setDesc_ptr, "rsinfo->setDesc");
				if (fexprstate->funcResultDesc)
					slotDesc = ConstPointer(
							LLVMPointerType(LLVMInt8Type(), 0),
							fexprstate->funcResultDesc);
				else
					slotDesc = setDesc;
				slot_ptr = GenerateCallBackend(
						builder, define_MakeSingleTupleTableSlot, &slotDesc, 1);
				LLVMBuildStore(builder, slot_ptr, funcResultSlot_ptr);

				/* loop back to top to start returning from tuplestore */
				LLVMBuildBr(builder, tuplestore_bb);
			}
			else
				LLVMBuildBr(builder, exit_bb);

			LLVMAddIncoming(mid_result.value, &func_result.value,
					&this_bb, 1);
			LLVMAddIncoming(mid_result.isNull, &func_result.isNull,
					&this_bb, 1);
			LLVMAddIncoming(mid_result.isDone, &func_result.isDone,
					&this_bb, 1);
			LLVMPositionBuilderAtEnd(builder, exit_bb);
			{
				LLVMValueRef result_isDone = LLVMBuildICmp(
						builder, LLVMIntNE, mid_result.isDone,
						LLVMConstInt(LLVMInt32Type(), ExprEndResult, false),
						"result_isDone");
				hasSetArg = LLVMBuildLoad(builder, hasSetArg_ptr, "hasSetArg");
				result_isDone = LLVMBuildAnd(builder,
						result_isDone, hasSetArg,
						"result_isDone && hasSetArg");
				mid_result.isDone = LLVMBuildSelect(builder, result_isDone,
						LLVMConstInt(LLVMInt32Type(), ExprMultipleResult, 0),
						mid_result.isDone,
						"select_isDone");
			}

			if (exit_no_calc_bb)
			{
				LLVMBuildBr(builder, exit_no_calc_bb);
				LLVMMoveBasicBlockAfter(exit_no_calc_bb, exit_bb);

				LLVMAddIncoming(result.value, &mid_result.value,
						&exit_bb, 1);
				LLVMAddIncoming(result.isNull, &mid_result.isNull,
						&exit_bb, 1);
				LLVMAddIncoming(result.isDone, &mid_result.isDone,
						&exit_bb, 1);
				LLVMPositionBuilderAtEnd(builder, exit_no_calc_bb);
			}
			else
				result = mid_result;

			/*
			 * For set returning functions we should only return if both
			 * the function and its set arg are done. Else we loop around
			 * to arg evaluation.
			 */
			if (retSet)
			{
				LLVMBasicBlockRef continue_bb =
					LLVMAppendBasicBlock(function, "FuncExpr_continue");
				LLVMValueRef result_isDone = LLVMBuildICmp(
						builder, LLVMIntEQ, result.isDone,
						LLVMConstInt(LLVMInt32Type(), ExprEndResult, false),
						"result_isDone");
				hasSetArg = LLVMBuildLoad(builder, hasSetArg_ptr, "hasSetArg");
				result_isDone = LLVMBuildAnd(builder,
						result_isDone, hasSetArg,
						"result_isDone && hasSetArg");
				argsAreDone = LLVMBuildLoad(builder, argsAreDone_ptr, "argsAreDone");
				result_isDone = LLVMBuildSelect(builder, argsAreDone,
						LLVMConstNull(LLVMInt1Type()),result_isDone, "select_isDone");
				LLVMBuildCondBr(builder, result_isDone, calculate_bb, continue_bb);

				LLVMPositionBuilderAtEnd(builder, continue_bb);
			}
			return result;
		}

		case T_BoolExpr:
		{
			BoolExprState *bexprstate = (BoolExprState*) exprstate;
			BoolExpr *boolexpr = (BoolExpr *) exprstate->expr;
			ListCell   *cell;

			LLVMBasicBlockRef this_bb = LLVMGetInsertBlock(builder);
			LLVMBasicBlockRef exit_bb;
			LLVMValueRef function = LLVMGetBasicBlockParent(this_bb);
			LLVMValueRef one = LLVMConstAllOnes(LLVMInt1Type());
			LLVMValueRef zero = LLVMConstNull(LLVMInt1Type());
			LLVMValueRef any_null = NULL;
			LLVMValueRef early_value, late_value;
			LLVMTupleAttr result = INIT_LLVMTUPLEATTR;

			if (boolexpr->boolop == NOT_EXPR)
			{
				ExprState *argstate = linitial(bexprstate->args);

				Assert(boolexpr->args->length == 1);

				result = GenerateExpr(
					builder, argstate, econtext, rtcontext);

				result.value = LLVMBuildIsNull(builder, result.value, "!val");
				result.value = LLVMBuildZExt(
					builder, result.value, LLVMInt64Type(), "!val");
				return result;
			}

			switch (boolexpr->boolop)
			{
				case AND_EXPR:
					early_value = zero;
					late_value = one;
					break;

				case OR_EXPR:
					early_value = one;
					late_value = zero;
					break;

				default:
					Assert(false);
			}

			exit_bb = LLVMAppendBasicBlock(function, "BoolExpr_exit");
			LLVMPositionBuilderAtEnd(builder, exit_bb);
			result.value = LLVMBuildPhi(builder, LLVMInt1Type(), "value");
			result.isNull = LLVMBuildPhi(builder, LLVMInt1Type(), "isNull");

			LLVMPositionBuilderAtEnd(builder, this_bb);

			foreach (cell, bexprstate->args)
			{
				ExprState *argstate = lfirst(cell);
				LLVMTupleAttr arg = GenerateExpr(
					builder, argstate, econtext, rtcontext);
				LLVMBasicBlockRef next_bb = LLVMAppendBasicBlock(
					function, "BoolExpr_next");
				LLVMValueRef not_null, early_exit;

				arg.value = LLVMBuildIsNotNull(builder, arg.value, "value");
				arg.isNull = LLVMBuildIsNotNull(
					builder, arg.isNull, "isNull");

				any_null = any_null
					? LLVMBuildOr(builder, any_null, arg.isNull, "any_null")
					: arg.isNull;

				not_null = LLVMBuildNot(builder, arg.isNull, "!isNull");
				early_exit = LLVMBuildICmp(
					builder, LLVMIntEQ, arg.value, early_value, "early_exit");
				early_exit = LLVMBuildAnd(
					builder, not_null, early_exit, "early_exit");

				this_bb = LLVMGetInsertBlock(builder);
				LLVMAddIncoming(result.value, &early_value, &this_bb, 1);
				LLVMAddIncoming(result.isNull, &zero, &this_bb, 1);
				LLVMBuildCondBr(builder, early_exit, exit_bb, next_bb);

				LLVMMoveBasicBlockBefore(next_bb, exit_bb);
				LLVMPositionBuilderAtEnd(builder, next_bb);
				this_bb = next_bb;
			}

			LLVMAddIncoming(result.value, &late_value, &this_bb, 1);
			LLVMAddIncoming(
				result.isNull, any_null ? &any_null : &zero, &this_bb, 1);
			LLVMBuildBr(builder, exit_bb);

			LLVMPositionBuilderAtEnd(builder, exit_bb);
			result.value = LLVMBuildZExt(
				builder, result.value, LLVMInt64Type(),
				boolexpr->boolop == AND_EXPR ? "and" : "or");
			result.isNull = LLVMBuildZExt(
				builder, result.isNull, LLVMInt8Type(), "isNull");
			return result;
		}

		case T_CaseExpr:
		{
			CaseExprState *caseExpr = (CaseExprState *) exprstate;
			List	   *clauses = caseExpr->args;
			ListCell   *cell;

			LLVMBasicBlockRef this_bb = LLVMGetInsertBlock(builder);
			LLVMValueRef function = LLVMGetBasicBlockParent(this_bb);
			LLVMBasicBlockRef done = LLVMAppendBasicBlock(
				function, "CaseExpr_done");
			LLVMTupleAttr result = INIT_LLVMTUPLEATTR;

			LLVMPositionBuilderAtEnd(builder, done);
			result.value = LLVMBuildPhi(
				builder, LLVMInt64Type(), "case_result");
			result.isNull = LLVMBuildPhi(
				builder, LLVMInt8Type(), "case_isNull");
			result.isDone = LLVMBuildPhi(
				builder, LLVMInt32Type(), "case_isDone");

			LLVMPositionBuilderAtEnd(builder, this_bb);

			foreach (cell, clauses)
			{
				CaseWhenState   *casewhen = lfirst(cell);
				LLVMTupleAttr clause_value, clause_result;

				LLVMBasicBlockRef isTrue_bb = LLVMAppendBasicBlock(
					function, "CaseExpr_isTrue");
				LLVMBasicBlockRef isFalse_bb = LLVMAppendBasicBlock(
					function, "CaseExpr_isFalse");
				LLVMBasicBlockRef current_bb;
				LLVMValueRef istrue, isnotnull, cmp;

				clause_value = GenerateExpr(
					builder, casewhen->expr, econtext, rtcontext);
				istrue = LLVMBuildIsNotNull(
					builder, clause_value.value, "value");
				isnotnull = LLVMBuildIsNull(
					builder, clause_value.isNull, "isnotnull");
				cmp = LLVMBuildAnd(builder, istrue, isnotnull, "cmp");
				LLVMBuildCondBr(builder, cmp, isTrue_bb, isFalse_bb);

				/*
				 * isTrue
				 */
				LLVMPositionBuilderAtEnd(builder, isTrue_bb);
				clause_result = GenerateExpr(
					builder, casewhen->result, econtext, rtcontext);
				current_bb = LLVMGetInsertBlock(builder);
				LLVMAddIncoming(result.value, &clause_result.value,
								&current_bb, 1);
				LLVMAddIncoming(result.isNull, &clause_result.isNull,
								&current_bb, 1);
				LLVMAddIncoming(result.isDone, &clause_result.isDone,
								&current_bb, 1);
				LLVMBuildBr(builder, done);

				/*
				 * isFalse
				 */
				LLVMPositionBuilderAtEnd(builder, isFalse_bb);
			}

			if (caseExpr->defresult)
			{
				LLVMTupleAttr defresult = GenerateExpr(
					builder, caseExpr->defresult, econtext, rtcontext);
				LLVMBasicBlockRef current_bb = LLVMGetInsertBlock(builder);

				LLVMAddIncoming(result.value, &defresult.value,
								&current_bb, 1);
				LLVMAddIncoming(result.isNull, &defresult.isNull,
								&current_bb, 1);
				LLVMAddIncoming(result.isDone, &defresult.isDone,
								&current_bb, 1);
				LLVMBuildBr(builder, done);
			}
			else
			{
				LLVMBasicBlockRef current_bb = LLVMGetInsertBlock(builder);
				LLVMValueRef llvm_null = LLVMConstInt(
					LLVMInt64Type(), 0, false);
				LLVMValueRef llvm_true = LLVMConstInt(
					LLVMInt8Type(), 1, false);

				LLVMAddIncoming(result.value, &llvm_null, &current_bb, 1);
				LLVMAddIncoming(result.isNull, &llvm_true, &current_bb, 1);
				LLVMBuildBr(builder, done);
			}

			LLVMPositionBuilderAtEnd(builder, done);
			return result;
		}

		case T_NullTest:
		{
			NullTestState *nstate = (NullTestState *) exprstate;
			NullTest *ntest = (NullTest *) nstate->xprstate.expr;
			LLVMValueRef isNull;
			LLVMTupleAttr result = INIT_LLVMTUPLEATTR;

			/*
			 * entry
			 */
			result = GenerateExpr(
				builder, nstate->arg, econtext, rtcontext);
			isNull = LLVMBuildIsNotNull(builder, result.isNull, "isNull");

			switch (ntest->nulltesttype)
			{
				case IS_NULL:
					result.value = LLVMBuildZExt(
						builder, isNull, LLVMInt64Type(), "isNull");
					break;

				case IS_NOT_NULL:
					result.value = LLVMBuildZExt(
						builder, LLVMBuildNot(builder, isNull, "notnull"),
						LLVMInt64Type(), "notnull");
					break;

				default:
					elog(ERROR, "unrecognized nulltesttype: %d",
						 (int) ntest->nulltesttype);
			}

			result.isNull = LLVMConstNull(LLVMInt8Type());
			return result;
		}

		case T_Aggref:
		{
			AggrefExprState *aggref = (AggrefExprState *) exprstate;
			LLVMTupleAttr result = INIT_LLVMTUPLEATTR;
			LLVMValueRef aggvalue_ptr = ConstPointer(
				LLVMPointerType(LLVMInt64Type(), 0),
				&econtext->ecxt_aggvalues[aggref->aggno]);
			LLVMValueRef aggnull_ptr = ConstPointer(
				LLVMPointerType(LLVMInt8Type(), 0),
				&econtext->ecxt_aggnulls[aggref->aggno]);

			Assert(econtext->ecxt_aggvalues);

			result.value = LLVMBuildLoad(builder, aggvalue_ptr, "aggvalue");
			result.isNull = LLVMBuildLoad(builder, aggnull_ptr, "aggnull");
			return result;
		}

		case T_ScalarArrayOpExpr:
		{
			ScalarArrayOpExprState *sstate =
				(ScalarArrayOpExprState *) exprstate;
			ScalarArrayOpExpr *opexpr = (ScalarArrayOpExpr *) exprstate->expr;
			bool useOr = opexpr->useOr;
			Const *arrayexpr = (Const *) lsecond(opexpr->args);
			FunctionCallInfo fcinfo = &sstate->fxprstate.fcinfo_data;
			ArrayType *array;
			char *s;
			bits8 *bitmap;
			int bitmask, nitems, itemno;

			LLVMTupleAttr attr[2];

			LLVMBasicBlockRef this_bb = LLVMGetInsertBlock(builder);
			LLVMValueRef function = LLVMGetBasicBlockParent(this_bb);
			LLVMBasicBlockRef exit_bb;

			LLVMTupleAttr null = {
				LLVMConstNull(LLVMInt64Type()),
				LLVMConstInt(LLVMInt8Type(), 1, false),
				LLVMConstInt(LLVMInt32Type(), ExprSingleResult, false)
			};
			LLVMTupleAttr early_result = {
				LLVMConstInt(LLVMInt64Type(), useOr, false),
				LLVMConstNull(LLVMInt8Type())
			};
			LLVMTupleAttr late_result = {
				LLVMConstInt(LLVMInt64Type(), !useOr, false),
				LLVMConstNull(LLVMInt8Type())
			};
			LLVMValueRef any_null;

			LLVMTupleAttr result = INIT_LLVMTUPLEATTR;
			LLVMTupleAttr scalar;

			init_fcache(opexpr->opfuncid, opexpr->inputcollid,
						&sstate->fxprstate, econtext->ecxt_per_query_memory,
						true);

			Assert(!sstate->fxprstate.func.fn_retset);

			if (arrayexpr->constisnull)
			{
				return null;
			}

			array = DatumGetArrayTypeP(arrayexpr->constvalue);
			nitems = ArrayGetNItems(ARR_NDIM(array), ARR_DIMS(array));

			if (nitems <= 0)
			{
				result.isNull = LLVMConstNull(LLVMInt8Type());
				result.value = LLVMConstInt(LLVMInt64Type(), !useOr, false);
				return result;
			}

			get_typlenbyvalalign(ARR_ELEMTYPE(array),
								 &sstate->typlen,
								 &sstate->typbyval,
								 &sstate->typalign);
			s = (char *) ARR_DATA_PTR(array);
			bitmap = ARR_NULLBITMAP(array);
			bitmask = 1;

			/*
			 * Evaluate arguments.
			 */
			scalar = GenerateExpr(
				builder, linitial(sstate->fxprstate.args), econtext,
				rtcontext);

			this_bb = LLVMGetInsertBlock(builder);
			exit_bb = LLVMAppendBasicBlock(
				function, "ScalarArrayOpExpr_exit");

			/*
			 * Create result phis.
			 */
			LLVMPositionBuilderAtEnd(builder, exit_bb);
			result.value = LLVMBuildPhi(
				builder, LLVMInt64Type(), "result.value");
			result.isNull = LLVMBuildPhi(
				builder, LLVMInt8Type(), "result.isNull");
			LLVMPositionBuilderAtEnd(builder, this_bb);

			/*
			 * entry - Return NULL if the scalar is NULL.
			 */
			if (sstate->fxprstate.func.fn_strict)
			{
				LLVMBasicBlockRef loop_bb = LLVMAppendBasicBlock(
					function, "ScalarArrayOpExpr_loop");
				LLVMValueRef isNull = LLVMBuildIsNotNull(
					builder, scalar.isNull, "scalar_isNull");

				LLVMMoveBasicBlockAfter(loop_bb, this_bb);

				LLVMAddIncoming(result.value, &null.value, &this_bb, 1);
				LLVMAddIncoming(result.isNull, &null.isNull, &this_bb, 1);
				LLVMBuildCondBr(builder, isNull, exit_bb, loop_bb);

				LLVMPositionBuilderAtEnd(builder, loop_bb);
				this_bb = loop_bb;
			}

			attr[0] = scalar;
			any_null = LLVMConstNull(LLVMInt1Type());

			/* Loop over the array elements */
			for (itemno = 0; itemno < nitems; ++itemno)
			{
				Datum elt_datum;
				bool elt_isNull;

				/* Get array element, checking for NULL */
				if (bitmap && (*bitmap & bitmask) == 0)
				{
					elt_datum = 0;
					elt_isNull = true;
				}
				else
				{
					elt_datum = fetch_att(
						s, sstate->typbyval, sstate->typlen);
					s = att_addlength_pointer(s, sstate->typlen, s);
					s = (char *) att_align_nominal(s, sstate->typalign);

					elt_isNull = false;
				}

				/* Call comparison function */
				if (elt_isNull && sstate->fxprstate.func.fn_strict)
				{
					any_null = LLVMBuildOr(
						builder, any_null, LLVMConstAllOnes(LLVMInt1Type()),
						"any_null");
				}
				else
				{
					LLVMBasicBlockRef checkresult_bb = LLVMAppendBasicBlock(
						function, "ScalarArrayOpExpr_checkresult");
					LLVMBasicBlockRef next_bb = LLVMAppendBasicBlock(
						function, "ScalarArrayOpExpr_next");
					LLVMValueRef true_llvm = LLVMConstAllOnes(LLVMInt1Type());
					LLVMValueRef fcinfo_llvm, isNull, early_exit, phi;
					LLVMValueRef hasSetArg = LLVMConstInt(LLVMInt1Type(), 0, 0);
					LLVMTupleAttr thisresult;

					LLVMMoveBasicBlockAfter(next_bb, this_bb);
					LLVMMoveBasicBlockAfter(checkresult_bb, this_bb);

					Assert(fcinfo->nargs == 2);
					attr[1].value = LLVMConstInt(
						LLVMInt64Type(), elt_datum, false);
					attr[1].isNull = LLVMConstInt(
						LLVMInt8Type(), elt_isNull, false);
					attr[1].isDone = LLVMConstInt(
						LLVMInt32Type(), ExprSingleResult, false);

					fcinfo_llvm = GenerateInitFCInfo(
						builder, fcinfo, rtcontext->fcinfo);
					thisresult = GenerateFunctionCallNCollNull(
						builder, fcinfo, fcinfo_llvm, attr, false, hasSetArg);
					isNull = LLVMBuildIsNotNull(
						builder, thisresult.isNull, "thisresult.isNull");
					LLVMBuildCondBr(builder, isNull, next_bb, checkresult_bb);

					/*
					 * checkresult
					 */
					LLVMPositionBuilderAtEnd(builder, checkresult_bb);
					early_exit = LLVMBuildICmp(
						builder, LLVMIntEQ, thisresult.value,
						early_result.value, "early_exit");
					LLVMAddIncoming(result.value, &early_result.value,
									&checkresult_bb, 1);
					LLVMAddIncoming(result.isNull, &early_result.isNull,
									&checkresult_bb, 1);
					LLVMBuildCondBr(builder, early_exit, exit_bb, next_bb);

					/*
					 * next
					 */
					LLVMPositionBuilderAtEnd(builder, next_bb);
					phi = LLVMBuildPhi(builder, LLVMInt1Type(), "any_null");
					LLVMAddIncoming(phi, &true_llvm, &this_bb, 1);
					LLVMAddIncoming(phi, &any_null, &checkresult_bb, 1);
					any_null = phi;

					this_bb = next_bb;
				}

				/* advance bitmap pointer if any */
				if (bitmap)
				{
					bitmask <<= 1;
					if (bitmask == 0x100)
					{
						bitmap++;
						bitmask = 1;
					}
				}
			}

			any_null = LLVMBuildZExt(
				builder, any_null, LLVMInt8Type(), "anu_null");
			LLVMAddIncoming(result.value, &late_result.value, &this_bb, 1);
			LLVMAddIncoming(result.isNull, &any_null, &this_bb, 1);
			LLVMBuildBr(builder, exit_bb);

			/*
			 * exit
			 */
			LLVMPositionBuilderAtEnd(builder, exit_bb);
			return result;
		}

		default:
		{
			return (LLVMTupleAttr) INIT_LLVMTUPLEATTR;
		}
	}
}


static RuntimeContext *
InitializeRuntimeContext(LLVMBuilderRef builder, LLVMValueRef ExecExpr,
						 ExprState *exprstate)
{
	RuntimeContext *rtcontext = palloc0(sizeof *rtcontext);

	/*
	 * Allocate convenience local vars.
	 */
	rtcontext->isNullPtr = LLVMBuildAlloca(
		builder, LLVMInt8Type(), "&isNull");
	rtcontext->isDonePtr = LLVMBuildAlloca(
		builder, LLVMInt32Type(), "&isDone");
	rtcontext->fcinfo = GenerateAllocFCInfo(builder);

	/*
	 * Get dynamic ExprContext from function arguments.
	 */
	rtcontext->econtext = LLVMGetParam(ExecExpr, 1);

	/*
	 * Preload all used attributes in Vars.
	 */
	LoadUsedAttrs(builder, exprstate->expr, rtcontext);

	return rtcontext;
}


static void
isinf_codegen(LLVMModuleRef mod)
{
	LLVMBuilderRef builder = LLVMCreateBuilder();
	LLVMTypeRef isinf_arg_types[] = {
		LLVMDoubleType()
	};
	LLVMTypeRef isinf_type = LLVMFunctionType(
		LLVMInt32Type(), isinf_arg_types, 1, false);
	LLVMValueRef isinf_f = LLVMAddFunction(
		mod, "__isinf", isinf_type);

	LLVMBasicBlockRef entry = LLVMAppendBasicBlock(isinf_f, "entry");
	LLVMValueRef left, right, ret, input;

	LLVMPositionBuilderAtEnd(builder, entry);
	input = LLVMGetParam(isinf_f, 0);
	left = LLVMBuildFCmp(builder, LLVMRealUEQ,
		LLVMConstReal(LLVMDoubleType(), INFINITY), input, "is_plus_inf");
	right = LLVMBuildFCmp(builder, LLVMRealUEQ,
		LLVMConstReal(LLVMDoubleType(), -INFINITY), input, "is_minus_inf");
	ret = LLVMBuildZExt(builder, LLVMBuildOr(builder, left, right, "isinf"),
						LLVMInt32Type(), "isinf");
	LLVMBuildRet(builder, ret);

	LLVMDisposeBuilder(builder);
}


/*
 * InitModule
 */
static LLVMModuleRef
InitModule(const char *module_name)
{
	LLVMModuleRef mod = LLVMModuleCreateWithName(module_name);

	/*
	 * Add optimized `isinf` function.
	 */
	isinf_codegen(mod);

	return mod;
}


/*
 * VerifyModule: verify generated LLVM module
 *
 * Returns true if the module is valid, false otherwise.
 */
static bool PG_USED_FOR_ASSERTS_ONLY
VerifyModule(LLVMModuleRef mod)
{
	char *message;

	if (LLVMVerifyModule(mod, LLVMReturnStatusAction, &message))
	{
		ereport(LOG, (errmsg("Broken LLVM module"),
					  errdetail("%s", message)));
		LLVMDisposeMessage(message);
		return false;
	}

	return true;
}


/*
 * RunPasses: optimize generated module.
 */
static void
RunPasses(LLVMExecutionEngineRef engine, LLVMModuleRef mod, LLVMValueRef main)
{
	LLVMPassManagerRef pass = LLVMCreatePassManager();

	LLVMAddTargetData(LLVMGetExecutionEngineTargetData(engine), pass);

	Assert(!LLVMGetNamedFunction(mod, "main"));
	LLVMSetValueName(main, "main");
	LLVMAddInternalizePass(pass, true);

	LLVMAddScopedNoAliasAAPass(pass);
	LLVMAddBasicAliasAnalysisPass(pass);
	LLVMAddIPSCCPPass(pass);
	LLVMAddFunctionInliningPass(pass);
	LLVMAddArgumentPromotionPass(pass);
	LLVMAddScalarReplAggregatesPass(pass);
	LLVMAddEarlyCSEPass(pass);
	LLVMAddCorrelatedValuePropagationPass(pass);
	LLVMAddTailCallEliminationPass(pass);
	LLVMAddCFGSimplificationPass(pass);
	LLVMAddLoopRotatePass(pass);
	LLVMAddLoopUnswitchPass(pass);
	LLVMAddInstructionCombiningPass(pass);
	LLVMAddIndVarSimplifyPass(pass);
	LLVMAddLoopIdiomPass(pass);
	LLVMAddLoopDeletionPass(pass);
	LLVMAddMemCpyOptPass(pass);
	LLVMAddJumpThreadingPass(pass);
	LLVMAddCorrelatedValuePropagationPass(pass);
	LLVMAddDeadArgEliminationPass(pass);
	LLVMAddLICMPass(pass);
	LLVMAddCFGSimplificationPass(pass);
	LLVMAddInstructionCombiningPass(pass);
	LLVMAddCFGSimplificationPass(pass);
	LLVMAddInstructionCombiningPass(pass);
	LLVMAddLoopUnrollPass(pass);
	LLVMAddInstructionCombiningPass(pass);
	LLVMAddLICMPass(pass);
	LLVMAddStripDeadPrototypesPass(pass);
	LLVMAddConstantPropagationPass(pass);
	LLVMAddGlobalDCEPass(pass);

	LLVMRunPassManager(pass, mod);
	LLVMDisposePassManager(pass);
}


static char *
GetDumpFileName(const char *base_format)
{
	static unsigned probe = 0;
	const size_t bufsize = 32;
	char filename_format[bufsize];
	char filename[bufsize];
	struct stat statres;

	snprintf(filename_format, bufsize, "llvm_dump/%s", base_format);

	/*
	 * Start with the last probe, trying to advance either backward or
	 * forward.
	 */
	if (snprintf(filename, bufsize, filename_format, probe),
		!stat(filename,  &statres))
	{
		/*
		 * Advance forward.
		 */
		do {
			snprintf(filename, bufsize, filename_format, ++probe);
		} while (!stat(filename,  &statres));
	}
	else
	{
		/*
		 * Advance backward.
		 */
		while (probe != 0 &&
			   (snprintf(filename, bufsize, filename_format, probe - 1),
				stat(filename,  &statres) < 0))
		{
			probe -= 1;
		}

		snprintf(filename, bufsize, filename_format, probe);
	}

	return strdup(filename);
}


static void
DumpExpression(Expr *expr, const char *format)
{
	char *filename = GetDumpFileName(format);
	int file = open(filename, O_WRONLY | O_CREAT | O_EXCL, 0666);
	int stdout_dup = dup(STDOUT_FILENO);

	fflush(stdout);
	dup2(file, STDOUT_FILENO);
	pprint(expr);
	fflush(stdout);
	dup2(stdout_dup, STDOUT_FILENO);

	close(stdout_dup);
	close(file);
	free(filename);
}


static void
DumpModule(LLVMModuleRef mod, const char *format)
{
	char *filename = GetDumpFileName(format);
	LLVMPrintModuleToFile(mod, filename, NULL);
	free(filename);
}


static ExprStateEvalFunc
CompileExpr(ExprState *exprstate, ExprContext *econtext)
{
	ExprStateEvalFunc func_addr;
	LLVMModuleRef mod = InitModule("expr");
	LLVMExecutionEngineRef engine = econtext->ecxt_estate->es_engine;
	LLVMValueRef ExecExpr_f = LLVMAddFunctionWithPrefix(
		mod, "ExecExpr", ExprStateEvalFuncType());
	LLVMBasicBlockRef entry_bb = LLVMAppendBasicBlock(ExecExpr_f, "entry");
	LLVMBuilderRef builder = LLVMCreateBuilder();
	LLVMTargetDataRef target_data = LLVMGetExecutionEngineTargetData(engine);

	LLVMSetDataLayout(mod, LLVMCopyStringRepOfTargetData(target_data));
	LLVMSetFunctionCallConv(ExecExpr_f, LLVMCCallConv);

	LLVMPositionBuilderAtEnd(builder, entry_bb);

	{
		RuntimeContext *rtcontext = InitializeRuntimeContext(
			builder, ExecExpr_f, exprstate);
		LLVMValueRef isNull_ptr = LLVMGetParam(ExecExpr_f, 2);
		LLVMValueRef isdone_ptr = LLVMGetParam(ExecExpr_f, 3);
		LLVMTupleAttr result = GenerateExpr(
			builder, exprstate, econtext, rtcontext);
		LLVMBasicBlockRef store_isdone_bb = LLVMAppendBasicBlock(
			ExecExpr_f, "store_isdone");
		LLVMBasicBlockRef return_bb = LLVMAppendBasicBlock(
			ExecExpr_f, "return");

		LLVMBuildStore(builder, result.isNull, isNull_ptr);
		LLVMBuildCondBr(builder,
						LLVMBuildIsNull(builder, isdone_ptr, "!isdone_ptr"),
						return_bb, store_isdone_bb);

		/*
		 * store_isdone
		 */
		LLVMPositionBuilderAtEnd(builder, store_isdone_bb);
		LLVMBuildStore(builder, result.isDone, isdone_ptr);
		LLVMBuildBr(builder, return_bb);

		/*
		 * return
		 */
		LLVMPositionBuilderAtEnd(builder, return_bb);
		LLVMBuildRet(builder, result.value);

		pfree(rtcontext);
	}

	Assert(VerifyModule(mod));

	if (enable_llvm_dump)
	{
		DumpExpression(exprstate->expr, "expr.%03u");
		DumpModule(mod, "dump.%03u.ll");
	}

	if (!debug_llvm_jit)
	{
		RunPasses(engine, mod, ExecExpr_f);
	}

	if (enable_llvm_dump)
	{
		DumpModule(mod, "dump.%03u.opt.ll");
	}

	LLVMAddModule(engine, mod);

	func_addr = (ExprStateEvalFunc) LLVMGetPointerToGlobal(
		engine, ExecExpr_f);

	LLVMRemoveModule(engine, mod, &mod, NULL);

	LLVMDisposeBuilder(builder);
	LLVMDisposeModule(mod);

	return func_addr;
}


/*
 * ExecCompileExprLLVM: compile expression with LLVM MCJIT
 *
 * If compilation is successful, `evalfunc` pointer is changed to point to
 * generated code and `true` is returned.
 */
bool
ExecCompileExprLLVM(ExprState *exprstate, ExprContext *econtext)
{
	if (!enable_llvm_jit || !exprstate || !econtext->ecxt_estate->es_engine)
	{
		return false;
	}

	if (IsA(exprstate, List))
	{
		bool changed = false;
		ListCell *cell;

		foreach (cell, (List *) exprstate)
		{
			ExprState *exprstate = lfirst(cell);

			changed |= ExecCompileExprLLVM(exprstate, econtext);
		}

		return changed;
	}

	if (IsA(exprstate, GenericExprState))
	{
		exprstate = ((GenericExprState *) exprstate)->arg;
	}

	if (IsA(exprstate->expr, Var))
		return false;

	if (IsExprSupportedLLVM(exprstate->expr))
	{
		ExprStateEvalFunc evalfunc = CompileExpr(exprstate, econtext);

		if (evalfunc)
		{
			exprstate->evalfunc = evalfunc;
			return true;
		}
	}

	return false;
}
