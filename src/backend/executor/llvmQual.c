#include "postgres.h"

#include "llvm_backend/llvm_backend_wrapper.h"
#include "access/htup_details.h"
#include "executor/executor.h"
#include "nodes/nodeFuncs.h"
#include "nodes/print.h"
#include "optimizer/planmain.h"
#include "utils/array.h"
#include "utils/lsyscache.h"

#include <llvm-c/Core.h>
#include <llvm-c/ExecutionEngine.h>
#include <llvm-c/Transforms/PassManagerBuilder.h>

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
		LLVMConstInt(LLVMInt32Type(), ExprSingleResult, 0)}


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
		LLVMConstInt(LLVMInt64Type(), (uintptr_t) pointer, 0), pointer_type);
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
		LLVMVoidType(), memset_arg_types, lengthof(memset_arg_types), 0);
	LLVMValueRef memset_f = AddLLVMIntrinsic(
		builder, "llvm.memset.p0i8.i64", memset_type);
	LLVMValueRef args[5];

	args[0] = LLVMBuildPointerCast(
		builder, dest, LLVMPointerType(LLVMInt8Type(), 0), "dest");
	args[1] = LLVMBuildTrunc(builder, val, LLVMInt8Type(), "val");
	args[2] = LLVMBuildZExt(builder, len, LLVMInt64Type(), "len");
	args[3] = LLVMConstInt(LLVMInt32Type(), alignment, 0);
	args[4] = LLVMConstNull(LLVMInt1Type());
	LLVMBuildCall(builder, memset_f, args, lengthof(args), "");
}


static LLVMTypeRef
BackendStructType(LLVMTypeRef (*define_func)(LLVMModuleRef))
{
	LLVMModuleRef mod = LLVMModuleCreateWithName("");
	LLVMTypeRef type = define_func(mod);
	LLVMDisposeModule(mod);
	return type;
}


static LLVMTypeRef
ExprStateEvalFuncType(void)
{
	LLVMTypeRef arg_types[] = {
		LLVMPointerType(LLVMInt8Type(), 0),
		LLVMPointerType(LLVMInt8Type(), 0),
		LLVMPointerType(LLVMInt8Type(), 0),
		LLVMPointerType(LLVMInt32Type(), 0)
	};
	LLVMTypeRef function_type = LLVMFunctionType(
		LLVMInt64Type(), arg_types, lengthof(arg_types), 0);

	StaticAssertStmt(sizeof(bool) == sizeof(int8), "bool is 8-bit");
	StaticAssertStmt(sizeof(ExprDoneCond) == sizeof(int32),
					 "ExprDoneCond is 32-bit");

	return function_type;
}


static LLVMValueRef
BuildLoadWithOffset(LLVMBuilderRef builder, LLVMValueRef pointer,
					long offset, LLVMTypeRef base_type, const char *name)
{
	LLVMValueRef gep_index = LLVMConstInt(LLVMInt64Type(), offset, 1);
	LLVMValueRef value_ptr, value;

	pointer = LLVMBuildPointerCast(
		builder, pointer, LLVMPointerType(LLVMInt8Type(), 0),
		LLVMGetValueName(pointer));
	value_ptr = LLVMBuildPointerCast(
		builder, LLVMBuildGEP(builder, pointer, &gep_index, 1, name),
		LLVMPointerType(base_type, 0), name);
	value = LLVMBuildLoad(builder, value_ptr, name);

	return value;
}


static LLVMValueRef
GenerateCallBackendWithTypeCheck(LLVMBuilderRef builder,
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
			builder, args[i], args_types[i], "arg_bitcast");
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
		LLVMBuildStructGEP(builder, fcinfo_llvm, 1, "context_ptr");
	resultinfo_ptr =
		LLVMBuildStructGEP(builder, fcinfo_llvm, 2, "resultinfo_ptr");
	fncollation_ptr =
		LLVMBuildStructGEP(builder, fcinfo_llvm, 3, "fncollation_ptr");
	nargs_ptr =
		LLVMBuildStructGEP(builder, fcinfo_llvm, 5, "nargs_ptr");

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
			LLVMInt32Type(), fcinfo->fncollation, 0), fncollation_ptr);
	LLVMBuildStore(builder, LLVMConstInt(
			LLVMInt16Type(), fcinfo->nargs, 0), nargs_ptr);

	/*
	 * Zero-initialize `argnull`.
	 */
	StaticAssertStmt(sizeof(bool) == sizeof(int8_t), "bool is 8-bit");
	argnulls = LLVMBuildStructGEP(builder, fcinfo_llvm, 7, "argnull_ptr");
	argnulls = LLVMBuildStructGEP(builder, argnulls, 0, "argnull_ptr");
	GenerateMemSet(builder, 1, argnulls,
				   LLVMConstNull(LLVMInt8Type()),
				   LLVMConstInt(LLVMInt32Type(), fcinfo->nargs, 0));

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

	fcinfo_type = BackendStructType(define_struct_FunctionCallInfoData);

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
		char func_name[16];
		LLVMTypeRef fcinfo_type, function_type_llvm;

		fcinfo_type = BackendStructType(define_struct_FunctionCallInfoData);
		fcinfo_type = LLVMPointerType(fcinfo_type, 0);
		function_type_llvm = LLVMFunctionType(
				LLVMInt64Type(), &fcinfo_type, 1, 0);

		snprintf(func_name, sizeof(func_name), "%d", flinfo->fn_oid);

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
							  bool retSet, bool hasSetArg)
{

	LLVMTupleAttr result;

	LLVMValueRef isNull_ptr;

	LLVMValueRef functionRef = define_llvm_pg_function(
		builder, fcinfo->flinfo);
	LLVMValueRef flinfo_ptr = LLVMBuildStructGEP(
			builder, fcinfo_llvm, 0, "flinfo_ptr");
	LLVMValueRef args = LLVMBuildStructGEP(builder, fcinfo_llvm, 6, "args");
	LLVMValueRef argnulls = LLVMBuildStructGEP(
		builder, fcinfo_llvm, 7, "argnulls");
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

	isNull_ptr = LLVMBuildStructGEP(builder, fcinfo_llvm, 4, "isNull_ptr");
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
			LLVMBuildStructGEP(builder, fcinfo_llvm, 2, "resultinfo_ptr");
		resultinfo_ptr = LLVMBuildBitCast(builder,
			LLVMBuildLoad(builder, resultinfo_ptr, ""),
			LLVMPointerType(BackendStructType(define_struct_ReturnSetInfo),
							0), "");
		rsinfo_isDone_ptr =
			LLVMBuildStructGEP(builder, resultinfo_ptr, 5, "&isDone");
	}

	LLVMGetParamTypes(LLVMGetElementType(LLVMTypeOf(functionRef)),
			&fcinfo_type);
	fcinfo_llvm = LLVMBuildBitCast(builder, fcinfo_llvm, fcinfo_type, "");

	result.value = LLVMBuildCall(
		builder, functionRef, &fcinfo_llvm, 1,
		get_func_name(fcinfo->flinfo->fn_oid));
	result.isNull = LLVMBuildLoad(builder, isNull_ptr, "isNull");

	if (retSet)
		result.isDone = LLVMBuildLoad(builder, rsinfo_isDone_ptr, "isDone");
	else
		if (hasSetArg)
		{
			for (arg_index = 0; arg_index < fcinfo->nargs; ++arg_index)
			{
				LLVMValueRef attr_isDone = LLVMBuildICmp(
						builder, LLVMIntEQ, attr[arg_index].isDone,
						LLVMConstInt(LLVMInt32Type(), ExprSingleResult, 0), "attr_isDone");
				result.isDone = LLVMBuildSelect(builder, attr_isDone,
						result.isDone, attr[arg_index].isDone, "select_isDone");
			}
		}
		else
			result.isDone = LLVMConstInt(
					LLVMInt32Type(), ExprSingleResult, 0);

	return result;
}


static void
FCInfoLLVMAddRetSet(LLVMBuilderRef builder, ExprContext* econtext,
					TupleDesc expectedDesc, LLVMValueRef fcinfo_llvm)
{
	LLVMValueRef resultinfo_ptr =
		LLVMBuildStructGEP(builder, fcinfo_llvm, 2, "resultinfo_ptr");
	LLVMTypeRef rsinfoType = BackendStructType(define_struct_ReturnSetInfo);
	LLVMValueRef rsinfo_ptr = LLVMBuildAlloca(builder, rsinfoType, "rsinfo");

	LLVMValueRef rsinfo_type_ptr =
		LLVMBuildStructGEP(builder, rsinfo_ptr, 0, "&rsinfo->type");
	LLVMValueRef rsinfo_econtext_ptr =
		LLVMBuildStructGEP(builder, rsinfo_ptr, 1, "&rsinfo->econtext");
	LLVMValueRef rsinfo_expectedDesc_ptr =
		LLVMBuildStructGEP(builder, rsinfo_ptr, 2, "&rsinfo->expectedDesc");
	LLVMValueRef rsinfo_allowedModes_ptr =
		LLVMBuildStructGEP(builder, rsinfo_ptr, 3, "&rsinfo->allowedModes");
	LLVMValueRef rsinfo_returnMode_ptr =
		LLVMBuildStructGEP(builder, rsinfo_ptr, 4, "&rsinfo->returnMode");
	LLVMValueRef rsinfo_isDone_ptr =
		LLVMBuildStructGEP(builder, rsinfo_ptr, 5, "&rsinfo->isDone");
	LLVMValueRef rsinfo_setResult_ptr =
		LLVMBuildStructGEP(builder, rsinfo_ptr, 6, "&rsinfo->setResult");
	LLVMValueRef rsinfo_setDesc_ptr =
		LLVMBuildStructGEP(builder, rsinfo_ptr, 7, "&rsinfo->setDesc");

	LLVMBuildStore(builder, LLVMConstInt(
		LLVMInt32Type(), T_ReturnSetInfo, 0), rsinfo_type_ptr);
	LLVMBuildStore(builder, ConstPointer(
		LLVMGetElementType(LLVMTypeOf(rsinfo_econtext_ptr)), econtext),
		rsinfo_econtext_ptr);
	LLVMBuildStore(builder, ConstPointer(
		LLVMGetElementType(LLVMTypeOf(rsinfo_expectedDesc_ptr)),
		expectedDesc), rsinfo_expectedDesc_ptr);
	LLVMBuildStore(builder, LLVMConstInt(
		LLVMInt32Type(), (int) (SFRM_ValuePerCall | SFRM_Materialize), 0),
		rsinfo_allowedModes_ptr);
	LLVMBuildStore(builder, LLVMConstInt(
		LLVMInt32Type(), SFRM_ValuePerCall, 0), rsinfo_returnMode_ptr);
	LLVMBuildStore(builder, LLVMConstInt(
				LLVMInt32Type(), ExprSingleResult, 0), rsinfo_isDone_ptr);
	LLVMBuildStore(builder, ConstPointer(
		LLVMGetElementType(LLVMTypeOf(rsinfo_setResult_ptr)), NULL),
		rsinfo_setResult_ptr);
	LLVMBuildStore(builder, ConstPointer(
		LLVMGetElementType(LLVMTypeOf(rsinfo_setDesc_ptr)), NULL),
		rsinfo_setDesc_ptr);
	rsinfo_ptr = LLVMBuildBitCast(builder, rsinfo_ptr,
		LLVMGetElementType(LLVMTypeOf(resultinfo_ptr)), "bitcast");
	LLVMBuildStore(builder, rsinfo_ptr, resultinfo_ptr);
}


static bool
IsExprSupported(ExprState *exprstate)
{
	switch (nodeTag(exprstate->expr))
	{
		case T_Const:
		case T_RelabelType:
		case T_RowExpr:
		case T_OpExpr:
		case T_FuncExpr:
		case T_Aggref:
		case T_BoolExpr:
			return true;

		case T_Var:
		{
			return IsA(exprstate, ExprState);
		}

		case T_CaseExpr:
		{
			CaseExprState *caseExpr = (CaseExprState *) exprstate;

			if (caseExpr->arg)
				return false;

			return true;
		}

		case T_NullTest:
		{
			NullTestState *nstate = (NullTestState *) exprstate;
			NullTest *ntest = (NullTest *) nstate->xprstate.expr;

			if (ntest->argisrow)
				return false;

			return true;
		}

		case T_ScalarArrayOpExpr:
		{
			ScalarArrayOpExpr *opexpr = (ScalarArrayOpExpr *) exprstate->expr;

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
		ConstPointer(LLVMPointerType(LLVMInt8Type(), 0), exprstate),
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
		LLVMConstInt(LLVMInt32Type(), attnum, 0)
	};

	GenerateCallBackendWithTypeCheck(
		builder, define_slot_getsomeattrs, args, lengthof(args));
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
		LLVMValueRef scanSlot = BuildLoadWithOffset(
			builder, rtcontext->econtext,
			offsetof(ExprContext, ecxt_scantuple),
			LLVMPointerType(LLVMInt8Type(), 0), "scanSlot");

		rtcontext->scanSlot = scanSlot;

		if (stats.hasScanSysAttr)
		{
			rtcontext->scanSlotTuple = BuildLoadWithOffset(
				builder, scanSlot, offsetof(TupleTableSlot, tts_tuple),
				LLVMPointerType(LLVMInt8Type(), 0), "scanSlotTuple");
			rtcontext->scanSlotTupleDesc = BuildLoadWithOffset(
				builder, scanSlot,
				offsetof(TupleTableSlot, tts_tupleDescriptor),
				LLVMPointerType(LLVMInt8Type(), 0), "scanSlotTupleDesc");
		}

		if (stats.numScanAttrs > 0)
		{
			GetSomeAttrs(builder, scanSlot, stats.numScanAttrs);

			rtcontext->scanSlotValues = BuildLoadWithOffset(
				builder, scanSlot, offsetof(TupleTableSlot, tts_values),
				LLVMPointerType(LLVMInt64Type(), 0), "scanSlotValues");
			rtcontext->scanSlotIsNull = BuildLoadWithOffset(
				builder, scanSlot, offsetof(TupleTableSlot, tts_isnull),
				LLVMPointerType(LLVMInt8Type(), 0), "scanSlotIsNull");
		}
	}

	if (stats.hasInnerSysAttr || stats.numInnerAttrs > 0)
	{
		LLVMValueRef innerSlot = BuildLoadWithOffset(
			builder, rtcontext->econtext,
			offsetof(ExprContext, ecxt_innertuple),
			LLVMPointerType(LLVMInt8Type(), 0), "innerSlot");

		rtcontext->innerSlot = innerSlot;

		if (stats.hasInnerSysAttr)
		{
			rtcontext->innerSlotTuple = BuildLoadWithOffset(
				builder, innerSlot, offsetof(TupleTableSlot, tts_tuple),
				LLVMPointerType(LLVMInt8Type(), 0), "innerSlotTuple");
			rtcontext->innerSlotTupleDesc = BuildLoadWithOffset(
				builder, innerSlot,
				offsetof(TupleTableSlot, tts_tupleDescriptor),
				LLVMPointerType(LLVMInt8Type(), 0), "innerSlotTupleDesc");
		}

		if (stats.numInnerAttrs > 0)
		{
			GetSomeAttrs(builder, innerSlot, stats.numInnerAttrs);

			rtcontext->innerSlotValues = BuildLoadWithOffset(
				builder, innerSlot, offsetof(TupleTableSlot, tts_values),
				LLVMPointerType(LLVMInt64Type(), 0), "innerSlotValues");
			rtcontext->innerSlotIsNull = BuildLoadWithOffset(
				builder, innerSlot, offsetof(TupleTableSlot, tts_isnull),
				LLVMPointerType(LLVMInt8Type(), 0), "innerSlotIsNull");
		}
	}

	if (stats.hasOuterSysAttr || stats.numOuterAttrs > 0)
	{
		LLVMValueRef outerSlot = BuildLoadWithOffset(
			builder, rtcontext->econtext,
			offsetof(ExprContext, ecxt_outertuple),
			LLVMPointerType(LLVMInt8Type(), 0), "outerSlot");

		rtcontext->outerSlot = outerSlot;

		if (stats.hasOuterSysAttr)
		{
			rtcontext->outerSlotTuple = BuildLoadWithOffset(
				builder, outerSlot, offsetof(TupleTableSlot, tts_tuple),
				LLVMPointerType(LLVMInt8Type(), 0), "outerSlotTuple");
			rtcontext->outerSlotTupleDesc = BuildLoadWithOffset(
				builder, outerSlot,
				offsetof(TupleTableSlot, tts_tupleDescriptor),
				LLVMPointerType(LLVMInt8Type(), 0), "outerSlotTupleDesc");
		}

		if (stats.numOuterAttrs > 0)
		{
			GetSomeAttrs(builder, outerSlot, stats.numOuterAttrs);

			rtcontext->outerSlotValues = BuildLoadWithOffset(
				builder, outerSlot, offsetof(TupleTableSlot, tts_values),
				LLVMPointerType(LLVMInt64Type(), 0), "outerSlotValues");
			rtcontext->outerSlotIsNull = BuildLoadWithOffset(
				builder, outerSlot, offsetof(TupleTableSlot, tts_isnull),
				LLVMPointerType(LLVMInt8Type(), 0), "outerSlotIsNull");
		}
	}
}


static LLVMValueRef
GetSysAttr(LLVMBuilderRef builder, LLVMValueRef tuple, AttrNumber attno,
		   LLVMValueRef tupleDesc, LLVMValueRef isNull)
{
	LLVMValueRef args[] = {
		tuple,
		LLVMConstInt(LLVMInt32Type(), attno, 1),
		tupleDesc,
		isNull
	};

	return GenerateCallBackendWithTypeCheck(
		builder, define_heap_getsysattr, args, lengthof(args));
}


static LLVMTupleAttr
GenerateExpr(LLVMBuilderRef builder,
			 ExprState *exprstate,
			 ExprContext *econtext,
			 RuntimeContext *rtcontext)
{
	if (!IsExprSupported(exprstate))
		return GenerateDefaultExpr(builder, exprstate, rtcontext);

	switch (nodeTag(exprstate->expr))
	{
		case T_Const:
		{
			Const  *con = (Const *) exprstate->expr;
			LLVMTupleAttr attr = INIT_LLVMTUPLEATTR;

			attr.isNull = LLVMConstInt(LLVMInt8Type(), con->constisnull, 0);
			attr.value = con->constisnull
				? LLVMConstNull(LLVMInt64Type())
				: LLVMConstInt(LLVMInt64Type(), con->constvalue, 0);
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
				Assert(slotValues && slotIsNull);

				result.value = BuildLoadWithOffset(
					builder, slotValues, sizeof(Datum) * (attno - 1),
					LLVMInt64Type(), "var_value");
				result.isNull = BuildLoadWithOffset(
					builder, slotIsNull, sizeof(bool) * (attno - 1),
					LLVMInt8Type(), "var_isNull");
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
				list[i] = LLVMConstInt(LLVMInt8Type(), 1, 0);
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
					LLVMConstInt(LLVMInt32Type(), 0, 0),
					LLVMConstInt(LLVMInt32Type(), i, 0)
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
				builder, rstate_tupdesc_ptr, "");
			heap_form_tuple_args[1] = values_llvm;
			heap_form_tuple_args[2] = isNull_llvm;
			tuple = GenerateCallBackendWithTypeCheck(
				builder, define_heap_form_tuple, heap_form_tuple_args, 3);

			/* HeapTupleGetDatum */
			t_data = LLVMBuildStructGEP(builder, tuple, 3, "t_data_ptr");
			t_data = LLVMBuildLoad(builder, t_data, "t_data");
			result.value = GenerateCallBackendWithTypeCheck(
				builder, define_HeapTupleHeaderGetDatum, &t_data, 1);
			result.isNull = LLVMConstInt(LLVMInt8Type(), 0, 0);

			return result;
		}

		case T_OpExpr:
		case T_FuncExpr:
		{
			FuncExprState *fexprstate = (FuncExprState *) exprstate;
			FunctionCallInfo fcinfo = &fexprstate->fcinfo_data;
			Node *args = NULL;
			Oid funcid = 0;
			Oid inputcollid = 0;
			bool strict, retSet, hasSetArg;
			LLVMTupleAttr result, func_result;
			LLVMTupleAttr attr[list_length(fexprstate->args)];
			LLVMValueRef fcinfo_llvm;
			ListCell* cell;
			short i;

			LLVMBasicBlockRef this_bb = LLVMGetInsertBlock(builder);
			LLVMValueRef function = LLVMGetBasicBlockParent(this_bb);
			LLVMBasicBlockRef exit_bb = 0;  /* -Wmaybe-uninitialized */

			if (IsA(exprstate->expr, OpExpr))
			{
				OpExpr *op = (OpExpr *) exprstate->expr;
				args = (Node *)op->args;
				funcid = op->opfuncid;
				inputcollid = op->inputcollid;
			}
			else if (IsA(exprstate->expr, FuncExpr))
			{
				FuncExpr *func_expr = (FuncExpr *) exprstate->expr;
				args = (Node *)func_expr->args;
				funcid = func_expr->funcid;
				inputcollid = func_expr->inputcollid;
			}
			else
			{
				Assert(false);
			}

			init_fcache(funcid, inputcollid, fexprstate,
						econtext->ecxt_per_query_memory, true);
			strict = fexprstate->func.fn_strict;

			if (strict)
			{
				exit_bb = LLVMAppendBasicBlock(function, "FuncExpr_exit");

				LLVMPositionBuilderAtEnd(builder, exit_bb);
				result.value = LLVMBuildPhi(
					builder, LLVMInt64Type(), "value");
				result.isNull = LLVMBuildPhi(
					builder, LLVMInt8Type(), "isNull");
				result.isDone = LLVMBuildPhi(
					builder, LLVMInt32Type(), "isDone");

				LLVMPositionBuilderAtEnd(builder, this_bb);
			}

			i = 0;
			foreach (cell, fexprstate->args)
			{
				ExprState *argstate = lfirst(cell);
				LLVMTupleAttr arg = GenerateExpr(
					builder, argstate, econtext, rtcontext);

				attr[i] = arg;
				i++;

				if (strict)
				{
					LLVMBasicBlockRef next_bb = LLVMAppendBasicBlock(
						function, "FuncExpr_next");
					LLVMValueRef null_llvm = LLVMConstNull(LLVMInt64Type());
					LLVMValueRef true_llvm = LLVMConstInt(
							LLVMInt8Type(), 1, 0);
					LLVMValueRef single_llvm = LLVMConstInt(
							LLVMInt32Type(), ExprSingleResult, 0);
					LLVMValueRef isNull;

					this_bb = LLVMGetInsertBlock(builder);
					LLVMAddIncoming(result.value, &null_llvm, &this_bb, 1);
					LLVMAddIncoming(result.isNull, &true_llvm, &this_bb, 1);
					LLVMAddIncoming(result.isDone, &single_llvm, &this_bb, 1);
					isNull = LLVMBuildIsNotNull(
						builder, arg.isNull, "isNull");
					LLVMBuildCondBr(builder, isNull, exit_bb, next_bb);

					LLVMPositionBuilderAtEnd(builder, next_bb);
					LLVMMoveBasicBlockBefore(next_bb, exit_bb);
					this_bb = next_bb;
				}
			}

			retSet = fexprstate->func.fn_retset;
			hasSetArg = expression_returns_set(args);
			fcinfo_llvm = GenerateInitFCInfo(
				builder, fcinfo, rtcontext->fcinfo);

			if (retSet)
			{
				FCInfoLLVMAddRetSet(
						builder, econtext, fexprstate->funcResultDesc,
						fcinfo_llvm);
			}

			func_result = GenerateFunctionCallNCollNull(
					builder, fcinfo, fcinfo_llvm, attr, retSet, hasSetArg);

			if (strict)
			{
				LLVMAddIncoming(result.value, &func_result.value,
								&this_bb, 1);
				LLVMAddIncoming(result.isNull, &func_result.isNull,
								&this_bb, 1);
				LLVMAddIncoming(result.isDone, &func_result.isDone,
								&this_bb, 1);
				LLVMBuildBr(builder, exit_bb);
				LLVMPositionBuilderAtEnd(builder, exit_bb);
			}
			else
			{
				result = func_result;
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
				LLVMBuildBr(builder, done);
			}
			else
			{
				LLVMBasicBlockRef current_bb = LLVMGetInsertBlock(builder);
				LLVMValueRef llvm_null = LLVMConstInt(LLVMInt64Type(), 0, 0);
				LLVMValueRef llvm_true = LLVMConstInt(LLVMInt8Type(), 1, 0);

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
				LLVMConstInt(LLVMInt8Type(), 1, 0),
				LLVMConstInt(LLVMInt32Type(), ExprSingleResult, 0)
			};
			LLVMTupleAttr early_result = {
				LLVMConstInt(LLVMInt64Type(), useOr, 0),
				LLVMConstNull(LLVMInt8Type())
			};
			LLVMTupleAttr late_result = {
				LLVMConstInt(LLVMInt64Type(), !useOr, 0),
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
				result.value = LLVMConstInt(LLVMInt8Type(), !useOr, 0);
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
					LLVMTupleAttr thisresult;

					LLVMMoveBasicBlockAfter(next_bb, this_bb);
					LLVMMoveBasicBlockAfter(checkresult_bb, this_bb);

					Assert(fcinfo->nargs == 2);

					attr[1].value = LLVMConstInt(
						LLVMInt64Type(), elt_datum, 0);
					attr[1].isNull = LLVMConstInt(
						LLVMInt8Type(), elt_isNull, 0);

					fcinfo_llvm = GenerateInitFCInfo(
						builder, fcinfo, rtcontext->fcinfo);
					thisresult = GenerateFunctionCallNCollNull(
						builder, fcinfo, fcinfo_llvm, attr, false, false);
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


/*
 * CreateCompiler: create execution engine for module.
 */
static LLVMExecutionEngineRef
CreateCompiler(LLVMModuleRef mod)
{
	LLVMExecutionEngineRef engine;
	struct LLVMMCJITCompilerOptions options;
	LLVMTargetDataRef target_data;
	char *error = NULL;

	LLVMInitializeMCJITCompilerOptions(&options, sizeof(options));
	options.OptLevel = 3;
	if (LLVMCreateMCJITCompilerForModule(&engine, mod, &options,
										 sizeof(options), &error) != 0)
	{
		fprintf(stderr, "%s\n", error);
		LLVMDisposeMessage(error);
		abort();
	}

	target_data = LLVMGetExecutionEngineTargetData(engine);
	LLVMSetDataLayout(mod, LLVMCopyStringRepOfTargetData(target_data));
	return engine;
}


static void
isinf_codegen(LLVMModuleRef mod)
{
	LLVMBuilderRef builder = LLVMCreateBuilder();
	LLVMTypeRef isinf_arg_types[] = {
		LLVMDoubleType()
	};
	LLVMTypeRef isinf_type = LLVMFunctionType(
		LLVMInt32Type(), isinf_arg_types, 1, 0);
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
	ret = LLVMBuildZExt(builder, LLVMBuildOr(builder, left, right, ""),
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
 * RunPasses: optimize generated module.
 */
static void
RunPasses(LLVMExecutionEngineRef engine, LLVMModuleRef mod)
{
#ifdef LLVM_DEBUG
	char *error = NULL;
	LLVMVerifyModule(mod, LLVMAbortProcessAction, &error);
	/* Handler == LLVMAbortProcessAction -> No need to check errors */
	LLVMDisposeMessage(error);
	error = NULL;
#else
	LLVMPassManagerBuilderRef pmbuilder = LLVMPassManagerBuilderCreate();
	LLVMPassManagerRef pass = LLVMCreatePassManager();

	LLVMAddTargetData(LLVMGetExecutionEngineTargetData(engine), pass);

	LLVMPassManagerBuilderSetOptLevel(pmbuilder, 2);
	LLVMPassManagerBuilderUseInlinerWithThreshold(pmbuilder, 275);

	LLVMPassManagerBuilderPopulateModulePassManager(pmbuilder, pass);

	LLVMRunPassManager(pass, mod);
	LLVMDisposePassManager(pass);
	LLVMPassManagerBuilderDispose(pmbuilder);
#endif
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
	LLVMExecutionEngineRef engine = CreateCompiler(mod);
	LLVMValueRef ExecExpr_f = LLVMAddFunctionWithPrefix(
		mod, "ExecExpr", ExprStateEvalFuncType());
	LLVMBasicBlockRef entry_bb = LLVMAppendBasicBlock(ExecExpr_f, "entry");
	LLVMBuilderRef builder = LLVMCreateBuilder();

	if (enable_llvm_dump)
	{
		DumpExpression(exprstate->expr, "expr.%03u");
	}

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

	if (enable_llvm_dump)
	{
		DumpModule(mod, "dump.%03u.ll");
	}

	LLVMSetFunctionCallConv(ExecExpr_f, LLVMCCallConv);
	RunPasses(engine, mod);

	if (enable_llvm_dump)
	{
		DumpModule(mod, "dump.%03u.opt.ll");
	}

	func_addr = (ExprStateEvalFunc) LLVMGetFunctionAddress(
		engine, LLVMGetValueName(ExecExpr_f));

	LLVMDisposeBuilder(builder);
	LLVMDisposeModule(mod);

	return func_addr;
}


/*
 * ExecCompileExpr: compile expression with LLVM MCJIT
 *
 * If compilation is successful, `evalfunc` pointer is changed to point to
 * generated code and `true` is returned.
 */
bool
ExecCompileExpr(ExprState *exprstate, ExprContext *econtext)
{
	if (!enable_llvm_jit || !exprstate)
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

			changed |= ExecCompileExpr(exprstate, econtext);
		}

		return changed;
	}

	if (IsA(exprstate, GenericExprState))
	{
		exprstate = ((GenericExprState *) exprstate)->arg;
	}

	if (IsExprSupported(exprstate))
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
