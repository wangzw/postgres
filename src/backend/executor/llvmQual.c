#include "postgres.h"

#include "llvm_backend/llvm_backend_wrapper.h"
#include "executor/executor.h"
#include "utils/array.h"
#include "utils/lsyscache.h"

#include <llvm-c/Core.h>
#include <llvm-c/ExecutionEngine.h>
#include <llvm-c/Transforms/PassManagerBuilder.h>


typedef struct LLVMTupleAttr {
	LLVMValueRef value;
	LLVMValueRef isnull;
} LLVMTupleAttr;


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
GetTypeByNameInContext(LLVMContextRef context, const char *type_name)
{
	/*
	 * This function is used to surpass the limitation of having to provide
	 * a module reference for `LLVMGetTypeByName`. It uses the fact that
	 * it is the context that LLVM named structs  are internally registered
	 * in, not the module.
	 */
	LLVMModuleRef mod = LLVMModuleCreateWithNameInContext("", context);
	LLVMTypeRef type = LLVMGetTypeByName(mod, type_name);
	LLVMDisposeModule(mod);
	return type;
}


static LLVMTypeRef
GetTypeByName(const char *type_name)
{
	return GetTypeByNameInContext(LLVMGetGlobalContext(), type_name);
}


static LLVMTypeRef
AddStructTypeIfNotExists(const char *name, LLVMTypeRef *types, int ntypes)
{
	LLVMTypeRef struct_type = GetTypeByName(name);

	if (!struct_type)
	{
		struct_type = LLVMStructCreateNamed(LLVMGetGlobalContext(), name);
		LLVMStructSetBody(struct_type, types, ntypes, 0);
	}

	return struct_type;
}


static LLVMTypeRef
ReturnSetInfoType(void)
{
	LLVMTypeRef fields[] = {
		LLVMInt32Type(),
		LLVMPointerType(LLVMInt8Type(), 0),
		LLVMPointerType(LLVMInt8Type(), 0),
		LLVMInt32Type(),
		LLVMInt32Type(),
		LLVMInt32Type(),
		LLVMPointerType(LLVMInt8Type(), 0),
		LLVMPointerType(LLVMInt8Type(), 0),
	};

	return AddStructTypeIfNotExists(
		"ReturnSetInfoLLVM", fields, lengthof(fields));
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


static LLVMTypeRef
FunctionCallInfoType(LLVMBuilderRef builder)
{
	LLVMTypeRef fcinfo_type;

	LLVMBasicBlockRef this_bb = LLVMGetInsertBlock(builder);
	LLVMValueRef this_function = LLVMGetBasicBlockParent(this_bb);
	LLVMModuleRef mod = LLVMGetGlobalParent(this_function);
	/*
	 * Hack: by defining(generating) a function(dpi) whose argument is
	 * the type of FunctionCallInfo structure we can get FunctionCallInfo
	 * as LLVMTypeRef.
	 * dpi (1610) - returns the constant PI
	 */
	LLVMValueRef functionRef = define_llvm_function(1610, mod);
	LLVMTypeRef function_type = LLVMGetElementType(
			LLVMTypeOf(functionRef));

	LLVMGetParamTypes(function_type, &fcinfo_type);

	return LLVMGetElementType(fcinfo_type);
}


static LLVMValueRef
BuildAllocaInBlock(LLVMBuilderRef builder, LLVMBasicBlockRef block,
				   LLVMTypeRef type, const char *name)
{
	LLVMBasicBlockRef current_block = LLVMGetInsertBlock(builder);

	if (block != current_block)
	{
		LLVMValueRef terminator = LLVMGetBasicBlockTerminator(block);
		LLVMValueRef alloca;

		LLVMPositionBuilderBefore(builder, terminator);
		alloca = LLVMBuildAlloca(builder, type, name);
		LLVMPositionBuilderAtEnd(builder, current_block);
		return alloca;
	}
	else
	{
		return LLVMBuildAlloca(builder, type, name);
	}
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

	fcinfo_type = FunctionCallInfoType(builder);

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

		fcinfo_type = FunctionCallInfoType(builder);
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
							  LLVMValueRef fcinfo_llvm, LLVMTupleAttr *attr)
{

	LLVMTupleAttr result;

	LLVMValueRef isnull_ptr;

	LLVMValueRef functionRef = define_llvm_pg_function(
		builder, fcinfo->flinfo);
	LLVMValueRef flinfo_ptr = LLVMBuildStructGEP(
			builder, fcinfo_llvm, 0, "flinfo_ptr");
	LLVMValueRef args = LLVMBuildStructGEP(builder, fcinfo_llvm, 6, "args");
	LLVMValueRef argnulls = LLVMBuildStructGEP(
		builder, fcinfo_llvm, 7, "argnulls");
	LLVMTypeRef fcinfo_type;
	int arg_index;

	for (arg_index = 0; arg_index < fcinfo->nargs; ++arg_index)
	{
		LLVMValueRef arg_ptr = LLVMBuildStructGEP(
				builder, args, arg_index, "arg_ptr");
		LLVMValueRef argnull_ptr = LLVMBuildStructGEP(
				builder, argnulls, arg_index, "argnull_ptr");

		LLVMBuildStore(builder, attr[arg_index].value, arg_ptr);
		LLVMBuildStore(builder, attr[arg_index].isnull, argnull_ptr);
	}

	isnull_ptr = LLVMBuildStructGEP(builder, fcinfo_llvm, 4, "isnull_ptr");
	LLVMBuildStore(builder, LLVMConstNull(LLVMInt8Type()), isnull_ptr);

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

	LLVMGetParamTypes(LLVMGetElementType(LLVMTypeOf(functionRef)),
					  &fcinfo_type);
	fcinfo_llvm = LLVMBuildBitCast(builder, fcinfo_llvm, fcinfo_type, "");

	result.value = LLVMBuildCall(
		builder, functionRef, &fcinfo_llvm, 1,
		get_func_name(fcinfo->flinfo->fn_oid));
	result.isnull = LLVMBuildLoad(builder, isnull_ptr, "isnull");

	return result;
}


static void
FCInfoLLVMAddRetSet(LLVMBuilderRef builder, ExprContext* econtext,
					TupleDesc expectedDesc, LLVMValueRef fcinfo_llvm)
{
	LLVMValueRef resultinfo_ptr =
		LLVMBuildStructGEP(builder, fcinfo_llvm, 2, "resultinfo_ptr");
	LLVMTypeRef rsinfoType = ReturnSetInfoType();
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
		case T_BoolExpr:
		case T_CaseExpr:
		case T_NullTest:
		case T_Aggref:
		case T_ScalarArrayOpExpr:
			return true;

		default:
			return false;
	}
}


static LLVMTupleAttr
GenerateExpr(LLVMBuilderRef builder,
			 ExprState *exprstate,
			 ExprContext *econtext,
			 LLVMBasicBlockRef entry_bb,
			 LLVMValueRef fcinfo_llvm)
{
	switch (nodeTag(exprstate->expr))
	{
		case T_Const:
		{
			Const  *con = (Const *) exprstate->expr;
			LLVMTupleAttr attr;

			attr.isnull = LLVMConstInt(LLVMInt8Type(), con->constisnull, 0);
			attr.value = con->constisnull
				? LLVMConstNull(LLVMInt64Type())
				: LLVMConstInt(LLVMInt64Type(), con->constvalue, 0);
			return attr;
		}

		case T_RelabelType:
		{
			GenericExprState *gstate = (GenericExprState *) exprstate;
			ExprState *argstate = gstate->arg;
			return GenerateExpr(
				builder, argstate, econtext, entry_bb, fcinfo_llvm);
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
			LLVMTupleAttr attr[natts], result;

			LLVMBasicBlockRef this_bb = LLVMGetInsertBlock(builder);
			LLVMValueRef this_function = LLVMGetBasicBlockParent(this_bb);
			LLVMModuleRef mod = LLVMGetGlobalParent(this_function);

			/* Allocate workspace */
			LLVMValueRef values_llvm = LLVMAddGlobalWithPrefix(
				mod, LLVMArrayType(LLVMInt64Type(), natts),
				"RowExpr_values");
			LLVMValueRef isnull_llvm = LLVMAddGlobalWithPrefix(
				mod, LLVMArrayType(LLVMInt8Type(), natts),
				"RowExpr_isnull");

			/* preset to nulls in case rowtype has some later-added columns */
			for (i = 0; i < natts; i++)
			{
				list[i] = LLVMConstInt(LLVMInt8Type(), 1, 0);
			}

			LLVMSetInitializer(values_llvm,
							   LLVMConstNull(
								   LLVMArrayType(LLVMInt64Type(), natts)));
			LLVMSetInitializer(isnull_llvm,
							   LLVMConstArray(LLVMInt8Type(), list, natts));
			LLVMSetLinkage(values_llvm, LLVMInternalLinkage);
			LLVMSetLinkage(isnull_llvm, LLVMInternalLinkage);

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
					builder, argstate, econtext, entry_bb, fcinfo_llvm);

				value = LLVMBuildInBoundsGEP(builder, values_llvm,
											 index, 2, "values[i]");
				null = LLVMBuildInBoundsGEP(builder, isnull_llvm,
											index, 2, "isnull[i]");
				LLVMBuildStore(builder, attr[i].value, value);
				LLVMBuildStore(builder, attr[i].isnull, null);

				i++;
			}

			/* heap_form_tuple */
			heap_form_tuple_args[0] = LLVMBuildLoad(
				builder, rstate_tupdesc_ptr, "");
			heap_form_tuple_args[1] = values_llvm;
			heap_form_tuple_args[2] = isnull_llvm;
			tuple = GenerateCallBackendWithTypeCheck(
				builder, define_heap_form_tuple, heap_form_tuple_args, 3);

			/* HeapTupleGetDatum */
			t_data = LLVMBuildStructGEP(builder, tuple, 3, "t_data_ptr");
			t_data = LLVMBuildLoad(builder, t_data, "t_data");
			result.value = GenerateCallBackendWithTypeCheck(
				builder, define_HeapTupleHeaderGetDatum, &t_data, 1);
			result.isnull = LLVMConstInt(LLVMInt8Type(), 0, 0);

			return result;
		}

		case T_OpExpr:
		case T_FuncExpr:
		{
			FuncExprState *fexprstate = (FuncExprState *) exprstate;
			FunctionCallInfo fcinfo = &fexprstate->fcinfo_data;
			Oid funcid = 0;
			Oid inputcollid = 0;
			bool strict;
			LLVMTupleAttr result, func_result;
			LLVMTupleAttr attr[list_length(fexprstate->args)];
			ListCell* cell;
			short i;

			LLVMBasicBlockRef this_bb = LLVMGetInsertBlock(builder);
			LLVMValueRef function = LLVMGetBasicBlockParent(this_bb);
			LLVMBasicBlockRef exit_bb = 0;  /* -Wmaybe-uninitialized */

			if (IsA(exprstate->expr, OpExpr))
			{
				OpExpr *op = (OpExpr *) exprstate->expr;
				funcid = op->opfuncid;
				inputcollid = op->inputcollid;
			}
			else if (IsA(exprstate->expr, FuncExpr))
			{
				FuncExpr *func = (FuncExpr *) exprstate->expr;
				funcid = func->funcid;
				inputcollid = func->inputcollid;
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
				result.isnull = LLVMBuildPhi(
					builder, LLVMInt8Type(), "isnull");

				LLVMPositionBuilderAtEnd(builder, this_bb);
			}

			i = 0;
			foreach (cell, fexprstate->args)
			{
				ExprState *argstate = lfirst(cell);
				LLVMTupleAttr arg = GenerateExpr(
					builder, argstate, econtext, entry_bb, fcinfo_llvm);

				attr[i] = arg;
				i++;

				if (strict)
				{
					LLVMBasicBlockRef next_bb = LLVMAppendBasicBlock(
						function, "FuncExpr_next");
					LLVMValueRef null_llvm = LLVMConstNull(LLVMInt64Type());
					LLVMValueRef true_llvm = LLVMConstInt(
						LLVMInt8Type(), 1, 0);
					LLVMValueRef isnull;

					this_bb = LLVMGetInsertBlock(builder);
					LLVMAddIncoming(result.value, &null_llvm, &this_bb, 1);
					LLVMAddIncoming(result.isnull, &true_llvm, &this_bb, 1);
					isnull = LLVMBuildIsNotNull(
						builder, arg.isnull, "isnull");
					LLVMBuildCondBr(builder, isnull, exit_bb, next_bb);

					LLVMPositionBuilderAtEnd(builder, next_bb);
					LLVMMoveBasicBlockBefore(next_bb, exit_bb);
					this_bb = next_bb;
				}
			}

			fcinfo_llvm = GenerateInitFCInfo(builder, fcinfo, fcinfo_llvm);
			if (fexprstate->func.fn_retset)
				FCInfoLLVMAddRetSet(
					builder, econtext, fexprstate->funcResultDesc,
					fcinfo_llvm);
			func_result = GenerateFunctionCallNCollNull(
				builder, fcinfo, fcinfo_llvm, attr);

			if (strict)
			{
				LLVMAddIncoming(result.value, &func_result.value,
								&this_bb, 1);
				LLVMAddIncoming(result.isnull, &func_result.isnull,
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
			LLVMTupleAttr result;

			if (boolexpr->boolop == NOT_EXPR)
			{
				ExprState *argstate = linitial(bexprstate->args);

				Assert(boolexpr->args->length == 1);

				result = GenerateExpr(
					builder, argstate, econtext, entry_bb, fcinfo_llvm);

				result.value = LLVMBuildIsNull(builder, result.value, "!val");
				result.value = LLVMBuildZExt(
					builder, result.value, LLVMInt64Type(), "!val");
				result.isnull = LLVMConstInt(LLVMInt8Type(), 0, 0);
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
			result.isnull = LLVMBuildPhi(builder, LLVMInt1Type(), "isnull");

			LLVMPositionBuilderAtEnd(builder, this_bb);

			foreach (cell, bexprstate->args)
			{
				ExprState *argstate = lfirst(cell);
				LLVMTupleAttr arg = GenerateExpr(
					builder, argstate, econtext, entry_bb, fcinfo_llvm);
				LLVMBasicBlockRef next_bb = LLVMAppendBasicBlock(
					function, "BoolExpr_next");
				LLVMValueRef not_null, early_exit;

				arg.value = LLVMBuildIsNotNull(builder, arg.value, "value");
				arg.isnull = LLVMBuildIsNotNull(
					builder, arg.isnull, "isnull");

				any_null = any_null
					? LLVMBuildOr(builder, any_null, arg.isnull, "any_null")
					: arg.isnull;

				not_null = LLVMBuildNot(builder, arg.isnull, "!isnull");
				early_exit = LLVMBuildICmp(
					builder, LLVMIntEQ, arg.value, early_value, "early_exit");
				early_exit = LLVMBuildAnd(
					builder, not_null, early_exit, "early_exit");

				this_bb = LLVMGetInsertBlock(builder);
				LLVMAddIncoming(result.value, &early_value, &this_bb, 1);
				LLVMAddIncoming(result.isnull, &zero, &this_bb, 1);
				LLVMBuildCondBr(builder, early_exit, exit_bb, next_bb);

				LLVMMoveBasicBlockBefore(next_bb, exit_bb);
				LLVMPositionBuilderAtEnd(builder, next_bb);
				this_bb = next_bb;
			}

			LLVMAddIncoming(result.value, &late_value, &this_bb, 1);
			LLVMAddIncoming(
				result.isnull, any_null ? &any_null : &zero, &this_bb, 1);
			LLVMBuildBr(builder, exit_bb);

			LLVMPositionBuilderAtEnd(builder, exit_bb);
			result.value = LLVMBuildZExt(
				builder, result.value, LLVMInt64Type(),
				boolexpr->boolop == AND_EXPR ? "and" : "or");
			result.isnull = LLVMBuildZExt(
				builder, result.isnull, LLVMInt8Type(), "isnull");
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
			LLVMTupleAttr result;

			if (caseExpr->arg)
				Assert(false);

			LLVMPositionBuilderAtEnd(builder, done);
			result.value = LLVMBuildPhi(
				builder, LLVMInt64Type(), "case_result");
			result.isnull = LLVMBuildPhi(
				builder, LLVMInt8Type(), "case_isnull");

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
					builder, casewhen->expr, econtext, entry_bb, fcinfo_llvm);
				istrue = LLVMBuildIsNotNull(
					builder, clause_value.value, "value");
				isnotnull = LLVMBuildIsNull(
					builder, clause_value.isnull, "isnotnull");
				cmp = LLVMBuildAnd(builder, istrue, isnotnull, "cmp");
				LLVMBuildCondBr(builder, cmp, isTrue_bb, isFalse_bb);

				/*
				 * isTrue
				 */
				LLVMPositionBuilderAtEnd(builder, isTrue_bb);
				clause_result = GenerateExpr(
					builder, casewhen->result, econtext, entry_bb,
					fcinfo_llvm);
				current_bb = LLVMGetInsertBlock(builder);
				LLVMAddIncoming(result.value, &clause_result.value,
								&current_bb, 1);
				LLVMAddIncoming(result.isnull, &clause_result.isnull,
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
					builder, caseExpr->defresult, econtext, entry_bb,
					fcinfo_llvm);
				LLVMBasicBlockRef current_bb = LLVMGetInsertBlock(builder);

				LLVMAddIncoming(result.value, &defresult.value,
								&current_bb, 1);
				LLVMAddIncoming(result.isnull, &defresult.isnull,
								&current_bb, 1);
				LLVMBuildBr(builder, done);
			}
			else
			{
				LLVMBasicBlockRef current_bb = LLVMGetInsertBlock(builder);
				LLVMValueRef llvm_null = LLVMConstInt(LLVMInt64Type(), 0, 0);
				LLVMValueRef llvm_true = LLVMConstInt(LLVMInt8Type(), 1, 0);

				LLVMAddIncoming(result.value, &llvm_null, &current_bb, 1);
				LLVMAddIncoming(result.isnull, &llvm_true, &current_bb, 1);
				LLVMBuildBr(builder, done);
			}

			LLVMPositionBuilderAtEnd(builder, done);
			return result;
		}

		case T_NullTest:
		{
			NullTestState *nstate = (NullTestState *) exprstate;
			NullTest *ntest = (NullTest *) nstate->xprstate.expr;
			LLVMValueRef isnull;
			LLVMTupleAttr result;

			Assert(!ntest->argisrow);

			/*
			 * entry
			 */
			result = GenerateExpr(
				builder, nstate->arg, econtext, entry_bb, fcinfo_llvm);
			isnull = LLVMBuildIsNotNull(builder, result.isnull, "isnull");

			switch (ntest->nulltesttype)
			{
				case IS_NULL:
					result.value = LLVMBuildZExt(
						builder, isnull, LLVMInt64Type(), "isnull");
					break;

				case IS_NOT_NULL:
					result.value = LLVMBuildZExt(
						builder, LLVMBuildNot(builder, isnull, "notnull"),
						LLVMInt64Type(), "notnull");
					break;

				default:
					elog(ERROR, "unrecognized nulltesttype: %d",
						 (int) ntest->nulltesttype);
			}

			result.isnull = LLVMConstNull(LLVMInt8Type());
			return result;
		}

		case T_Aggref:
		{
			AggrefExprState *aggref = (AggrefExprState *) exprstate;
			LLVMTupleAttr result;
			LLVMValueRef aggvalue_ptr = ConstPointer(
				LLVMPointerType(LLVMInt64Type(), 0),
				&econtext->ecxt_aggvalues[aggref->aggno]);
			LLVMValueRef aggnull_ptr = ConstPointer(
				LLVMPointerType(LLVMInt8Type(), 0),
				&econtext->ecxt_aggnulls[aggref->aggno]);

			Assert(econtext->ecxt_aggvalues);

			result.value = LLVMBuildLoad(builder, aggvalue_ptr, "aggvalue");
			result.isnull = LLVMBuildLoad(builder, aggnull_ptr, "aggnull");
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
				LLVMConstInt(LLVMInt8Type(), 1, 0)
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
			LLVMTupleAttr scalar, result;

			init_fcache(opexpr->opfuncid, opexpr->inputcollid,
						&sstate->fxprstate, econtext->ecxt_per_query_memory,
						true);
			Assert(!sstate->fxprstate.func.fn_retset);
			Assert(IsA(arrayexpr, Const));

			if (arrayexpr->constisnull)
			{
				return null;
			}

			array = DatumGetArrayTypeP(arrayexpr->constvalue);
			nitems = ArrayGetNItems(ARR_NDIM(array), ARR_DIMS(array));

			if (nitems <= 0)
			{
				result.isnull = LLVMConstNull(LLVMInt8Type());
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
				entry_bb, fcinfo_llvm);

			this_bb = LLVMGetInsertBlock(builder);
			exit_bb = LLVMAppendBasicBlock(
				function, "ScalarArrayOpExpr_exit");

			/*
			 * Create result phis.
			 */
			LLVMPositionBuilderAtEnd(builder, exit_bb);
			result.value = LLVMBuildPhi(
				builder, LLVMInt64Type(), "result.value");
			result.isnull = LLVMBuildPhi(
				builder, LLVMInt8Type(), "result.isnull");
			LLVMPositionBuilderAtEnd(builder, this_bb);

			/*
			 * entry - Return NULL if the scalar is NULL.
			 */
			if (sstate->fxprstate.func.fn_strict)
			{
				LLVMBasicBlockRef loop_bb = LLVMAppendBasicBlock(
					function, "ScalarArrayOpExpr_loop");
				LLVMValueRef isnull = LLVMBuildIsNotNull(
					builder, scalar.isnull, "scalar_isnull");

				LLVMMoveBasicBlockAfter(loop_bb, this_bb);

				LLVMAddIncoming(result.value, &null.value, &this_bb, 1);
				LLVMAddIncoming(result.isnull, &null.isnull, &this_bb, 1);
				LLVMBuildCondBr(builder, isnull, exit_bb, loop_bb);

				LLVMPositionBuilderAtEnd(builder, loop_bb);
				this_bb = loop_bb;
			}

			attr[0] = scalar;
			any_null = LLVMConstNull(LLVMInt1Type());

			/* Loop over the array elements */
			for (itemno = 0; itemno < nitems; ++itemno)
			{
				Datum elt_datum;
				bool elt_isnull;

				/* Get array element, checking for NULL */
				if (bitmap && (*bitmap & bitmask) == 0)
				{
					elt_datum = 0;
					elt_isnull = true;
				}
				else
				{
					elt_datum = fetch_att(
						s, sstate->typbyval, sstate->typlen);
					s = att_addlength_pointer(s, sstate->typlen, s);
					s = (char *) att_align_nominal(s, sstate->typalign);

					elt_isnull = false;
				}

				/* Call comparison function */
				if (elt_isnull && sstate->fxprstate.func.fn_strict)
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
					LLVMValueRef isnull, early_exit, phi;
					LLVMTupleAttr thisresult;

					LLVMMoveBasicBlockAfter(next_bb, this_bb);
					LLVMMoveBasicBlockAfter(checkresult_bb, this_bb);

					Assert(fcinfo->nargs == 2);

					attr[1].value = LLVMConstInt(
						LLVMInt64Type(), elt_datum, 0);
					attr[1].isnull = LLVMConstInt(
						LLVMInt8Type(), elt_isnull, 0);

					fcinfo_llvm = GenerateInitFCInfo(
						builder, fcinfo, fcinfo_llvm);
					thisresult = GenerateFunctionCallNCollNull(
						builder, fcinfo, fcinfo_llvm, attr);
					isnull = LLVMBuildIsNotNull(
						builder, thisresult.isnull, "thisresult.isnull");
					LLVMBuildCondBr(builder, isnull, next_bb, checkresult_bb);

					/*
					 * checkresult
					 */
					LLVMPositionBuilderAtEnd(builder, checkresult_bb);
					early_exit = LLVMBuildICmp(
						builder, LLVMIntEQ, thisresult.value,
						early_result.value, "early_exit");
					LLVMAddIncoming(result.value, &early_result.value,
									&checkresult_bb, 1);
					LLVMAddIncoming(result.isnull, &early_result.isnull,
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
			LLVMAddIncoming(result.isnull, &any_null, &this_bb, 1);
			LLVMBuildBr(builder, exit_bb);

			/*
			 * exit
			 */
			LLVMPositionBuilderAtEnd(builder, exit_bb);
			return result;
		}

		default:
		{
			LLVMValueRef evalfunc_ptr = ConstPointer(
				LLVMPointerType(
					LLVMPointerType(ExprStateEvalFuncType(), 0), 0),
				&exprstate->evalfunc);
			LLVMValueRef evalfunc = LLVMBuildLoad(
				builder, evalfunc_ptr, "evalfunc");
			LLVMValueRef isnull_ptr = BuildAllocaInBlock(
				builder, entry_bb, LLVMInt8Type(), "isnull_ptr");
			LLVMValueRef args[] = {
				ConstPointer(LLVMPointerType(LLVMInt8Type(), 0), exprstate),
				ConstPointer(LLVMPointerType(LLVMInt8Type(), 0), econtext),
				isnull_ptr,
				LLVMConstNull(LLVMPointerType(LLVMInt32Type(), 0))
			};
			LLVMTupleAttr result;

			result.value = LLVMBuildCall(
				builder, evalfunc, args, lengthof(args), "value");
			result.isnull = LLVMBuildLoad(builder, isnull_ptr, "isnull");
			return result;
		}
	}
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


static ExprStateEvalFunc
CompileExpr(ExprState *exprstate, ExprContext *econtext)
{
	LLVMModuleRef mod = InitModule("expr");
	LLVMExecutionEngineRef engine = CreateCompiler(mod);
	LLVMValueRef ExecExpr_f = LLVMAddFunctionWithPrefix(
		mod, "ExecExpr", ExprStateEvalFuncType());
	LLVMBasicBlockRef entry_bb = LLVMAppendBasicBlock(ExecExpr_f, "entry");
	LLVMBuilderRef builder = LLVMCreateBuilder();
	LLVMValueRef fcinfo;

	LLVMPositionBuilderAtEnd(builder, entry_bb);
	fcinfo = GenerateAllocFCInfo(builder);

	{
		LLVMTupleAttr result = GenerateExpr(
			builder, exprstate, econtext, entry_bb, fcinfo);
		LLVMValueRef isnull_ptr = LLVMGetParam(ExecExpr_f, 2);

		LLVMBuildStore(builder, result.isnull, isnull_ptr);
		LLVMBuildRet(builder, result.value);
	}

#ifdef LLVM_DUMP
	LLVMPrintModuleToFile(mod, "dump.ll", NULL);
#endif

	LLVMSetFunctionCallConv(ExecExpr_f, LLVMCCallConv);
	RunPasses(engine, mod);

#ifdef LLVM_DUMP
	LLVMPrintModuleToFile(mod, "dump.opt.ll", NULL);
#endif

	LLVMDisposeBuilder(builder);
	return (ExprStateEvalFunc) LLVMGetFunctionAddress(
		engine, LLVMGetValueName(ExecExpr_f));
}


/*
 * ExecCompileExpr: compile expression with LLVM MCJIT.
 */
ExprStateEvalFunc
ExecCompileExpr(ExprState *exprstate, ExprContext *econtext)
{
	return IsExprSupported(exprstate)
		? CompileExpr(exprstate, econtext)
		: NULL;
}
