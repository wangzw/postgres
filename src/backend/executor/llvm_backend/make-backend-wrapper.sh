#!/bin/bash
# -*- mode: shell-script; indent-tabs-mode: t; tab-width: 4; -*-
set -euo pipefail
IFS=$'\n'

PG_SRC=$1
shift

BACKEND_H=llvm_backend.h
BACKEND_WRAPPER_H=llvm_backend_wrapper.h
BACKEND_WRAPPER_CPP=llvm_backend_wrapper.cpp

EXTRA_FUNC_LIST="\
HeapTupleHeaderGetDatum
heap_form_tuple
heap_getsysattr
slot_getsomeattrs"

FUNCTION_LIST=$(\
	grep --only-matching --no-filename --perl-regexp \
		 '(?<=Function\* ).*(?=\(Module \*mod\) \{)' $@ |
		sort)

TYPE_LIST=$(\
	grep --only-matching --no-filename --perl-regexp \
		 '(?<=Type\* ).*(?=\(Module \*mod\) \{)' $@ |
		sort)

sed -i "1i#include \"$(basename $BACKEND_H)\"\n" $@

#
# Generate BACKEND_WRAPPER_H.
#
cat > $BACKEND_WRAPPER_H <<-EOF
	#ifndef LLVM_BACKEND_WRAPPER_H
	#define LLVM_BACKEND_WRAPPER_H

	#include <llvm-c/Core.h>

	LLVMValueRef define_llvm_function(int, LLVMModuleRef);
	int is_function_supported(int);

	$(awk '{ print "LLVMValueRef define_" $0 "(LLVMModuleRef);" }' \
		<<<"$EXTRA_FUNC_LIST")

	$(awk '{ print "LLVMTypeRef " $0 "(LLVMModuleRef);" }' \
		<<<"$TYPE_LIST")

	#endif
EOF

#
# Generate BACKEND_H.
#
cat > $BACKEND_H <<-EOF
	#ifndef LLVM_BACKEND_H
	#define LLVM_BACKEND_H

	#include <llvm/Pass.h>
	#include <llvm/ADT/SmallVector.h>
	#include <llvm/IR/Verifier.h>
	#include <llvm/IR/BasicBlock.h>
	#include <llvm/IR/CallingConv.h>
	#include <llvm/IR/Constants.h>
	#include <llvm/IR/DerivedTypes.h>
	#include <llvm/IR/Function.h>
	#include <llvm/IR/GlobalVariable.h>
	#include <llvm/IR/IRPrintingPasses.h>
	#include <llvm/IR/InlineAsm.h>
	#include <llvm/IR/Instructions.h>
	#include <llvm/IR/LLVMContext.h>
	#include <llvm/IR/LegacyPassManager.h>
	#include <llvm/IR/Module.h>
	#include <llvm/Support/FormattedStream.h>
	#include <llvm/Support/MathExtras.h>
	#include <algorithm>

	using namespace llvm;

	$(awk '{ print "Function* " $0 "(Module *mod);" }' <<<"$FUNCTION_LIST")

	$(awk '{ print "Type* " $0 "(Module *mod);" }' <<<"$TYPE_LIST")

	#endif
EOF

#
# Generate BACKEND_WRAPPER_CPP.
#
cat > $BACKEND_WRAPPER_CPP <<-EOF
	#include "$(basename $BACKEND_H)"

	#define lengthof(array) (sizeof (array) / sizeof ((array)[0]))

	typedef Function* (*llvmFuncDefineRef)(Module*);

	struct {
	    int             		funcid;
	    llvmFuncDefineRef     	func;
	} llvmDefineFunc [] = {
	$(join -1 4 -2 1 -o "1.1 1.2 1.3 1.4 1.5" <(\
			awk '/^ *{/ { print "  {", $2, "& define_" $7, "}," }' \
				$PG_SRC/backend/utils/fmgrtab.c |
			sort -k4) -<<<"$FUNCTION_LIST" |
	   sort -k2 -n)
	};

	static int
	binary_search(int funcid, int start, int end)
	{
	    if (start > end)
	        return -1;
	    else
	    {
	        int mid = start + (end - start) / 2;

		    if (llvmDefineFunc[mid].funcid > funcid)
		        return binary_search(funcid, start, mid - 1);
		    else if (llvmDefineFunc[mid].funcid < funcid)
		        return binary_search(funcid, mid + 1, end);
		    else
		        return mid;
	    }
	}

	extern "C" LLVMValueRef
	define_llvm_function(int funcid, LLVMModuleRef mod)
	{
	    int i = binary_search(funcid, 0, lengthof(llvmDefineFunc));
	    return i != -1 ? wrap(llvmDefineFunc[i].func(unwrap(mod))) : NULL;
	}

	extern "C" int
	is_function_supported(int funcid)
	{
	    int ret = binary_search(funcid, 0, lengthof(llvmDefineFunc));
	    return ret != -1 ? 1 : 0;
	}
	$(awk '{ print "\n" \
		"extern \"C\" LLVMValueRef\n" \
		"define_" $0 "(LLVMModuleRef mod)\n" \
		"{\n" \
		"    return wrap(define_" $0 "(unwrap(mod)));\n" \
		"}" }' <<<"$EXTRA_FUNC_LIST")
	$(awk '{ print "\n" \
		"extern \"C\" LLVMTypeRef\n" \
		 $0 "(LLVMModuleRef mod)\n" \
		"{\n" \
		"    return wrap(" $0 "(unwrap(mod)));\n" \
		"}" }' <<<"$TYPE_LIST")
EOF
