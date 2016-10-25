#-------------------------------------------------------------------------
#
# Makefile--
#    Makefile for LLVM JIT backend
#
# IDENTIFICATION
#    src/backend/executor/llvm_backend/Makefile
#
#-------------------------------------------------------------------------

subdir = src/backend/executor/llvm_backend
top_builddir = ../../../..
include $(top_builddir)/src/Makefile.global

CXXFLAGS := $(shell $(llvm_config) --cxxflags) -g0
LLVM_BACKEND_FILE_LIMIT := 50
OBJS := llvm_backend_wrapper.o llvm_backend.a

llvm_backend.o: llvm_backend.cpp llvm_backend_wrapper.o

llvm_backend_wrapper.cpp: llvm_backend.cpp
	./backend-split.sh $(LLVM_BACKEND_FILE_LIMIT) $< |\
		xargs ./make-backend-wrapper.sh $(top_srcdir)/src

llvm_backend.a: llvm_backend_wrapper.cpp
	OBJ_PARTS=$$(echo llvm_backend.*.cpp | xargs -n1 sh -c 'echo $${0%.cpp}.o') &&\
	$(MAKE) -f $(top_srcdir)/src/backend/common.mk CXXFLAGS="$(CXXFLAGS)" $$OBJ_PARTS &&\
	$(AR) r $@ $$OBJ_PARTS

ifeq ($(enable_jit_rebuild), yes)
  SUBDIRS := passes
  BINDIR := $(shell $(llvm_config) --bindir)
  OPT := $(BINDIR)/opt
  LLVM_LINK := $(BINDIR)/llvm-link
  LLC := $(BINDIR)/llc

  OPT_OPTIONS := -O3
  CLANG_OPTS := $(CFLAGS) -g0 \
	-Wno-invalid-command-line-argument \
	-Wno-unused-command-line-argument \
	-Wno-ignored-attributes \
	-Wno-missing-prototypes

  LLVM_BACKEND_LIST := access/common/heaptuple.bc \
        executor/execAmi.bc \
        executor/execTuples.bc \
        utils/adt/arrayfuncs.bc \
        utils/adt/arrayutils.bc \
        utils/adt/bool.bc \
        utils/adt/cash.bc \
        utils/adt/char.bc \
        utils/adt/date.bc \
        utils/adt/datetime.bc \
        utils/adt/enum.bc \
        utils/adt/expandeddatum.bc \
        utils/adt/float.bc \
        utils/adt/int.bc \
        utils/adt/int8.bc \
        utils/adt/like.bc \
        utils/adt/numeric.bc \
        utils/adt/oracle_compat.bc \
        utils/adt/timestamp.bc \
        utils/adt/varchar.bc \
        utils/adt/varlena.bc \
        utils/fmgr/fmgr.bc

  llvm_backend.cpp: $(LLVM_BACKEND_LIST) passes/prune_static_vars.so
	$(LLVM_LINK) $(filter %.bc,$^) -o backend.bc
	$(OPT) $(OPT_OPTIONS) -disable-inlining backend.bc -o backend_opt.bc
	$(OPT) -load passes/prune_static_vars.so -prune-static-vars backend_opt.bc -o backend_pruned.bc
	$(OPT) $(OPT_OPTIONS) backend_pruned.bc -o backend_pruned_opt.bc
	$(LLC) -march=cpp -cppgen=functions backend_pruned_opt.bc -o $@
	$(LLC) -march=cpp -cppgen=types backend_pruned_opt.bc -o - >>$@

  $(LLVM_BACKEND_LIST): %.bc: $(top_srcdir)/src/backend/%.c
	mkdir -p $(dir $@)
	$(clang) -c -emit-llvm -I $(top_srcdir)/src/include $(CLANG_OPTS) $< -o $@
else
  CP := ln -s

  llvm_backend.cpp:
	$(CP) prebuild_llvm_backend.cpp llvm_backend.cpp

endif

clean:
	$(RM) llvm_backend_wrapper.cpp llvm_backend_wrapper.h \
		llvm_backend.cpp llvm_backend.h
	$(RM) $(wildcard llvm_backend.*.*)
	find \( -name "*.bc" -o -type d -empty \) -delete

include $(top_srcdir)/src/backend/common.mk
