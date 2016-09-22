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

CLANG := clang-3.7
ifeq ($(CLANG),)
 CLANG := clang
 CLANGVERSION := $(shell $(CLANG) --version |\
	head -n1 | sed 's/^[^0-9]*\([0-9.]*\).*/\1/')
 CLANGVERSION := $(shell expr $(CLANGVERSION) : '\([0-9]*\.[0-9]*\)')
 ifneq "$(CLANGVERSION)" "3.7"
  $(error "Need Clang version 3.7 ($(CLANGVERSION))")
 endif
endif

OPT := opt
LLVM_LINK := llvm-link
LLC := llc
LLVM_FLAGS  := $(shell llvm-config --libs --ldflags --system-libs all)
CXXFLAGS := $(shell llvm-config --cxxflags)
OPT_OPTIONS := -O3
CLANG_OPTS := $(shell pg_config --cflags) -g0 \
	-Wno-invalid-command-line-argument \
	-Wno-unused-command-line-argument \
	-Wno-ignored-attributes \
	-Wno-missing-prototypes
LLVM_BACKEND_FILE_LIMIT := 9000

LLVM_BACKEND_SRC := llvm_backend.cpp
LLVM_BACKEND_WRAPPER := llvm_backend_wrapper.cpp
LLVM_BACKEND_LIB := llvm_backend.a
OBJS := llvm_backend_wrapper.o llvm_backend.a
SUBDIRS := passes
EXTRA_PASSES := passes/prune_static_vars.so
LLVM_BACKEND_LIST := access/common/heaptuple.bc \
        access/hash/hashfunc.bc \
        access/heap/tuptoaster.bc \
        executor/execAmi.bc \
        executor/execTuples.bc \
        storage/buffer/bufmgr.bc \
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
        utils/fmgr/fmgr.bc \
        utils/mb/mbutils.bc \
        utils/mb/wchar.bc \
        utils/mmgr/mcxt.bc

llvm_backend.o: llvm_backend.cpp llvm_backend_wrapper.o

$(LLVM_BACKEND_SRC): $(EXTRA_PASSES) $(LLVM_BACKEND_LIST)
	$(LLVM_LINK) $(filter %.bc,$^) -o backend.bc
	$(OPT) $(OPT_OPTIONS) -disable-inlining backend.bc -o backend_opt.bc
	$(OPT) -load passes/prune_static_vars.so -prune-static-vars backend_opt.bc -o backend_pruned.bc
	$(OPT) $(OPT_OPTIONS) backend_pruned.bc -o backend_pruned_opt.bc
	$(LLC) -march=cpp -cppgen=functions backend_pruned_opt.bc -o $@

$(LLVM_BACKEND_WRAPPER): $(LLVM_BACKEND_SRC)
	./backend-split.sh $(LLVM_BACKEND_FILE_LIMIT) $< |\
		xargs ./make-backend-wrapper.sh $(top_srcdir)/src

$(LLVM_BACKEND_LIB): $(LLVM_BACKEND_WRAPPER)
	OBJ_PARTS=$$(echo llvm_backend.*.cpp | xargs -n1 sh -c 'echo $${0%.cpp}.o') &&\
	$(MAKE) -f $(top_srcdir)/src/backend/common.mk CXXFLAGS="$(CXXFLAGS)" $$OBJ_PARTS &&\
	$(AR) r $@ $$OBJ_PARTS

$(LLVM_BACKEND_LIST): %.bc: $(top_srcdir)/src/backend/%.c
	mkdir -p $(dir $@)
	$(CLANG) -c -emit-llvm -I $(top_srcdir)/src/include $(CLANG_OPTS) $< -o $@

clean:
	$(RM) llvm_backend_wrapper.cpp \
		llvm_backend_wrapper.h \
		llvm_backend.cpp \
		llvm_backend.h
	$(RM) $(wildcard llvm_backend.*.*)
	find \( -name "*.bc" -o -type d -empty \) -delete

include $(top_srcdir)/src/backend/common.mk
