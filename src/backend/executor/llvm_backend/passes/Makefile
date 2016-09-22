#-------------------------------------------------------------------------
#
# Makefile--
#    Makefile for extra LLVM passes for LLVM JIT
#
# IDENTIFICATION
#    src/backend/executor/llvm_backend/passes/Makefile
#
#-------------------------------------------------------------------------

subdir = src/backend/executor/llvm_backend/passes
top_builddir = ../../../../..
include $(top_builddir)/src/Makefile.global

CXXFLAGS := $(shell llvm-config --cxxflags)
OBJS = prune_static_vars.o prune_static_vars.so

include $(top_srcdir)/src/backend/common.mk
