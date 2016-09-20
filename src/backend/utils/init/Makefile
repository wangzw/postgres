#-------------------------------------------------------------------------
#
# Makefile--
#    Makefile for utils/init
#
# IDENTIFICATION
#    src/backend/utils/init/Makefile
#
#-------------------------------------------------------------------------

subdir = src/backend/utils/init
top_builddir = ../../../..
include $(top_builddir)/src/Makefile.global

OBJS = globals.o miscinit.o postinit.o

postinit.o: CFLAGS += $(shell llvm-config --cflags)

include $(top_srcdir)/src/backend/common.mk
