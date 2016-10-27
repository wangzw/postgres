#!/bin/bash
# -*- mode: shell-script; indent-tabs-mode: t; tab-width: 4; -*-
set -euo pipefail
IFS=$'\n'

SPLIT_LIMIT=$1
BACKEND_FILE=$2
BACKEND_H=llvm_backend.h

rm -f ${BACKEND_FILE%%.cpp}.*.cpp

grep -nE '^(Function|Type)\* .*\(Module \*mod\) \{$' $BACKEND_FILE |
	awk -v SPLIT_LIMIT=$SPLIT_LIMIT \
		'{ if (NR % SPLIT_LIMIT == 1) print $0 }' |
	cut -d: -f1 |
	sed p |
	sed 1d |
	xargs -n2 printf "%s\n%s-1\n" |
	bc |
	sed 's:-1:$:' |
	xargs -n2 printf "%s,%sp\n" |
	nl -w2 -nrz -s$'\t' |
	while IFS=$'\t' read PART SLICE; do
		PART_FILE=${BACKEND_FILE%%.cpp}.$PART.cpp
		printf "#include \"$(basename $BACKEND_H)\"\n\n" >$PART_FILE
		sed -n $SLICE $BACKEND_FILE >>$PART_FILE
		echo $PART_FILE
	done
