#!/bin/bash
# -*- mode: shell-script; indent-tabs-mode: t; tab-width: 4; -*-
set -euo pipefail
IFS=$'\n'

SPLIT_LIMIT=$1
BACKEND_FILE=$2

rm -f ${BACKEND_FILE%%.cpp}.*.cpp

grep --line-number --only-matching --perl-regexp \
	 '(?<=Function\* ).*(?=\(Module \*mod\) \{)' $BACKEND_FILE |
	awk -v SPLIT_LIMIT=$SPLIT_LIMIT \
		'{ if (NR % SPLIT_LIMIT == 1) print $0 }' |
	cut --delimiter=: --fields=1 |
	sed p |
	sed 1d |
	xargs -n2 printf "%s\n%s-1\n" |
	bc |
	sed 's:-1:$:' |
	xargs -n2 printf "%s,%sp\n" |
	nl --number-width=2 --number-format=rz --number-separator=$'\t' |
	while IFS=$'\t' read PART SLICE; do
		PART_FILE=${BACKEND_FILE%%.cpp}.$PART.cpp
		sed -n $SLICE $BACKEND_FILE >$PART_FILE
		echo $PART_FILE
	done
