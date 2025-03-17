#!/bin/bash

WD=$(cd $(dirname $0); pwd)
cd $WD

cond=1
OUTDIR="work"
rm -rf ${OUTDIR}
mkdir -p ${OUTDIR}

export PYTHONPATH=$PWD
python test_MPPAWR_DSD.py --config config/test_baseline.yaml > out_test_baseline.txt 2>&1
