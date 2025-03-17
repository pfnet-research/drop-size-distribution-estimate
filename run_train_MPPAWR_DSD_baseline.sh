#!/bin/bash

#pip install torchdiffeq

WD=$(cd $(dirname $0); pwd)
cd $WD

cond=1
OUTDIR="work"
rm -rf ${OUTDIR}
mkdir -p ${OUTDIR}

export PYTHONPATH=$PWD
python train_MPPAWR_DSD.py --config config/baseline.yaml > out_baseline.txt 2>&1
