#!/bin/bash
set -e

export OPENBLAS_VERBOSE=1
OPENBLAS_NUM_THREADS=4


#export LD_LIBRARY_PATH ${LD_LIBRARY_PATH}:/mnt/dv/wid/projects2/Roy-common/programs/thirdparty/gsl-2.6/lib

## Algo type zero Should be used with legacy!!!!!
mkdir -p output/alg_0
./run_nmtf --output output/alg_0/ --algotype 0 --legacy  --data input/toy/A.txt --n_samples 95 --n_features 120 --k1 3 --k2 3

#### Additional implementation, less effective though.
mkdir -p output/alg_1
./run_nmtf --output output/alg_0/ --algotype 1 --legacy  --data input/toy/A.txt --n_samples 95 --n_features 120 --k1 3 --k2 3

### Additional implementations
mkdir -p output/alg_2
./run_nmtf --output output/alg_0/ --algotype 2 --legacy --data input/toy/A.txt --n_samples 95 --n_features 120 --k1 3 --k2 3

## Using legacy test alpha
mkdir -p output/alg_0_lambda_alpha_1000
./run_nmtf --output output/alg_0_lambda_alpha_1000/ --legacy --data input/toy/A.txt --n_samples 95 --n_features 120 --k1 3 --k2 3 \
--lambda_u 1000 --lambda_v 1000 --alpha_u 1000 --alpha_v 1000

### using unit
mkdir -p output/unit
./run_nmtf --output output/unit --data input/toy/A.txt --n_samples 95 --n_features 120 --k1 3 --k2 3

### Test unit test regularization
mkdir -p output/unit_lambda_alpha_0_1
./run_nmtf --output output/unit_lambda_alpha_0_1 --data input/toy/A.txt --n_samples 95 --n_features 120 --k1 3 --k2 3 \
--lambda_u 0.01 --lambda_v 0.01 --alpha_u 0.01 --alpha_v 0.01

## Test multi K

./run_nmtf --output output/unit_lambda_alpha_0_1 --data input/toy/A.txt --n_samples 95 --n_features 120 --mult_k test_mult_k.txt \
--lambda_u 0.1 --lambda_v 0.1 --alpha_u  0.1 --alpha_v 0.1

