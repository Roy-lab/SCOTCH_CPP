export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/mnt/dv/wid/projects2/Roy-common/programs/thirdparty/gsl-2.6/lib
ALGO=$1

mkdir -p output/alg_${ALGO}

./run_nmtf input/toy/A.txt 95 120 3 3 -o output/alg_${ALGO} -u ${ALGO} -m 10000

