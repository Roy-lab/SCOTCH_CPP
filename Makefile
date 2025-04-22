#CHANGE PATHS AS NEEDED:
#GSL=/mnt/dv/wid/projects2/Roy-common/programs/thirdparty/gsl-2.6
GSL=/opt/homebrew/opt/gsl
INCLUDE_PATH=${GSL}/include
LIBRARY_PATH=${GSL}/lib
#compiler: gcc for C programs, g++ for C++ programs
XX = g++
CC = gcc

#compiler flags
CFLAGS = -g -std=c++11
GSLFLAGS = -lgsl -lgslcblas

#subset of files
NMF = SCOTCH_CPP/modules/initialization.cpp SCOTCH_CPP/modules/nmtf.cpp SCOTCH_CPP/modules/utils.cpp SCOTCH_CPP/modules/io.cpp

all: clean run_nmtf

matf:
	$(CC) -c -o SCOTCH_CPP/modules/random_svd/matrix_funcs.o SCOTCH_CPP/modules/random_svd/matrix_vector_functions_gsl.c -I ${INCLUDE_PATH}

rsvd:
	$(CC) -c -o SCOTCH_CPP/modules/random_svd/rsvd.o SCOTCH_CPP/modules/random_svd/low_rank_svd_algorithms_gsl.c -I ${INCLUDE_PATH}

run_nmtf:
	$(XX) run_nmtf.cpp $(NMF) SCOTCH_CPP/modules/random_svd/*.o -o run_nmtf $(CFLAGS) -L ${LIBRARY_PATH} ${GSLFLAGS} -I ${INCLUDE_PATH}

clean:
	rm run_nmtf
