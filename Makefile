#CHANGE PATHS AS NEEDED:
GSL=/mnt/dv/wid/projects2/Roy-common/programs/thirdparty/gsl-2.6
INCLUDE_PATH=${GSL}/include
LIBRARY_PATH=${GSL}/lib
#compiler: gcc for C programs, g++ for C++ programs
XX = g++
CC = gcc

#compiler flags
CFLAGS = -g
GSLFLAGS = -lgsl -lgslcblas

#subset of files
NMF = modules/initialization.cpp modules/nmtf.cpp modules/utils.cpp modules/io.cpp

all: clean run_nmtf

matf:
	$(CC) -c -o modules/random_svd/matrix_funcs.o modules/random_svd/matrix_vector_functions_gsl.c -I${INCLUDE_PATH}

rsvd:
	$(CC) -c -o modules/random_svd/rsvd.o modules/random_svd/low_rank_svd_algorithms_gsl.c -I${INCLUDE_PATH}

run_nmtf:
	$(XX) run_nmtf.cpp modules/*.cpp modules/random_svd/*.o -o run_nmtf $(CFLAGS) -L${LIBRARY_PATH} ${GSLFLAGS} -I${INCLUDE_PATH}

clean:
	rm run_nmtf
