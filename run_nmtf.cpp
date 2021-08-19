#include <iostream>
#include <fstream>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_vector.h>
#include <list>
#include <vector>
#include <math.h>
#include <string>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <sys/time.h>
#include <sys/resource.h>
#include "modules/io.h"
#include "modules/utils.h"
#include "modules/initialization.h"
#include "modules/nmtf.h"

int main(int argc, char **argv)
{
	struct timeval beginTime;
	gettimeofday(&beginTime,NULL);

	struct rusage bUsage;
	getrusage(RUSAGE_SELF,&bUsage);

	const char* matrixFile;
	int nSamples = -1;
	int nFeatures = -1;
	int uComponents = -1;
	int vComponents = -1;
	
	string outputPrefix = string("");
	int randomState = 1010;
	int verbose = true;
	int maxIter = 300;
	double tol = 1;
	double alpha = 10; //not currently used
	double lambda = 0; //not currently used
	string usage = string("usage_nmtf.txt");

	int c;
	while((c = getopt(argc, argv, "o:r:s:m:t:a:l:g")) != -1)
		switch (c) {
			case 'o':
				outputPrefix = string(optarg);
				break;
			case 'r':
				randomState = atoi(optarg);
				break;
			case 's':
				verbose = false;
				break;
			case 'm':
				maxIter = atoi(optarg);
				break;
			case 't':
				tol = atof(optarg);
				break;
			case 'a':
				alpha = atof(optarg);
				break;
			case 'l':
				lambda = atof(optarg);
				break;
			case 'h':
				io::print_usage(usage);
				return 0;
			case '?':
				io::print_usage(usage);
				return 0;
			default:
				io::print_usage(usage);
				return 0;
		}	

	if ((argc - optind) < 5) {
		io::print_usage(usage);
		return 1;
	} else {
		matrixFile = argv[optind];
		nSamples = atoi(argv[optind+1]);
		nFeatures = atoi(argv[optind+2]);
		uComponents = atoi(argv[optind+3]);
		vComponents = atoi(argv[optind+4]);
	}
	
	string matrixFileName(matrixFile);
	gsl_matrix* X = gsl_matrix_calloc(nSamples, nFeatures);
	gsl_matrix* U = gsl_matrix_calloc(uComponents, nSamples);
	gsl_matrix* V = gsl_matrix_calloc(vComponents, nFeatures);
	gsl_matrix* S = gsl_matrix_calloc(uComponents, vComponents);
	io::read_dense_matrix(matrixFileName, X);

	NMTF nmtf = NMTF(uComponents, vComponents, random_init,  maxIter,randomState,verbose,tol);
	nmtf.fit(X, U, V, S);
	io::write_dense_matrix(outputPrefix+"U.txt", U);
	io::write_dense_matrix(outputPrefix+"V.txt", V);
	io::write_dense_matrix(outputPrefix+"S.txt", S);
	
	
	struct timeval endTime;
	gettimeofday(&endTime,NULL);

	struct rusage eUsage;
	getrusage(RUSAGE_SELF,&eUsage);

	unsigned long int bt = beginTime.tv_sec;
	unsigned long int et = endTime.tv_sec;

	cout << "Total time elapsed: " << et - bt << " seconds" << endl;

	unsigned long int bu = bUsage.ru_maxrss;
	unsigned long int eu = eUsage.ru_maxrss;
	
	cout << "Memory usage: " << (eu - bu)/1000 << "MB" << endl;
	
	gsl_matrix_free(X);
	gsl_matrix_free(U);
	gsl_matrix_free(V);
	gsl_matrix_free(S);
	return 0;
}
