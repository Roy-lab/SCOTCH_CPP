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
#include "modules/nmf.h" 
#include "modules/utils.h"
#include "modules/node.h"
#include "modules/leaf.h"
#include "modules/initialization.h"
#include "modules/root.h"
#include "modules/branch.h"
#include "modules/tmi.h"

int main(int argc, char **argv)
{
	struct timeval beginTime;
	gettimeofday(&beginTime,NULL);

	struct rusage bUsage;
	getrusage(RUSAGE_SELF,&bUsage);

	const char* treeFile;
	int nFeatures = -1;
	int nComponents = -1;
	
	string outputPrefix = string("");
	int randomState = 1010;
	int verbose = true;
	int maxIter = 300;
	double tol = 1;
	double alpha = 10;
	double lambda = 0;
	int neighborhoodRadius = 2; 
	string graphFile = "";
	string usage = string("usage.txt");

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

	if ((argc - optind) < 3) {
		io::print_usage(usage);
		return 1;
	} else {
		treeFile = argv[optind];
		nFeatures = atoi(argv[optind+1]);
		nComponents = atoi(argv[optind+2]);
	}

	string treeFileName(treeFile);
	vector<int> parentIds;
	vector<int> nSamples; //# of cells
	vector<string> aliases, fileNames;
	io::read_tree(treeFileName, parentIds, aliases, fileNames, nSamples);

	TMI tmi = TMI(nComponents,maxIter,randomState,verbose,tol,alpha,lambda);
	tmi.make_tree(parentIds, aliases, fileNames, nSamples, nFeatures);
	//tmi.make_tree_asymm(parentIds, aliases, fileNames, nSamples, nFeatures);
	tmi.fit();
	tmi.print_factors(outputPrefix);

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
	
	return 0;
}
