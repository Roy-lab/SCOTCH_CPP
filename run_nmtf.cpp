#include <iostream>
#include <fstream>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_rng.h>
#include <list>
#include <vector>
#include <string>
#include <sstream>
#include <stdlib.h>
#include <getopt.h>
#include <unistd.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <sys/resource.h>
#include "SCOTCH_CPP/modules/io.h"
#include "SCOTCH_CPP/modules/initialization.h"
#include "SCOTCH_CPP/modules/nmtf.h"



using namespace std;

typedef vector<pair<int, int>> k_vec_t;

//Global to reduce function call syntax. These are all required matrices
gsl_matrix *U, *V, *S, *P, *Q, *R, *X;
int nSamples=-1, nFeatures=-1;
NMTF nmtf;

//global times and mem usage devices;
timeval beginTime{};
timeval factorTime{};
timeval endTime{};
rusage bUsage{};
rusage eUsage{};
rusage fUsage{};


// Extracted initialization logic for RNG
gsl_rng* initialize_random_generator(int seed) {
	const gsl_rng_type* T;
	gsl_rng* ri;
	gsl_rng_env_setup();
	T = gsl_rng_default;
	ri = gsl_rng_alloc(T);
	gsl_rng_set(ri, seed);
	return ri;
}

int initialize_factors(int k1, int k2, int nSamples, int nFeatures) {
	U = gsl_matrix_calloc(k1, nSamples);
	V = gsl_matrix_calloc(k2, nFeatures);
	S = gsl_matrix_calloc(k1, k2);
	P = gsl_matrix_calloc(k2, nSamples);
	Q = gsl_matrix_calloc(k1, nFeatures);
	return 0;
}

//Frees all matrices except X and R
int free_factors()
{
	if (U != NULL) {
		gsl_matrix_free(U);
		U = NULL; // Optional: Set to NULL after freeing
	}

	if (V != NULL) {
		gsl_matrix_free(V);
		V = NULL; // Optional: Set to NULL after freeing
	}

	if (S != NULL) {
		gsl_matrix_free(S);
		S = NULL; // Optional: Set to NULL after freeing
	}

	if (P != NULL) {
		gsl_matrix_free(P);
		P = NULL; // Optional: Set to NULL after freeing
	}

	if (Q != NULL) {
		gsl_matrix_free(Q);
		Q = NULL; // Optional: Set to NULL after freeing
	}

	return 0;
}


int free_factors_fixed_k1()
{
	gsl_matrix_free(V);
	gsl_matrix_free(S);
	gsl_matrix_free(P);
	gsl_matrix_free(Q);
	return 0;
}

int free_factors_fixed_k2()
{
	gsl_matrix_free(U);
	gsl_matrix_free(S);
	gsl_matrix_free(P);
	gsl_matrix_free(Q);
	return 0;
}

int free_matrices()
{
	free_factors();
	gsl_matrix_free(R);
	gsl_matrix_free(X);
	return 0;
}


string build_directory_path(const string& outputPrefix, int k1, int k2)
{
	stringstream out_dir_str;
	out_dir_str << outputPrefix << '/' << "k1_" << k1 << "_k2_" << k2 << "/";
	return out_dir_str.str();
}

int load_completed_run(const string& outputPrefix, int index, k_vec_t k_list, int &prev_k1, int &prev_k2)
{

	prev_k1 = k_list[index].first;
	prev_k2 = k_list[index].second;

	string in_dir = build_directory_path(outputPrefix, prev_k1, prev_k2);

	free_factors();

	U = gsl_matrix_calloc(prev_k1, nSamples);
	V = gsl_matrix_calloc(prev_k2, nFeatures);
	S = gsl_matrix_calloc(prev_k1, prev_k2);
	P = gsl_matrix_calloc(prev_k2, nSamples);
	Q = gsl_matrix_calloc(prev_k1, nFeatures);
	nmtf.reset_k1_k2(prev_k1, prev_k2);
	io::read_prev_results(in_dir, U, V, S);
	return 0;
}


int processRun(const std::string& outputPrefix, int k1, int k2, const gsl_rng* rng) {

	timeval factorTime;
	timeval endTime;
	rusage eUsage;

	//Write outputs
	std::string out_dir = build_directory_path(outputPrefix, k1, k2);
	mkdir(outputPrefix.c_str(), 0766);
	mkdir(out_dir.c_str(), 0766);

	if(nmtf.test)
	{
		nmtf.outpath = out_dir;
	}

	//Set up run params
	nmtf.reset_k1_k2(k1, k2);
	initialize_factors(k1, k2, nSamples, nFeatures);

	//Initialize_matrices(Random only supported option)
	init::random(U, rng);
	init::random(V, rng);
	init::random(S, rng);

	//Fit
	gettimeofday(&factorTime, nullptr);
	nmtf.fit(X, U, V, S, P, Q, R);


		// Log memory and timing
	gettimeofday(&endTime, nullptr);
	getrusage(RUSAGE_SELF, &eUsage);


	unsigned long memoryUsage = (eUsage.ru_maxrss - bUsage.ru_maxrss) / 1000;
	io::write_mem_and_time(out_dir + "usage.txt", endTime.tv_sec - factorTime.tv_sec, memoryUsage);


	list<double>* err= nmtf.reconstruction_err_;
	list<double>* slope = nmtf.reconstruction_slope_;
	io::write_nmtf_output(U, V, S, R, err, slope, out_dir);

	//std::string tar_cmd = "tar czf " + outputPrefix + ".tgz " + outputPrefix;
	//system(tar_cmd.c_str());
	//sleep(10);
	return 0;
}


int
increaseRun(const std::string& outputPrefix, const gsl_rng *rng, int new_k1, int new_k2)
{
	timeval factorTime;
	timeval endTime;
	rusage eUsage;


	gettimeofday(&factorTime, nullptr);
	//Add handels so we don't accentently replace with uninitialized pointers
	gsl_matrix *U_new = U;
	gsl_matrix *V_new = V;
	gsl_matrix *S_new = S;
	gsl_matrix *P_new = P;
	gsl_matrix *Q_new = Q;

	if (new_k1 > nmtf.u_components){
		U_new = gsl_matrix_calloc(new_k1, X->size1);
	}

	if (new_k2 > nmtf.v_components)
	{
		V_new = gsl_matrix_calloc(new_k2, X->size2);
	}

	P_new = gsl_matrix_calloc(new_k2, X->size1);
	Q_new = gsl_matrix_calloc(new_k1, X->size2);
	S_new = gsl_matrix_calloc(new_k1, new_k2);

	if (new_k1 > nmtf.u_components && new_k2 > nmtf.v_components)
	{
		nmtf.increase_k1_k2(new_k1, new_k2, X, U, V, S, P, Q, R, U_new, V_new, S_new, P_new, Q_new, rng);
		free_factors();
	}else if (new_k1 > nmtf.u_components && new_k2 == nmtf.v_components)
	{
		nmtf.increase_k1_fixed_k2(new_k1, X, U, V, S, P, Q, R, U_new, S_new, P_new, Q_new, rng);
		free_factors_fixed_k2();
	}else if (new_k1 == nmtf.u_components && new_k2 > nmtf.v_components)
	{
		nmtf.increase_k2_fixed_k1(new_k2, X, U, V, S, P, Q, R, V_new, S_new, P_new, Q_new, rng);
		free_factors_fixed_k1();
	}


	U = U_new;
	V = V_new;
	S = S_new;
	P = P_new;
	Q = Q_new;

	//make out directory
	std::string out_dir = build_directory_path(outputPrefix, new_k1, new_k2);
	mkdir(outputPrefix.c_str(), 0766);
	mkdir(out_dir.c_str(), 0766);

	//log memory and time
	gettimeofday(&endTime, nullptr);
	getrusage(RUSAGE_SELF, &eUsage);
	unsigned long memoryUsage = (eUsage.ru_maxrss - bUsage.ru_maxrss) / 1000;
	io::write_mem_and_time(out_dir + "usage.txt", endTime.tv_sec - factorTime.tv_sec, memoryUsage);

	//log NMTF results
	list<double>* err= nmtf.reconstruction_err_;
	list<double>* slope = nmtf.reconstruction_slope_;
	io::write_nmtf_output(U, V, S, R, err, slope, out_dir);
	return 0;
}


int check_run_completion(k_vec_t k_list, const string& outputPrefix, int& completed)
{
	int k1(0), k2(0);
	ifstream Ufile;
	ifstream Vfile;
	ifstream Sfile;

	for (int i = 0; i < k_list.size(); i++)
	{
		k1 = k_list[i].first;
		k2 = k_list[i].second;
		string out_dir=build_directory_path(outputPrefix, k1, k2);
		Ufile.open((out_dir + "U.txt").c_str());
		Vfile.open((out_dir + "V.txt").c_str());
		Sfile.open((out_dir + "S.txt").c_str());
		if(Ufile && Vfile && Sfile){
			completed = i;
		}
		Ufile.close();
		Vfile.close();
		Sfile.close();
	}
	return completed;
}


int
decreaseRun(const string& outputPrefix, const gsl_rng *rng, int new_k1, int new_k2, k_vec_t k_list, int curr_i)
{
	//If the next element on the k1 and k2 is decreasing with respect to the previous,
	//an old matrix is loaded that has smaller k1 or K2 or we run from scratch.
	//free_factors();
	int prev_k1;
	int prev_k2;
	string in_path;

	int j;
	//Find the previous element in the k1 and k2 list that is smaller than the current element.
	for (j = curr_i-1; j >= 0; j--) // Go through index if J backwards
	{
		int test_k1 = k_list[j].first;
		int test_k2 = k_list[j].second;
		if (test_k1 <= new_k1 && test_k2 <= new_k2)
		{
			prev_k1 = test_k1;
			prev_k2 = test_k2;
			break;
		}
	}

	if (j == -1)
	{
		processRun(outputPrefix, new_k1, new_k2, rng);
	}else
	{
		//in_path = build_directory_path(outputPrefix, prev_k1, prev_k2);
		load_completed_run(outputPrefix, j, k_list, prev_k1, prev_k2);
		increaseRun(outputPrefix, rng, new_k1, new_k2);
	}
	return 0;
}

int run_multiple_NMTF_runs(const std::string& k_file, const std::string& outputPrefix, const gsl_rng *rng)
{
	//Read in k1 and k2
	k_vec_t k_list;
	io::read_k1_k2_list(k_file, k_list);

	int completed = -1;
	int prev_k1;
	int prev_k2;
	int new_k1;
	int new_k2;

	string out_dir;
	ifstream Ufile;
	ifstream Vfile;
	ifstream Sfile;

	//Check if any runs are completed
	completed = check_run_completion(k_list, outputPrefix, completed);
	if (completed >= 0)
	{
		load_completed_run(outputPrefix, completed, k_list, prev_k1, prev_k2);
		completed++;
	}else
	{
		prev_k1 = k_list[0].first;
		prev_k2 = k_list[0].second;
		processRun(outputPrefix, prev_k1, prev_k2, rng);
		completed = 1;
	}



	//Continue until all runs are completed
	while (completed < k_list.size())
	{
		new_k1 = k_list[completed].first;
		new_k2 = k_list[completed].second;
		if (new_k1 >= prev_k1 && new_k2 >= prev_k2)
		{
			increaseRun(outputPrefix, rng, new_k1, new_k2);
		}else
		{
			decreaseRun(outputPrefix, rng, new_k1, new_k2, k_list, completed);
		}
		completed++;
	}


	return 0;
}


int main(int argc, char **argv)
{
	gettimeofday(&beginTime,NULL);
	getrusage(RUSAGE_SELF,&bUsage);

	//*************** SET ALL 4AULTS *********************************
	//initialize factor and matrix size.
	string matrixFile;
	int k1 = -1, k2 = -1;

	//initialize file paths
	string outputPrefix = "./", k_file, inPrefix;

	//initialize_fit options
	bool multK=false, legacy = false, verbose = true; bool t = false;

	//initialize convergence and generative process
	int seed = 1010, maxIter = 100, algotype=0;
	double tol = 1e-5;

	//initialize regularization
	double alphaU = 0, lambdaU = 0, alphaV = 0, lambdaV = 0;

	// Initialize converge tracking elements
	list<double> *err= new list<double>;
	list<double> *slope = new list<double>;

	//Usage file
	string usage = string("nmtf_usage.txt");


	// ******************************** PARSE ARGS ******************************************
	// Parse optional parameters

	struct option long_options[] = {
		{"output",     required_argument, nullptr, 'o'},
		{"seed",       required_argument, nullptr, 'r'},
		{"silent",     no_argument,       nullptr, 's'},
		{"max_iter",   required_argument, nullptr, 'm'},
		{"tolerance",  required_argument, nullptr, 't'},
		{"alpha_v",    required_argument, nullptr, 'A'},
		{"lambda_v",   required_argument, nullptr, 'L'},
		{"alpha_u",    required_argument, nullptr, 'a'},
		{"lambda_u",   required_argument, nullptr, 'l'},
		{"mult_k",     required_argument, nullptr, 'i'},
		{"help",       no_argument,       nullptr, 'h'},
		{"algotype",   required_argument, nullptr, 'u'},
		{"legacy",     no_argument, nullptr, 'e'},
		{"input",      required_argument, nullptr, 'p'},

		// New options for the former positional arguments
		{"data",       required_argument, nullptr,  'X' },
		{"n_samples",  required_argument, nullptr,  'n'},
		{"n_features", required_argument, nullptr,  'f' },
		{"k1",         required_argument, nullptr,  'k' },
		{"k2",         required_argument, nullptr,  'K' },
		{"test",	no_argument,	 nullptr,	 'T'},
		{nullptr,      0,                 nullptr,  0 } // End of list
	};







	int c = 0;
	int option_index = 0;
	while ((c = getopt_long(argc, argv, ":o:r:sm:t:A:a:L:l:k:K:X:n:f:i:h:u:eTp:", long_options, &option_index)) != -1)
 {
		switch (c) {
			case 'o': outputPrefix = optarg; break;
			case 'r': seed = std::atoi(optarg); break;
			case 's': verbose = true; break;
			case 'm': maxIter = std::atoi(optarg); break;
			case 't': tol = std::atof(optarg); break;
			case 'A': alphaV = std::atof(optarg); break;
			case 'L': lambdaV = std::atof(optarg); break;
			case 'a': alphaU = std::atof(optarg); break;
			case 'l': lambdaU = std::atof(optarg); break;
			case 'k': k1 = std::atoi(optarg); break;
			case 'K': k2 = std::atoi(optarg); break;
			case 'X': matrixFile = optarg; break;
			case 'n': nSamples = std::atoi(optarg); break;
			case 'f': nFeatures = std::atoi(optarg); break;
			case 'i': k_file = optarg; multK = true; break;
			case 'h': io::print_usage(usage); return 0;
			case 'u': algotype = std::atoi(optarg); break;
			case 'e': legacy = true; break;
			case 'p': inPrefix = optarg; break;
			case 'T': t = true; break;
			default: io::print_usage(usage); return 0;
		}
	}

	// Example validation
	if (matrixFile.empty()) {
		std::cerr << "Error: Missing matrix file. Please provide --data.\n";
		io::print_usage(usage);
		return 1;  // Exit with an error
	}

	if (k1<=0 || k2<=0)
	{
		if (k_file.empty()) {
			std::cerr << "Error: Missing factor. Please provide --k1 and --k2 or --mult_k.\n";
			io::print_usage(usage);
			return 1;  // Exit with an error
		}
	}

	if (nSamples<=0	|| nFeatures<=0) {
		std::cerr << "Error: Missing matrix sizes. Please provide --n_samples or --n_features.\n";
		io::print_usage(usage);
		return 1;  // Exit with an error
	}

	//Initialize Random generator:
	// Extracted initialization logic for RNG
	gsl_rng* rng = initialize_random_generator(seed);

	//Initialize X and R
	X = gsl_matrix_calloc(nSamples, nFeatures);
	R = gsl_matrix_calloc(nSamples, nFeatures);
	string matrixFileName(matrixFile);
	io::read_dense_matrix(matrixFileName, X);




	nmtf.set_NMTF_params(random_init, maxIter, seed, verbose, tol, err, slope, alphaU, lambdaU, alphaV, lambdaV);
	nmtf.set_size(nSamples, nFeatures);
	nmtf.setLegacy(legacy);
	nmtf.set_test(t);
	if (legacy)
	{
		nmtf.setAlgotype(algotype);
	}

	if (multK)
	{
		run_multiple_NMTF_runs(k_file, outputPrefix, rng);
	}
	else
	{
		processRun(outputPrefix, k1, k2, rng);
	}

	/*
	if (nmtf.test)
	{
		io::write_dense_matrix(outputPrefix + "/X_out.txt", nmtf.X);
	}
	*/

	free_matrices();
	gsl_rng_free(rng);
	delete err;
	delete slope;
	return 0;
}










