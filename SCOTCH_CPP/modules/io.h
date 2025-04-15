#include <gsl/gsl_matrix.h>
#include <list>
#include <vector>
#include <string>
#ifndef _io_
#define _io_
using namespace std;
namespace io
{
	int read_sparse_matrix(string, gsl_matrix*);
	//SR added this
	int read_dense_matrix(string, gsl_matrix*);
	int write_dense_matrix(string, gsl_matrix*);
	int write_list(string, list<double>&);
	int read_tree(string, vector<int>&, vector<string>&, vector<string>&, vector<int>&);
	int print_usage(string);
	int read_k1_k2_list(string inputFile, vector<int>&, vector<int>&);
	int read_k1_k2_list(string inputFile, vector<pair<int, int>>&);
	int write_nmtf_output(gsl_matrix*, gsl_matrix*, gsl_matrix*, gsl_matrix*, list<double>*, list<double>*, string&);
	int write_mem_and_time(string, unsigned long int, unsigned long int);
	int read_prev_results(string, gsl_matrix*, gsl_matrix*, gsl_matrix*);
};
#endif
