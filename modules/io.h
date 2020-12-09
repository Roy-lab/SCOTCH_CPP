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
};
#endif
