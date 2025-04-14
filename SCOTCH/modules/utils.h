#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <vector>
#ifndef _utils_
#define _utils_
using namespace std;

namespace utils
{
	double get_frobenius_norm(gsl_matrix*);
	double get_sum_vector_one_norm(gsl_matrix*);
	int get_inverse(gsl_matrix*);
	int concat_matrix_rows(gsl_matrix*, gsl_matrix*, gsl_matrix*);
	int concat_matrix_columns(gsl_matrix*, gsl_matrix*, gsl_matrix*);
	int concat_matrix_diagonal(gsl_matrix*, gsl_matrix*, gsl_matrix*);
	int matrix_abs(gsl_matrix*, gsl_matrix*);
	int neg_matrix_elements(gsl_matrix*, gsl_matrix*);
	int pos_matrix_elements(gsl_matrix*, gsl_matrix*);	
};
#endif
