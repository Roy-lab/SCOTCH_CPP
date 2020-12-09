#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <list>
#include <vector>
#include <math.h>
#include "utils.h"

double utils::get_frobenius_norm(gsl_matrix* X) {
	int n = X->size1;
	int m = X->size2;
	double sum = 0;
	gsl_vector* row_copy = gsl_vector_alloc(m);
	for (int i = 0; i < n; i++) {
		gsl_vector_view row = gsl_matrix_row(X, i);
		gsl_vector_memcpy(row_copy, &row.vector); 
		gsl_vector_mul(row_copy, &row.vector);
		double rowsum = gsl_blas_dasum(row_copy);
		sum += rowsum;
	}
	gsl_vector_free(row_copy);
	return sum;
}

int utils::get_inverse(gsl_matrix* A) {
	gsl_linalg_cholesky_decomp1(A);
	gsl_linalg_cholesky_invert(A);
	return 0;
}
