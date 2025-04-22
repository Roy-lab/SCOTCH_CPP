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

double utils::get_sum_vector_one_norm(gsl_matrix* X) {
	int n = X-> size1;
	int m = X-> size2;
	double sum = 0;
	for (int i = 0; i < n; i ++) {
		gsl_vector_view row  = gsl_matrix_row(X, i);
		sum += gsl_blas_dasum(&row.vector); 
	}
	return sum;
}
	


int utils::get_inverse(gsl_matrix* A) {
	gsl_linalg_cholesky_decomp(A);
	gsl_linalg_cholesky_invert(A);
	return 0;
}

int utils::concat_matrix_rows(gsl_matrix* M1, gsl_matrix* M2, gsl_matrix* M_new){
        gsl_matrix_view m1_view = gsl_matrix_submatrix( M_new, 0, 0, M1->size1, M1->size2);
        gsl_matrix_view m2_view = gsl_matrix_submatrix( M_new, M1->size1, 0, M2->size1, M2->size2);
        gsl_matrix_memcpy( &m1_view.matrix, M1);
        gsl_matrix_memcpy( &m2_view.matrix, M2);
        return 0;
}

int utils::concat_matrix_columns( gsl_matrix* M1, gsl_matrix* M2, gsl_matrix* M_new){
        gsl_matrix_view m1_view = gsl_matrix_submatrix( M_new, 0, 0, M1->size1, M1->size2);
        gsl_matrix_view m2_view = gsl_matrix_submatrix( M_new, 0, M1->size2, M2->size1, M2->size2);
        gsl_matrix_memcpy( &m1_view.matrix, M1);
        gsl_matrix_memcpy( &m2_view.matrix, M2);
        return 0;
}

int utils::concat_matrix_diagonal(gsl_matrix* M1, gsl_matrix* M2, gsl_matrix* M_new){
        gsl_matrix_view m1_view = gsl_matrix_submatrix( M_new, 0, 0, M1->size1, M1->size2);
        gsl_matrix_view m2_view = gsl_matrix_submatrix( M_new, M1->size1, M1->size2, M2->size1, M2->size2);
        gsl_matrix_memcpy( &m1_view.matrix, M1);
        gsl_matrix_memcpy( &m2_view.matrix, M2);
        return 0;
}

int utils::matrix_abs(gsl_matrix* M, gsl_matrix* abs_M){
	gsl_matrix_memcpy(abs_M, M);
	for(int i=0; i < abs_M->size1; i++){
		for(int j=0; j < abs_M->size2; j++){
			abs_M -> data[i * abs_M->tda +j] = fabs( abs_M->data[i * abs_M->tda +j]);
		}
	}
	return 0;
}

int utils::pos_matrix_elements(gsl_matrix* M, gsl_matrix* pos_M){
	gsl_matrix_memcpy(pos_M, M);
	for(int i=0; i < pos_M->size1; i++){
                for(int j=0; j < pos_M->size2; j++){
			if( pos_M -> data[i * pos_M->tda +j] >=0 ){
                        	pos_M -> data[i * pos_M->tda +j] = pos_M->data[i * pos_M->tda +j];
                	}
		}
        }
	return 0;
}

int utils::neg_matrix_elements(gsl_matrix* M, gsl_matrix* neg_M){
        gsl_matrix_memcpy(neg_M, M);
        for(int i=0; i < neg_M->size1; i++){
                for(int j=0; j < neg_M->size2; j++){
                        if( neg_M -> data[i * neg_M->tda +j] <=0 ){
                                neg_M -> data[i * neg_M->tda +j] = -neg_M->data[i * neg_M->tda +j];
                        }
                }
        }
	return 0;
}

