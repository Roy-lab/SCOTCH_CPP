#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_blas.h>
#include <vector>
#include <cmath>
#include <iostream>
#include "node.h"
#include "leaf.h"
#include "utils.h"
#include "io.h"

int Leaf::update_kth_block_of_U(int k) {
	gsl_vector_view u_k = gsl_matrix_row(U, k);
	gsl_vector_view v_k = gsl_matrix_row(V, k);
	double v_norm = pow(gsl_blas_dnrm2(&v_k.vector), 2);
	gsl_blas_dgemv(CblasNoTrans, 1, R, &v_k.vector, 0, &u_k.vector);
	int n = U->size2;
	for (int i = 0; i < n; i++) {
		double *val = &((&u_k.vector)->data[i]);
		if (*val < 0) {
			*val = 0;
		}
	}
	gsl_vector_scale(&u_k.vector, 1/v_norm);
	//if (gsl_vector_isnull(&u_k.vector)) {
	//	gsl_vector_add_constant(&u_k.vector, 1e-3);
	//}
	return 0;
}	

int Leaf::update_kth_block_of_V(int k) {
	gsl_vector_view u_k = gsl_matrix_row(U, k);
	gsl_vector_view v_k = gsl_matrix_row(V, k);
	gsl_vector_view p_k = gsl_matrix_row(parent->get_V(), k);
	gsl_vector_memcpy(&v_k.vector, &p_k.vector);
	double u_norm = pow(gsl_blas_dnrm2(&u_k.vector), 2);
	gsl_blas_dgemv(CblasTrans, 1, R, &u_k.vector, alpha, &v_k.vector);
	int m = V->size2;
	if (lambda > 0) {
		gsl_vector_add_constant(&v_k.vector, -0.5*lambda);
	}
	for (int i = 0; i < m; i++) {
		double *val = &((&v_k.vector)->data[i]);
		if (*val < 0) {
			*val = 0;
		}
	}
	gsl_vector_scale(&v_k.vector, 1/(u_norm + alpha));
	//if (gsl_vector_isnull(&v_k.vector)) {
	//	gsl_vector_add_constant(&v_k.vector, 1e-3);
	//}
	return 0;
}

int Leaf::normalize_and_scale() {
	for (int k = 0; k < n_components; k++) {
		gsl_vector_view u_k = gsl_matrix_row(U, k);
		gsl_vector_view v_k = gsl_matrix_row(V, k);
		double norm = gsl_blas_dnrm2(&v_k.vector);
		gsl_vector_scale(&v_k.vector, 1/norm);
		gsl_vector_scale(&u_k.vector, norm);
	}
	return 0;
}

int Leaf::update() {
	for (int k = 0; k < n_components; k++) {
		gsl_vector_view u_k = gsl_matrix_row(U, k);
		gsl_vector_view v_k = gsl_matrix_row(V, k);
		gsl_blas_dger(1, &u_k.vector, &v_k.vector, R);
		update_kth_block_of_U(k);
		update_kth_block_of_V(k);
		gsl_blas_dger(-1, &u_k.vector, &v_k.vector, R);
	}
	normalize_and_scale();
	return 0;
}

double Leaf::calculate_objective() {
	//X-UV
	gsl_matrix_memcpy(R, X);
  gsl_blas_dgemm(CblasTrans, CblasNoTrans, -1, U, V, 1, R);
	double error = utils::get_frobenius_norm(R);
	//sparsity regualrization
	if (lambda > 0) {
		for (int k = 0; k < n_components; k++) {
			gsl_vector_view v_k = gsl_matrix_row(V, k);
			error += lambda * gsl_blas_dasum(&v_k.vector);
		}
	}
	//minimizing different to parent's V
	gsl_matrix* temp = gsl_matrix_alloc(V->size1, V->size2);
	gsl_matrix_memcpy(temp, V);
	gsl_matrix_sub(temp, parent->get_V());
	double treeRegError = alpha * utils::get_frobenius_norm(temp);
	gsl_matrix_free(temp);
	error += treeRegError;
	return error;
}

