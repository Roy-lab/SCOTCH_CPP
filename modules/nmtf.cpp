#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_blas.h>
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include "initialization.h"
#include "nmtf.h"
#include "utils.h"
#include "io.h"

NMTF::NMTF(int k, init_method initMethod, int maxIter, int seed, bool verb, double termTol, list<double> *err) {
	u_components = k;
	v_components = k;
	init = initMethod;
	max_iter = maxIter;
	random_state = seed;
	verbose = verb;
	tol = termTol;
	reconstruction_err_ = err;
	alphaU = 0;
        lambdaU = 0;
        alphaV = 0;
        lambdaV = 0;
	test=0;
}

NMTF::NMTF(int k1, int k2, init_method initMethod, int maxIter, int seed, bool verb, double termTol, list<double> *err){
	u_components = k1;
	v_components = k2;
	init = initMethod;
	max_iter = maxIter;
	random_state = seed;
	verbose = verb;
	tol = termTol;
	reconstruction_err_ = err;
	alphaU = 0;
        lambdaU = 0;
        alphaV = 0;
        lambdaV = 0;
	test=0;
}

NMTF::NMTF(int k1, int k2, init_method initMethod, int maxIter, int seed, bool verb, double termTol, list<double> *err, double aU, double lU){
        u_components = k1;
        v_components = k2;
        init = initMethod;
        max_iter = maxIter;
        random_state = seed;
        verbose = verb;
        tol = termTol;
        reconstruction_err_ = err;
	alphaU = aU;
	lambdaU = lU;
	alphaV = 0;
	lambdaV = 0;
	test=0;
}

NMTF::NMTF(int k1, int k2, init_method initMethod, int maxIter, int seed, bool verb, double termTol, list<double> *err, double aU, double lU, double aV, double lV){
        u_components = k1;
        v_components = k2;
        init = initMethod;
        max_iter = maxIter;
        random_state = seed;
        verbose = verb;
        tol = termTol;
        reconstruction_err_ = err;
	alphaU = aU;
        lambdaU = lU;
        alphaV = aV;
        lambdaV = lV;
	test=0;
}



NMTF::~NMTF() {
}

int NMTF::update_kth_block_of_U(int k){
	gsl_vector_view u_k = gsl_matrix_row(U, k);
	gsl_vector_view q_k = gsl_matrix_row(Q, k);
	double q_norm = pow(gsl_blas_dnrm2(&q_k.vector), 2);
	gsl_blas_dgemv(CblasNoTrans, 1/q_norm, R, &q_k.vector, 0, &u_k.vector);
	int n = R->size1;
	
	//Orthogonality Term
	if (lambdaU > 0) {
                gsl_vector* u_others = gsl_vector_calloc(n);
                for (int j = 0; j < u_components; j++) {
                        if (j != k) {
                                gsl_vector_view u_j = gsl_matrix_row(U, j);
                                gsl_vector_add(u_others, &u_j.vector);
                        }
                }
                gsl_vector_scale(u_others, lambdaU/q_norm);
                gsl_vector_sub(&u_k.vector, u_others);
                gsl_vector_free(u_others);
        }
	
	//Sparsity Term
	if (alphaU > 0 ){
		gsl_vector* alpha = gsl_vector_calloc(n);
		gsl_vector_set_all(alpha, alphaU/q_norm);
		gsl_vector_sub(&u_k.vector, alpha);
	}

	for (int i = 0; i < n; i++) {
		double *val = &((&u_k.vector)->data[i]);
		if (*val < 0) {
			*val = 0;
		}
	}
	
	if (gsl_vector_isnull(&u_k.vector)) {
		gsl_vector_set_all(&u_k.vector, 0.01);
	}
	return 0;
}

int NMTF::update_kth_block_of_V(int k){
	gsl_vector_view p_k = gsl_matrix_row(P, k);
	gsl_vector_view v_k = gsl_matrix_row(V, k);
	double p_norm = pow(gsl_blas_dnrm2(&p_k.vector), 2);
	gsl_blas_dgemv(CblasTrans, 1/p_norm, R, &p_k.vector, 0, &v_k.vector);
	int m = R->size2;
	
	//Orthogonality Term
	if (lambdaV > 0) {
		gsl_vector* v_others = gsl_vector_calloc(m);
		for (int j = 0; j < v_components; j++) {
			if (j != k) {
				gsl_vector_view v_j = gsl_matrix_row(V, j);
				gsl_vector_add(v_others, &v_j.vector);
			}
		}
		gsl_vector_scale(v_others, lambdaV/p_norm);
		gsl_vector_sub(&v_k.vector, v_others);
		gsl_vector_free(v_others);
	}
	
	//Sparsity Term
	if (alphaV > 0 ){
		gsl_vector* alpha = gsl_vector_calloc(m);
		gsl_vector_set_all(alpha, alphaV/p_norm);
		gsl_vector_sub(&v_k.vector, alpha);	
	}


	for (int i = 0; i < m; i++) {
		double *val = &((&v_k.vector)->data[i]);
		if (*val < 0) {
			*val = 0;
		}
	}

	
	if (gsl_vector_isnull(&v_k.vector)) {
		gsl_vector_set_all(&v_k.vector, 0.01);
	}
	return 0;
}

int NMTF::update_ith_jth_of_S(int k1, int k2){
	gsl_vector* temp = gsl_vector_alloc(n);
	double* s_k1_k2 = gsl_matrix_ptr(S, k1, k2);
	gsl_vector_view u_k1 = gsl_matrix_row(U, k1);
	gsl_vector_view v_k2 = gsl_matrix_row(V, k2);
	double u_norm=pow(gsl_blas_dnrm2(&u_k1.vector), 2);
	double v_norm=pow(gsl_blas_dnrm2(&v_k2.vector), 2);		
	gsl_blas_dgemv(CblasNoTrans, 1/(u_norm*v_norm), R, &v_k2.vector, 0, temp);
	gsl_blas_ddot(&u_k1.vector, temp, s_k1_k2);
	if (*s_k1_k2 < 0) {
		*s_k1_k2 = 0;
	}
	gsl_vector_free(temp);
	return 0; 		
}

int NMTF::update() {
	//write_test_files("_before_U_");
	for (int k1 = 0; k1 < u_components; k1++) {
		gsl_vector_view u_k1 = gsl_matrix_row(U, k1);
		gsl_vector_view q_k1 = gsl_matrix_row(Q, k1);
		gsl_blas_dger( 1, &u_k1.vector, &q_k1.vector, R);
		update_kth_block_of_U(k1);
		gsl_blas_dger(-1, &u_k1.vector, &q_k1.vector, R);
	}
	update_P();
	//write_test_files("_before_V_");
	for (int k2 = 0; k2 < v_components; k2++){
                gsl_vector_view v_k2 = gsl_matrix_row(V, k2);
                gsl_vector_view p_k2 = gsl_matrix_row(P, k2);
		gsl_blas_dger( 1, &p_k2.vector, &v_k2.vector, R);
		update_kth_block_of_V(k2);
                gsl_blas_dger(-1, &p_k2.vector, &v_k2.vector, R);
	}
	update_Q();
	//write_test_files("_before_S_");
	for (int k1 = 0; k1 < u_components; k1++){
		for(int k2 = 0; k2 < v_components; k2++){
			double* s_k1_k2 = gsl_matrix_ptr(S, k1, k2);
			gsl_vector_view u_k1 = gsl_matrix_row(U, k1);
			gsl_vector_view v_k2 = gsl_matrix_row(V, k2);
			gsl_blas_dger( *s_k1_k2, &u_k1.vector, &v_k2.vector, R);
			update_ith_jth_of_S(k1, k2); 	
			gsl_blas_dger(-*s_k1_k2, &u_k1.vector, &v_k2.vector, R);
		}
		gsl_vector_view s_k1 = gsl_matrix_row(S, k1);
	 	if (gsl_vector_isnull(&s_k1.vector)) {
                	gsl_vector_set_all(&s_k1.vector, 0.01);
        	}
	}
	for (int k2 = 0; k2 < v_components; k2++){
		gsl_vector_view s_k2 = gsl_matrix_column(S, k2);	
		if (gsl_vector_isnull(&s_k2.vector)) {
                        gsl_vector_set_all(&s_k2.vector, 0.01);
        	}
	}
	//write_test_files("_before_Norm_");
	update_P();
	update_Q();
	return 0;
}


int NMTF::update_P() {
	//gsl_matrix_memcpy(R, X);
      	//gsl_blas_dgemm(CblasTrans, CblasNoTrans, -1, P, V, 1, R);
        gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1, S, U, 0, P);
	return 0;
}


int NMTF::update_Q() {
	//gsl_matrix_memcpy(R, X);
        //gsl_blas_dgemm(CblasTrans, CblasNoTrans, -1, P, V, 1, R);
	gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1, S, V, 0, Q);
	return 0;
}


int NMTF::normalize_and_scale_u() {
        for (int k1 = 0; k1 < u_components; k1++) {
                gsl_vector_view s_k1 = gsl_matrix_row(S, k1);
                gsl_vector_view u_k1 = gsl_matrix_row(U, k1);
                double norm = gsl_blas_dnrm2(&u_k1.vector);
		gsl_vector_scale(&u_k1.vector, 1/norm);
                gsl_vector_scale(&s_k1.vector, norm);
        }
	update_P();
        update_Q();
	return 0;
}


int NMTF::normalize_and_scale_v() {
	for (int k2 = 0; k2 < v_components; k2++) {
		gsl_vector_view s_k2 = gsl_matrix_column(S, k2);
		gsl_vector_view v_k2 = gsl_matrix_row(V, k2);
		double norm = gsl_blas_dnrm2(&v_k2.vector);
		gsl_vector_scale(&v_k2.vector, 1/norm);
		gsl_vector_scale(&s_k2.vector, norm);
	}
	update_P();
	update_Q();
	return 0;
}

		
double NMTF::calculate_objective() {
	gsl_matrix_memcpy(R, X);
  	gsl_blas_dgemm(CblasTrans, CblasNoTrans, -1, P, V, 1, R);
	double error = utils::get_frobenius_norm(R);
	reconstruction_err_->push_back(error);
	return error;
}

int NMTF::write_test_files(string s){
	stringstream ss;
	ss << "test/" << test;
	io::write_dense_matrix(ss.str() + s + "X.txt", X);
	io::write_dense_matrix(ss.str() + s + "U.txt", U);
        io::write_dense_matrix(ss.str() + s + "V.txt", V);
        io::write_dense_matrix(ss.str() + s + "S.txt", S);
        io::write_dense_matrix(ss.str() + s + "P.txt", P);
        io::write_dense_matrix(ss.str() + s + "Q.txt", Q);
	return 0;
}

int NMTF::fit(gsl_matrix* inputmat, gsl_matrix* W, gsl_matrix* H, gsl_matrix* D) {
	X = inputmat;
	n = X->size1;
	m = X->size2;
	U = W;
	V = H;
	S = D;
	P = gsl_matrix_alloc(v_components, n);
	Q = gsl_matrix_alloc(u_components, m);
	R = gsl_matrix_alloc(n,m);
	
	if ((U->size1 != u_components) || (V->size1 != v_components)) {
		cout << "The first dimension of U and V (i.e. their number of rows) should equal the number of components specified when instantiating NMF." << endl;
		return 1;
	} 

	if (verbose) {
		cout << "Initializing..." << endl;
	}
	if (init == random_init) {
		init::random(U, V, S, random_state);
	} else { //still needs to be editted
		init::nndsvd(X, U, V, random_state);
	}
	
	update_P();
	update_Q();
	normalize_and_scale_u();
	normalize_and_scale_v();
	double old_error = calculate_objective();
	double old_slope;
	for (int n_iter =0; n_iter < max_iter; n_iter++){
		update();
		normalize_and_scale_u();
		normalize_and_scale_v();
		//write_test_files("_after_Norm_");
		//test++;
		double error = calculate_objective();
		double slope = old_error - error;
		if (verbose) {
			cout << "Itr " << n_iter+1 << " error = " << error << ", slope = " << slope << endl;
		}
		if (slope < tol) {
			if (verbose) {
				cout << "Converged at iteration " << n_iter+1 << endl;	
			}
			break;
		} else {
			old_error = error;
			old_slope = slope;
		}
	}
	
	gsl_matrix_free(R);
	gsl_matrix_free(P);
	gsl_matrix_free(Q);
	/*
	for (int i = 0; i < n_components; i++) {
		for (int j = 0; j < 10; j++) {
			cout << U->data[i * U->tda + j] << ", ";
		}
		cout << endl;
	}
	*/
	return 0;
}

