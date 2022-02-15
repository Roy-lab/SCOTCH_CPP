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

NMTF::NMTF(int k, init_method initMethod, int maxIter, int seed, bool verb, double termTol, list<double> *err, list<double> *slope) {
	u_components = k;
	v_components = k;
	init = initMethod;
	max_iter = maxIter;
	random_state = seed;
	verbose = verb;
	tol = termTol;
	reconstruction_err_ = err;
	reconstruction_slope_ = slope;
	alphaU = 0;
        lambdaU = 0;
        alphaV = 0;
        lambdaV = 0;
	test=0;
}

NMTF::NMTF(int k1, int k2, init_method initMethod, int maxIter, int seed, bool verb, double termTol, list<double> *err, list<double> *slope){
	u_components = k1;
	v_components = k2;
	init = initMethod;
	max_iter = maxIter;
	random_state = seed;
	verbose = verb;
	tol = termTol;
	reconstruction_err_ = err;
	reconstruction_slope_ = slope;
	alphaU = 0;
        lambdaU = 0;
        alphaV = 0;
        lambdaV = 0;
	test=0;
}

NMTF::NMTF(int k1, int k2, init_method initMethod, int maxIter, int seed, bool verb, double termTol, list<double> *err, list<double> *slope, double aU, double lU){
        u_components = k1;
        v_components = k2;
        init = initMethod;
        max_iter = maxIter;
        random_state = seed;
        verbose = verb;
        tol = termTol;
        reconstruction_err_ = err;
	reconstruction_slope_ = slope;
	alphaU = aU;
	lambdaU = lU;
	alphaV = 0;
	lambdaV = 0;
	test=0;
}

NMTF::NMTF(int k1, int k2, init_method initMethod, int maxIter, int seed, bool verb, double termTol, list<double> *err, list<double> *slope, double aU, double lU, double aV, double lV){
        u_components = k1;
        v_components = k2;
        init = initMethod;
        max_iter = maxIter;
        random_state = seed;
        verbose = verb;
        tol = termTol;
        reconstruction_err_ = err;
	reconstruction_slope_ = slope;
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
		gsl_vector_free(alpha);
	}

	for (int i = 0; i < n; i++) {
		double *val = &((&u_k.vector)->data[i]);
		if (*val < 0) {
			*val = 0;
		}
	}
	
	if (gsl_vector_isnull(&u_k.vector)) {
		gsl_vector_set_all(&u_k.vector, 1/double(n));
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
		gsl_vector_free(alpha);
	}


	for (int i = 0; i < m; i++) {
		double *val = &((&v_k.vector)->data[i]);
		if (*val < 0) {
			*val = 0;
		}
	}

	
	if (gsl_vector_isnull(&v_k.vector)) {
		gsl_vector_set_all(&v_k.vector, 1/double(m));
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
	for (int k1 = 0; k1 < u_components; k1++) {
		gsl_vector_view u_k1 = gsl_matrix_row(U, k1);
		gsl_vector_view q_k1 = gsl_matrix_row(Q, k1);
		gsl_blas_dger( 1, &u_k1.vector, &q_k1.vector, R);
		update_kth_block_of_U(k1);
		gsl_blas_dger(-1, &u_k1.vector, &q_k1.vector, R);
	}
	update_P();
	for (int k2 = 0; k2 < v_components; k2++){
                gsl_vector_view v_k2 = gsl_matrix_row(V, k2);
                gsl_vector_view p_k2 = gsl_matrix_row(P, k2);
		gsl_blas_dger( 1, &p_k2.vector, &v_k2.vector, R);
		update_kth_block_of_V(k2);
                gsl_blas_dger(-1, &p_k2.vector, &v_k2.vector, R);
	}
	update_Q();
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
                	gsl_vector_set_all(&s_k1.vector, 1/double(u_components));
        	}
	}
	for (int k2 = 0; k2 < v_components; k2++){
		gsl_vector_view s_k2 = gsl_matrix_column(S, k2);	
		if (gsl_vector_isnull(&s_k2.vector)) {
                        gsl_vector_set_all(&s_k2.vector, 1/double(v_components));
        	}
	}
	update_P();
	update_Q();
	return 0;
}

int NMTF::update_US() {
        for (int k1 = 0; k1 < u_components; k1++) {
                gsl_vector_view u_k1 = gsl_matrix_row(U, k1);
                gsl_vector_view q_k1 = gsl_matrix_row(Q, k1);
                gsl_blas_dger( 1, &u_k1.vector, &q_k1.vector, R);
                update_kth_block_of_U(k1);
                gsl_blas_dger(-1, &u_k1.vector, &q_k1.vector, R);
        }
        update_P();
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
                        gsl_vector_set_all(&s_k1.vector, 1/double(u_components));
                }
        }
        for (int k2 = 0; k2 < v_components; k2++){
                gsl_vector_view s_k2 = gsl_matrix_column(S, k2);
                if (gsl_vector_isnull(&s_k2.vector)) {
                        gsl_vector_set_all(&s_k2.vector, 1/double(v_components));
                }
        }
        update_P();
        update_Q();
        return 0;
}       

int NMTF::update_SV() {
	for (int k2 = 0; k2 < v_components; k2++){
                gsl_vector_view v_k2 = gsl_matrix_row(V, k2);
                gsl_vector_view p_k2 = gsl_matrix_row(P, k2);
                gsl_blas_dger( 1, &p_k2.vector, &v_k2.vector, R);
                update_kth_block_of_V(k2);
                gsl_blas_dger(-1, &p_k2.vector, &v_k2.vector, R);
        }
        update_Q();
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
                        gsl_vector_set_all(&s_k1.vector, 1/double(u_components));
                }
        }
        for (int k2 = 0; k2 < v_components; k2++){
                gsl_vector_view s_k2 = gsl_matrix_column(S, k2);
                if (gsl_vector_isnull(&s_k2.vector)) {
                        gsl_vector_set_all(&s_k2.vector, 1/double(v_components));
                }
        }
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

int NMTF::fit(gsl_matrix* inputmat, gsl_matrix* W, gsl_matrix* H, gsl_matrix* D, gsl_matrix* Ris) {
	X = inputmat;
	n = X->size1;
	m = X->size2;
	U = W;
	V = H;
	S = D;
	R = Ris;
	P = gsl_matrix_alloc(v_components, n);
	Q = gsl_matrix_alloc(u_components, m);
	reconstruction_err_->clear();
	reconstruction_slope_->clear();
	int num_converge = 0;
	
	if ((U->size1 != u_components) || (V->size1 != v_components)) {
		cout << "The first dimension of U and V (i.e. their number of rows) should equal the number of components specified when instantiating NMF." << endl;
		return 1;
	} 
	
	update_P();
	update_Q();
	normalize_and_scale_u();
	normalize_and_scale_v();
	double old_error = calculate_objective();
	reconstruction_err_->push_back(old_error);
	double old_slope;
	
	for (int n_iter =0; n_iter < max_iter; n_iter++){
		update();
		normalize_and_scale_u();
		normalize_and_scale_v();
		double error = calculate_objective();
		double slope = (old_error - error)/old_error;
		reconstruction_err_->push_back(error);
		reconstruction_slope_->push_back(slope);
		if (verbose) {
			cout << "Itr " << n_iter+1 << " error = " << error << ", slope = " << slope << endl;
		}
		if (0 < slope && slope < tol) {
			num_converge++;
			if (num_converge > 0){
				if (verbose) {
					cout << "Converged at iteration " << n_iter+1 << endl;	
				}
				break;
			}
		} else {
			old_error = error;
			old_slope = slope;
			num_converge = 0;
		}
	}
	
	gsl_matrix_free(P);
	gsl_matrix_free(Q);
	return 0;
}


int NMTF::fit_US(gsl_matrix* inputmat, gsl_matrix* W, gsl_matrix* H, gsl_matrix* D, gsl_matrix* Ris) {
        X = inputmat;
        n = X->size1;
        m = X->size2;
        U = W;
        V = H;
        S = D;
        R = Ris;
        P = gsl_matrix_alloc(v_components, n);
        Q = gsl_matrix_alloc(u_components, m);
        reconstruction_err_->clear();
        reconstruction_slope_->clear();
        int num_converge = 0;

        if ((U->size1 != u_components) || (V->size1 != v_components)) {
                cout << "The first dimension of U and V (i.e. their number of rows) should equal the number of components specified when instantiating NMF." << endl;
                return 1;
        }

        update_P();
        update_Q();
        normalize_and_scale_u();
        normalize_and_scale_v();
        double old_error = calculate_objective();
        reconstruction_err_->push_back(old_error);
        double old_slope;

        for (int n_iter =0; n_iter < max_iter; n_iter++){
                update_US();
                normalize_and_scale_u();
                normalize_and_scale_v();
                double error = calculate_objective();
                double slope = (old_error - error)/old_error;
                reconstruction_err_->push_back(error);
                reconstruction_slope_->push_back(slope);
                if (verbose) {
                        cout << "Itr " << n_iter+1 << " error = " << error << ", slope = " << slope << endl;
                }
                if (0 < slope && slope < tol) {
                        num_converge++;
                        if (num_converge > 0){
                                if (verbose) {
                                        cout << "Converged at iteration " << n_iter+1 << endl;
                                }
                                break;
                        }
                } else {
                        old_error = error;
                        old_slope = slope;
                        num_converge = 0;
                }
        }
        gsl_matrix_free(P);
        gsl_matrix_free(Q);
        return 0;
}

int NMTF::fit_SV(gsl_matrix* inputmat, gsl_matrix* W, gsl_matrix* H, gsl_matrix* D, gsl_matrix* Ris) {
        X = inputmat;
        n = X->size1;
        m = X->size2;
        U = W;
        V = H;
        S = D;
        R = Ris;
        P = gsl_matrix_alloc(v_components, n);
        Q = gsl_matrix_alloc(u_components, m);
        reconstruction_err_->clear();
        reconstruction_slope_->clear();
        int num_converge = 0;

        if ((U->size1 != u_components) || (V->size1 != v_components)) {
                cout << "The first dimension of U and V (i.e. their number of rows) should equal the number of components specified when instantiating NMF." << endl;
                return 1;
        }

        update_P();
        update_Q();
        double old_error = calculate_objective();
        reconstruction_err_->push_back(old_error);
        double old_slope;

        for (int n_iter =0; n_iter < max_iter; n_iter++){
                update_SV();
                normalize_and_scale_u();
                normalize_and_scale_v();
                double error = calculate_objective();
                double slope = (old_error - error)/old_error;
                reconstruction_err_->push_back(error);
                reconstruction_slope_->push_back(slope);
                if (verbose) {
                        cout << "Itr " << n_iter+1 << " error = " << error << ", slope = " << slope << endl;
                }
                if (0 < slope && slope < tol) {
                        num_converge++;
                        if (num_converge > 0){
                                if (verbose) {
                                        cout << "Converged at iteration " << n_iter+1 << endl;
                                }
                                break;
                        }
                } else {
                        old_error = error;
                        old_slope = slope;
                        num_converge = 0;
                }
        }
        gsl_matrix_free(P);
        gsl_matrix_free(Q);
        return 0;
}





int NMTF::increase_k1_fixed_k2(int k1, gsl_matrix* U, gsl_matrix* V, gsl_matrix* S, gsl_matrix* R, gsl_matrix* X, gsl_matrix* U_new, gsl_matrix* S_new, gsl_rng* ri){
        int nSamples = X->size1;
        int nComponents = X->size2;
	int k1_diff = k1 - u_components;
	u_components = k1_diff;
        gsl_matrix *U_diff = gsl_matrix_calloc(k1_diff, nSamples);
        gsl_matrix *S_diff = gsl_matrix_calloc(k1_diff, V->size1);
        gsl_matrix *R_diff = gsl_matrix_calloc(nSamples, nComponents);
	gsl_matrix* R_abs = gsl_matrix_calloc(nSamples, nComponents);
        utils::matrix_abs(R, R_abs);
	
        init::random(U_diff, ri);
        init::random(S_diff, ri);
	string out_dir("temp/");	
        fit_US(R_abs, U_diff, V, S_diff, R_diff);
	//io::write_nmtf_output(U_diff, V, S_diff, R_abs, reconstruction_err_, reconstruction_slope_, out_dir);
	
	/*for(int i=0; i < k1_diff; i++){
                gsl_vector_view U_k = gsl_matrix_row(U_diff, i);
                subtract_factors(U, &U_k.vector);
        }*/
        utils::concat_matrix_rows( U, U_diff, U_new);
        utils::concat_matrix_rows( S, S_diff, S_new);
	//io::write_nmtf_output(U_new, V, S_new, R_abs, reconstruction_err_, reconstruction_slope_, out_dir);

	u_components = k1;
        fit(X, U_new, V, S_new, R);
	//io::write_nmtf_output(U_new, V, S_new, R, reconstruction_err_, reconstruction_slope_, out_dir);
        gsl_matrix_free(U_diff);
        gsl_matrix_free(S_diff);
        gsl_matrix_free(R_diff);
        gsl_matrix_free(R_abs);
	return 0;
}

int NMTF::increase_k2_fixed_k1(int k2, gsl_matrix* U, gsl_matrix* V, gsl_matrix* S, gsl_matrix* R, gsl_matrix* X, gsl_matrix* V_new, gsl_matrix* S_new, gsl_rng* ri){
        int nSamples = X->size1;
        int nComponents = X->size2;
        int k2_diff=k2 - v_components;
	v_components=k2_diff;
	gsl_matrix *V_diff = gsl_matrix_calloc(k2_diff, nComponents);
        gsl_matrix *S_diff = gsl_matrix_calloc(U->size1, k2_diff);
        gsl_matrix *R_diff = gsl_matrix_calloc(nSamples, nComponents);
	gsl_matrix* R_abs = gsl_matrix_calloc(nSamples, nComponents);
        utils::matrix_abs(R, R_abs);
	
        init::random(V_diff, ri);
        init::random(S_diff, ri);
	string out_dir("temp/");	
        fit_SV(R_abs, U, V_diff, S_diff, R_diff);
	//io::write_nmtf_output(U, V_diff, S_diff, R_abs, reconstruction_err_, reconstruction_slope_, out_dir);
        
	/*for(int i=0; i < k2_diff; i++){
                gsl_vector_view V_k = gsl_matrix_row(V_diff, i);
                subtract_factors(V, &V_k.vector);
        }*/
	utils::concat_matrix_rows( V, V_diff, V_new);
        utils::concat_matrix_columns( S, S_diff, S_new);
	//io::write_nmtf_output(U, V_new, S_new, R, reconstruction_err_, reconstruction_slope_, out_dir);
	
	v_components = k2;
        fit(X, U, V_new, S_new, R);
        //io::write_nmtf_output(U, V_new, S_new, R, reconstruction_err_, reconstruction_slope_, out_dir);
	gsl_matrix_free(V_diff);
        gsl_matrix_free(S_diff);
        gsl_matrix_free(R_diff);
        gsl_matrix_free(R_abs);
	return 0;
}

int NMTF::increase_k1_k2(int k1, int k2, gsl_matrix* U, gsl_matrix* V, gsl_matrix* S, gsl_matrix* R, gsl_matrix* X, gsl_matrix* U_new, gsl_matrix* V_new, gsl_matrix* S_new, gsl_rng* ri){
        int nSamples = X->size1;
        int nComponents = X->size2;
        int k1_diff = k1 - u_components;
	int k2_diff = k2 - v_components;
	
	u_components = k1_diff;
	v_components = k2_diff;
	gsl_matrix* U_diff = gsl_matrix_calloc(k1_diff, nSamples);
        gsl_matrix* V_diff = gsl_matrix_calloc(k2_diff, nComponents);
        gsl_matrix* S_diff = gsl_matrix_calloc(k1_diff, k2_diff);
        gsl_matrix* R_diff = gsl_matrix_calloc(nSamples, nComponents);
	gsl_matrix* R_abs = gsl_matrix_calloc(nSamples, nComponents);
	utils::matrix_abs(R, R_abs);
        
	init::random(U_diff, ri);
        init::random(V_diff, ri);
        init::random(S_diff, ri);
	
	//string out_dir("temp/");
        fit(R_abs, U_diff, V_diff, S_diff, R_diff);
	//io::write_nmtf_output(U_diff, V_diff, S_diff, R_abs, reconstruction_err_, reconstruction_slope_, out_dir);
	
	/*for(int i=0; i < k1_diff; i++){
		gsl_vector_view U_k = gsl_matrix_row(U_diff, i);
		subtract_factors(U, &U_k.vector);
	}*/	
	utils::concat_matrix_rows(U, U_diff, U_new);
	
	/*for(int i=0; i < k2_diff; i++){
		gsl_vector_view V_k = gsl_matrix_row(V_diff, i);
		subtract_factors(V, &V_k.vector);
	}*/
        utils::concat_matrix_rows(V, V_diff, V_new);
        
	utils::concat_matrix_diagonal(S, S_diff, S_new);
	//io::write_nmtf_output(U_new, V_new, S_new, R, reconstruction_err_, reconstruction_slope_, out_dir);
	u_components = k1;
	v_components = k2;
        fit(X, U_new, V_new, S_new, R);
	
	//io::write_nmtf_output(U_new, V_new, S_new, R, reconstruction_err_, reconstruction_slope_, out_dir);
	gsl_matrix_free(U_diff);
        gsl_matrix_free(V_diff);
        gsl_matrix_free(S_diff);
        gsl_matrix_free(R_diff);
	gsl_matrix_free(R_abs);
        return 0;
}

int NMTF::subtract_factors(gsl_matrix* A, gsl_vector* b){
	int nFactors = A->size1;
	int nTerms = A->size2;
	for(int i=0; i<nFactors; i++){
		gsl_vector_view a_k = gsl_matrix_row(A, i);
		gsl_vector_sub(&a_k.vector, b);
		//inforce non_negativity constraint.
		for(int j=0; j<nTerms; j++){
			double *val=&((&a_k.vector)->data[j]);
			if(*val < 0 ) {
				*val = 0;
			}
		}		
	}	
	
}

int NMTF::reset_k1_k2(int new_k1, int new_k2){
	u_components = new_k1;
	v_components = new_k2;
	return 0;
}
