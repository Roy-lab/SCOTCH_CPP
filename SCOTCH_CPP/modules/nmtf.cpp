#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>
#include <sys/time.h>
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include "initialization.h"
#include "nmtf.h"
#include "utils.h"
#include "io.h"

NMTF::NMTF()
{
	// Initialization where k1 and k2 are equal. This works correctly but isn't used in run_nmtf.
	u_components = 0;
	v_components = 0;
	init = random_init;
	max_iter = 100;
	random_state = 1101;
	verbose = true;
	tol = 1e-5;
	reconstruction_err_ = nullptr;
	reconstruction_slope_ = nullptr;
	alphaU = 0;
	lambdaU = 0;
	alphaV = 0;
	lambdaV = 0;
	test = 0;
	algotype = 0;
	legacy = false;
}

NMTF::NMTF(int k, init_method initMethod, int maxIter, int seed, bool verb, double termTol, list<double> *err,
           list<double> *slope)
{
	// Initialization where k1 and k2 are equal. This works correctly but isn't used in run_nmtf.
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
	test = 0;
	algotype = 0;
	legacy = false;
}

NMTF::NMTF(int k1, int k2, init_method initMethod, int maxIter, int seed, bool verb, double termTol, list<double> *err,
           list<double> *slope)
{
	//Initialization without regulatization
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
	test = 0;
	legacy = false;
}

NMTF::NMTF(int k1, int k2, init_method initMethod, int maxIter, int seed, bool verb, double termTol, list<double> *err,
           list<double> *slope, double aU, double lU)
{
	//Initialization with sparsity regularization
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
	test = 0;
	legacy = false;
}

NMTF::NMTF(int k1, int k2, init_method initMethod, int maxIter, int seed, bool verb, double termTol, list<double> *err,
           list<double> *slope, double aU, double lU, double aV, double lV)
{
	//Initialization with L1 regularization and orthognality regularization.
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
	test = 0;
	legacy = false;
}


NMTF::~NMTF()
{
}


int
NMTF::set_NMTF_params(int k1, int k2, init_method initMethod, int maxIter, int seed, bool verb, double termTol,
	list<double>* err, list<double>* slope, double aU, double lU, double aV, double lV)
{
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
	test = 0;
	legacy = false;
}

int
NMTF::set_NMTF_params_python(int k1, int k2, int maxIter, int seed, bool verb, double termTol,
	list<double>* err, list<double>* slope, double aU, double lU, double aV, double lV)
{
	set_NMTF_params(k1, k2, random_init, maxIter, seed, verb, termTol, err, slope, aU, lU, aV, lV);
}

int
NMTF::set_NMTF_params(init_method initMethod, int maxIter, int seed, bool verb, double termTol,
	list<double>* err, list<double>* slope, double aU, double lU, double aV, double lV)
{

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
	test = 0;
	legacy = false;
}

int
NMTF::update_kth_block_of_U(int k)
{
	//update function for kth row of U. Computed using the Q matrix and R.
	gsl_vector_view u_k = gsl_matrix_row(U, k);
	gsl_vector_view q_k = gsl_matrix_row(Q, k);
	double q_norm = pow(gsl_blas_dnrm2(&q_k.vector), 2);
	//R_i = X - sum_{ h ne k} u_h q_h^T Rank one matrices not including u_k
	//Update equation: R * q_i /  || q_i ||_2^2.
	gsl_blas_dgemv(CblasNoTrans, 1 / q_norm, R, &q_k.vector, 0, &u_k.vector);
	enforce_min_val(&u_k.vector);
	unit_normalize(&u_k.vector);
	return 0;
}

int
NMTF::update_kth_block_of_V(int k)
{
	//update function for the kth row of V. Compute using the P matrix and R
	gsl_vector_view p_k = gsl_matrix_row(P, k);
	gsl_vector_view v_k = gsl_matrix_row(V, k);
	double p_norm = pow(gsl_blas_dnrm2(&p_k.vector), 2);
	// R_j = X - sum_{ h ne k} p_h v_h^T Rank one matrices not included v_k
	// Update equation: R * p_i/ || p_i || ^2
	gsl_blas_dgemv(CblasTrans, 1 / p_norm, R, &p_k.vector, 0, &v_k.vector);
	enforce_min_val(&v_k.vector);
	unit_normalize(&v_k.vector);
	return 0;
}

int
NMTF::update_ith_jth_of_S(int k1, int k2)
{
	//update the ith and jth element at a time. Picked this method because no computation of inverse needed.
	// R_ij = X - sum_{c ne i} sum_{d ne j} u_c s_{c,d} v_d^T \\rank 1 matrix corresponding to the the c and d vector.
	//Update equation s_i,j = u_i^T R_ij v_j / (|| u_i||_2^2 ||v_j||_2^2)
	gsl_vector *temp = gsl_vector_alloc(n);
	double *s_k1_k2 = gsl_matrix_ptr(S, k1, k2);
	gsl_vector_view u_k1 = gsl_matrix_row(U, k1);
	gsl_vector_view v_k2 = gsl_matrix_row(V, k2);
	double u_norm = pow(gsl_blas_dnrm2(&u_k1.vector), 2);
	double v_norm = pow(gsl_blas_dnrm2(&v_k2.vector), 2);
	gsl_blas_dgemv(CblasNoTrans, 1 / (u_norm * v_norm), R, &v_k2.vector, 0, temp);
	gsl_blas_ddot(&u_k1.vector, temp, s_k1_k2);
	if (*s_k1_k2 < 0)
	{
		*s_k1_k2 = 0;
	}

	gsl_vector_free(temp);
	return 0;
}

int
NMTF::update_kth_block_of_S(int k)
{
	//Using Sushmita Roy's update equations for the ith block of S.
	//We will need to compute the inverse of (V^T V), which is a k2 x k2 matrix.
	//R_i = X - sum(C ne i) u_c s_.c V^T \\ this is the rank1 update in question.
	//R_i = R + u_i s_.i V^T
	//Update Equation s_i = u_k^T R_k V(V^T V)^(-1)/(u_k^T u_k)

	//Initialize temp matrices
	gsl_vector_view u_k = gsl_matrix_row(U, k);
	gsl_vector_view s_k = gsl_matrix_row(S, k);
	gsl_matrix *VTV = gsl_matrix_calloc(v_components, v_components);
	gsl_matrix *V_VTV_inv = gsl_matrix_calloc(v_components, m);
	gsl_matrix *R_V_VTV_inv = gsl_matrix_calloc(v_components, n);

	//Compute V^T V
	//io::write_dense_matrix("test/V.txt", V);
	gsl_blas_dsyrk(CblasLower, CblasNoTrans, 1, V, 0, VTV);
	//io::write_dense_matrix("test/VTV.txt", VTV);
	//Compute (V^T V)^(inv)
	gsl_linalg_cholesky_decomp(VTV);
	gsl_linalg_cholesky_invert(VTV);
	//io::write_dense_matrix("test/VTV_inv.txt", VTV);
	//Compute V (V^T V)^(inv)
	gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1, VTV, V, 0, V_VTV_inv);
	//Compute R_k V (V^T V) ^(inv)
	gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1, V_VTV_inv, R, 0, R_V_VTV_inv);
	//io::write_dense_matrix("test/R_V_VTV_inv.txt", R_V_VTV_inv);
	double u_norm = pow(gsl_blas_dnrm2(&u_k.vector), 2);
	gsl_blas_dgemv(CblasNoTrans, u_norm, R_V_VTV_inv, &u_k.vector, 0, &s_k.vector);
	//io::write_dense_matrix("test/S.txt", S);

	//Inforce Positivity
	for (int i = 0; i < v_components; i++)
	{
		double *val = &((&s_k.vector)->data[i]);
		if (*val < 0)
		{
			*val = 0;
		}
	}

	if (gsl_vector_isnull(&s_k.vector))
	{
		gsl_vector_set_all(&s_k.vector, 1 / double(v_components));
	}

	gsl_matrix_free(VTV);
	gsl_matrix_free(V_VTV_inv);
	gsl_matrix_free(R_V_VTV_inv);
	return 0;
}

int
NMTF::apply_orthog_u(int k, double q_norm)
{
	if (lambdaU > 0)
	{
		gsl_vector_view u_k = gsl_matrix_row(U, k);
		gsl_vector *u_others = gsl_vector_calloc(n);
		for (int j = 0; j < u_components; j++)
		{
			if (j != k)
			{
				gsl_vector_view u_j = gsl_matrix_row(U, j);
				gsl_vector_add(u_others, &u_j.vector);
			}
		}
		gsl_vector_scale(u_others, lambdaU / q_norm);
		//subtract regularization term.
		gsl_vector_sub(&u_k.vector, u_others);
		gsl_vector_free(u_others);

		enforce_min_val(&u_k.vector);
		unit_normalize(&u_k.vector);
	}
	return 0;
}

int
NMTF::apply_orthog_v(int k, double p_norm)
{
	if (lambdaV > 0)
	{
		gsl_vector_view v_k = gsl_matrix_row(V, k);
		gsl_vector *v_others = gsl_vector_calloc(m);
		for (int j = 0; j < v_components; j++)
		{
			if (j != k)
			{
				gsl_vector_view v_j = gsl_matrix_row(V, j);
				gsl_vector_add(v_others, &v_j.vector);
			}
		}
		gsl_vector_scale(v_others, lambdaV / p_norm);
		gsl_vector_sub(&v_k.vector, v_others);
		gsl_vector_free(v_others);

		enforce_min_val(&v_k.vector);
		unit_normalize(&v_k.vector);
	}
	return 0;
}

int
NMTF::apply_sparsity_u(int k, double q_norm)
{
	if (alphaU > 0)
	{
		gsl_vector_view u_k = gsl_matrix_row(U, k);
		enforce_min_val(&u_k.vector, alphaU / q_norm);
		unit_normalize(&u_k.vector);
	}
	return 0;
}


int
NMTF::apply_sparsity_v(int k, double p_norm)
{
	if (alphaV > 0)
	{
		gsl_vector_view v_k = gsl_matrix_row(V, k);
		enforce_min_val(&v_k.vector, alphaV / p_norm);
		unit_normalize(&v_k.vector);
	}
	return 0;
}

int
NMTF::update_U()
{
	double q_norm;
	for (int k1 = 0; k1 < u_components; k1++)
	{
		gsl_vector_view u_k1 = gsl_matrix_row(U, k1);
		gsl_vector_view q_k1 = gsl_matrix_row(Q, k1);
		q_norm = pow(gsl_blas_dnrm2(&q_k1.vector), 2); //note we want the squared value
		//Make R_i see equation for u_k update above for datail
		gsl_blas_dger(1, &u_k1.vector, &q_k1.vector, R);
		// update U_k
		update_kth_block_of_U(k1);
		apply_orthog_u(k1, q_norm);
		apply_sparsity_u(k1, q_norm);
		//Make R. Simply add the new rank 1 matrix.
		gsl_blas_dger(-1, &u_k1.vector, &q_k1.vector, R);
	}
	return 0;
}


int
NMTF::update_V()
{
	double p_norm;
	for (int k2 = 0; k2 < v_components; k2++)
	{
		gsl_vector_view v_k2 = gsl_matrix_row(V, k2);
		gsl_vector_view p_k2 = gsl_matrix_row(P, k2);
		p_norm = pow(gsl_blas_dnrm2(&p_k2.vector), 2); // note we want the norm squared
		//Make R_j see equation for v_k update above for detail.
		gsl_blas_dger(1, &p_k2.vector, &v_k2.vector, R);
		//Udpate V_k
		update_kth_block_of_V(k2);
		apply_orthog_v(k2, p_norm);
		apply_sparsity_v(k2, p_norm);
		//Make R
		gsl_blas_dger(-1, &p_k2.vector, &v_k2.vector, R);
	}
	return 0;
}

int
NMTF::update_S()
{
	for (int k1 = 0; k1 < u_components; k1++)
	{
		for (int k2 = 0; k2 < v_components; k2++)
		{
			double *s_k1_k2 = gsl_matrix_ptr(S, k1, k2);
			gsl_vector_view u_k1 = gsl_matrix_row(U, k1);
			gsl_vector_view v_k2 = gsl_matrix_row(V, k2);
			//Make R_i,j See equation for s_ij update above for details
			gsl_blas_dger(*s_k1_k2, &u_k1.vector, &v_k2.vector, R);
			//Update S_ij
			update_ith_jth_of_S(k1, k2);
			//Make R
			gsl_blas_dger(-*s_k1_k2, &u_k1.vector, &v_k2.vector, R);
		}
		//Check that all rows are nonzero.
		gsl_vector_view s_k1 = gsl_matrix_row(S, k1);
		if (gsl_vector_isnull(&s_k1.vector))
		{
			gsl_vector_set_all(&s_k1.vector, 1 / double(u_components));
		}
	}
	for (int k2 = 0; k2 < v_components; k2++)
	{
		//Check that all columns are nonzero
		gsl_vector_view s_k2 = gsl_matrix_column(S, k2);
		if (gsl_vector_isnull(&s_k2.vector))
		{
			gsl_vector_set_all(&s_k2.vector, 1 / double(v_components));
		}
	}
	return 0;
}

int
NMTF::update()
{
	update_U();
	update_P();

	update_V();
	update_Q();

	update_S();

	//Normalize and scale u v and S.
	normalize_and_scale_u();
	normalize_and_scale_v();
	//Compute P = US and Q=SV^T
	update_P();
	update_Q();
	return 0;
}

//SR is trying this other code version where we update the  ij block together like for the S matrix.
int
NMTF::update_ijblock()
{
	//Initialize tracking variables.
	//bool trained_u[u_components];
	//for(int k1 = 0; k1 < u_components; k1++){
	//	trained_u[k1] = 0;
	//}
	//bool trained_v[v_components];
	//for(int k2 = 0; k2 < v_components; k2++){
	//	trained_v[k2] = 0;
	//}


	// RUN UPDATE
	for (int k1 = 0; k1 < u_components; k1++)
	{
		for (int k2 = 0; k2 < v_components; k2++)
		{
			double *s_k1_k2 = gsl_matrix_ptr(S, k1, k2);
			gsl_vector_view u_k1 = gsl_matrix_row(U, k1);
			gsl_vector_view v_k2 = gsl_matrix_row(V, k2);
			gsl_vector_view q_k1 = gsl_matrix_row(Q, k1);

			// Update U
			gsl_blas_dger(1, &u_k1.vector, &q_k1.vector, R);
			update_kth_block_of_U(k1);
			update_P();
			gsl_blas_dger(-1, &u_k1.vector, &q_k1.vector, R);

			gsl_vector_view p_k2 = gsl_matrix_row(P, k2);
			//Update V
			gsl_blas_dger(1, &p_k2.vector, &v_k2.vector, R);
			update_kth_block_of_V(k2);
			update_Q();
			gsl_blas_dger(-1, &p_k2.vector, &v_k2.vector, R);

			//Update S
			gsl_blas_dger(*s_k1_k2, &u_k1.vector, &v_k2.vector, R);
			update_ith_jth_of_S(k1, k2);
			update_P();
			update_Q();

			//Make R
			gsl_blas_dger(-1 * (*s_k1_k2), &u_k1.vector, &v_k2.vector, R);
		}
		//Check that all rows are nonzero.
		gsl_vector_view s_k1 = gsl_matrix_row(S, k1);
		if (gsl_vector_isnull(&s_k1.vector))
		{
			gsl_vector_set_all(&s_k1.vector, 1 / double(u_components));
		}
		normalize_and_scale_u();
		normalize_and_scale_v();
	}
	for (int k2 = 0; k2 < v_components; k2++)
	{
		//Check that all columns are nonzero
		gsl_vector_view s_k2 = gsl_matrix_column(S, k2);
		if (gsl_vector_isnull(&s_k2.vector))
		{
			gsl_vector_set_all(&s_k2.vector, 1 / double(v_components));
		}
	}
	//Normalize and scale u v and S.
	normalize_and_scale_u();
	normalize_and_scale_v();
	//Compute P = US and Q=SV^T
	update_P();
	update_Q();
	return 0;
}

//Used to update with block S
int
NMTF::update_viaSBlock()
{
	//Update Via the S_kth row update equations that SR generated (see update_kth_block_of_S function for details).

	//Start with update of V. It is best to update V and S before the correspounding S because the update of S
	//is sensitive to bad V and S.
	for (int k2 = 0; k2 < v_components; k2++)
	{
		gsl_vector_view v_k2 = gsl_matrix_row(V, k2);
		gsl_vector_view p_k2 = gsl_matrix_row(P, k2);
		gsl_blas_dger(1, &p_k2.vector, &v_k2.vector, R);
		update_kth_block_of_V(k2);
		gsl_blas_dger(-1, &p_k2.vector, &v_k2.vector, R);
	}
	//Recompute the Q matrix using the new V matrix
	update_P();
	update_Q();

	//Compute U and S. Note that since S changes Q will change. This will need to be recomputed to get correct
	//partial residual R_i
	for (int k1 = 0; k1 < u_components; k1++)
	{
		gsl_vector_view u_k1 = gsl_matrix_row(U, k1);
		gsl_vector_view q_k1 = gsl_matrix_row(Q, k1);
		gsl_blas_dger(1, &u_k1.vector, &q_k1.vector, R);
		update_kth_block_of_U(k1);
		update_kth_block_of_S(k1);
		//update_Q();	      		//Need because S changes.
		gsl_blas_dger(-1, &u_k1.vector, &q_k1.vector, R);
	}

	for (int k2 = 0; k2 < v_components; k2++)
	{
		gsl_vector_view s_k2 = gsl_matrix_column(S, k2);
		if (gsl_vector_isnull(&s_k2.vector))
		{
			gsl_vector_set_all(&s_k2.vector, 1 / double(u_components));
		}
	}

	normalize_and_scale_u();
	normalize_and_scale_v();
	update_P();
	update_Q();
	write_test_files("");
	return 0;
}


int
NMTF::update_US()
{
	if (legacy)
	{
		update_U();
		update_P();
		update_S();
		normalize_and_scale_u();
		normalize_and_scale_v();
		update_P();
		update_Q();
	}else
	{
		update_U_unit();
		update_P();
		update_S_unit();
		update_P();
		update_Q();
	}
	return 0;

}

int
NMTF::update_SV()
{
	if (legacy)
	{
		update_U();
		update_P();
		update_S();
		normalize_and_scale_u();
		normalize_and_scale_v();
		update_P();
		update_Q();
	} else
	{
		update_U_unit();
		update_P();
		update_S_unit();
		update_P();
		update_Q();
	}
	return 0;
}


int
NMTF::update_P()
{
	//Computes the Product of U * S
	//gsl_matrix_memcpy(R, X);
	//gsl_blas_dgemm(CblasTrans, CblasNoTrans, -1, P, V, 1, R);
	gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1, S, U, 0, P);
	return 0;
}


int
NMTF::update_Q()
{
	//Compute the product S * V^T
	//gsl_matrix_memcpy(R, X);
	//gsl_blas_dgemm(CblasTrans, CblasNoTrans, -1, P, V, 1, R);
	gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1, S, V, 0, Q);
	return 0;
}


int
NMTF::normalize_and_scale_u()
{
	//Makes all feature length 1.
	for (int k1 = 0; k1 < u_components; k1++)
	{
		gsl_vector_view s_k1 = gsl_matrix_row(S, k1);
		gsl_vector_view u_k1 = gsl_matrix_row(U, k1);
		double norm = gsl_blas_dnrm2(&u_k1.vector);
		gsl_vector_scale(&u_k1.vector, 1 / norm);
		gsl_vector_scale(&s_k1.vector, norm);
	}
	return 0;
}


int
NMTF::normalize_and_scale_v()
{
	//Makes all feature length 1.
	for (int k2 = 0; k2 < v_components; k2++)
	{
		gsl_vector_view s_k2 = gsl_matrix_column(S, k2);
		gsl_vector_view v_k2 = gsl_matrix_row(V, k2);
		double norm = gsl_blas_dnrm2(&v_k2.vector);
		gsl_vector_scale(&v_k2.vector, 1 / norm);
		gsl_vector_scale(&s_k2.vector, norm);
	}
	return 0;
}


int
NMTF::compute_R()
{
	// Sets R = X
	gsl_matrix_memcpy(R, X);
	// Sets R = X - U S V^T
	gsl_blas_dgemm(CblasTrans, CblasNoTrans, -1, P, V, 1, R);
	return 0;
}


double
NMTF::calculate_objective()
{
	//Computes R by First setting to X then subtracting product of P V. P = U S.
	double error = pow(utils::get_frobenius_norm(R), 2);

	//Compute lU component
	double lu_reg = 0;
	if (lambdaU > 0)
	{
		gsl_matrix *U_overlap = gsl_matrix_calloc(u_components, u_components);
		gsl_blas_dsyrk(CblasUpper, CblasNoTrans, 1, U, 0, U_overlap);
		gsl_vector_view diag = gsl_matrix_diagonal(U_overlap);
		gsl_vector_set_zero(&diag.vector);
		lu_reg = lambdaU * utils::get_sum_vector_one_norm(U_overlap);
		//do not need to devide by 2 becuase U_overlap is upper diagonal with zero on diag.
		gsl_matrix_free(U_overlap);
	} else
	{
		lu_reg = 0;
	}

	//Compute LV component
	double lv_reg = 0;
	if (lambdaV > 0)
	{
		gsl_matrix *V_overlap = gsl_matrix_calloc(v_components, v_components);
		gsl_blas_dsyrk(CblasUpper, CblasNoTrans, 1, V, 0, V_overlap);
		gsl_vector_view diag = gsl_matrix_diagonal(V_overlap);
		gsl_vector_set_zero(&diag.vector);
		lv_reg = lambdaV * utils::get_sum_vector_one_norm(V_overlap);
		//do not need to devide by 2 because V_overlap is upper diagonal with zero on diag.
		gsl_matrix_free(V_overlap);
	} else
	{
		lv_reg = 0;
	}


	//Compute aU component
	double au_reg = 0;
	if (alphaU > 0)
	{
		au_reg = alphaU / 2 * utils::get_sum_vector_one_norm(U);
	} else
	{
		au_reg = 0;
	}

	//Compute aV component
	double av_reg = 0;
	if (alphaV > 0)
	{
		av_reg = alphaV / 2 * utils::get_sum_vector_one_norm(V);
	} else
	{
		av_reg = 0;
	}

	error = error + lu_reg + lv_reg + au_reg + av_reg;
	return error;
}

int
NMTF::write_test_files(string s)
{
	//Writes all matrices.
	stringstream ss;
	ss << "test/";
	io::write_dense_matrix(ss.str() + s + "U.txt", U);
	io::write_dense_matrix(ss.str() + s + "V.txt", V);
	io::write_dense_matrix(ss.str() + s + "S.txt", S);
	io::write_dense_matrix(ss.str() + s + "P.txt", P);
	io::write_dense_matrix(ss.str() + s + "Q.txt", Q);
	return 0;
}


int
NMTF::initialize_matrices(gsl_matrix *inputmat, gsl_matrix *W, gsl_matrix *H, gsl_matrix *D, gsl_matrix *O,
                          gsl_matrix *L, gsl_matrix *Ris)
{
	// Sets up required matrices (mostly need this to insure that R is computed during iterative learning)
	X = inputmat;
	n = X->size1;
	m = X->size2;
	U = W;
	V = H;
	S = D;
	P = O;
	Q = L;
	R = Ris;
	normalize_and_scale_u();
	normalize_and_scale_v();
	update_P();
	update_Q();
	//Compute R
	compute_R();
	return 0;
}

int
NMTF::fit(gsl_matrix *inputmat, gsl_matrix *W, gsl_matrix *H, gsl_matrix *D, gsl_matrix *O, gsl_matrix *L,
          gsl_matrix *Ris)
{
	initialize_matrices(inputmat, W, H, D, O, L, Ris);
	write_test_files("0");
	reconstruction_err_->clear();
	reconstruction_slope_->clear();
	int num_converge = 0;
	// Check U and V sizes.
	if ((U->size1 != u_components) || (V->size1 != v_components))
	{
		cout <<
			"The first dimension of U and V (i.e. their number of rows) should equal the number of components specified when instantiating NMF."
			<< endl;
		return 1;
	}
	//Compute R and find error
	double old_error = calculate_objective();
	reconstruction_err_->push_back(old_error);
	double old_slope;
	struct timeval begTime;
	struct timeval endTime;
	unsigned long int bt;
	unsigned long int et;
	for (int n_iter = 0; n_iter < max_iter; n_iter++)
	{
		//Update U S V
		gettimeofday(&begTime, NULL);
		if (legacy)
		{
			if (algotype == 0)
			{
				update();
			} else if (algotype == 1)
			{
				update_ijblock();
			} else if (algotype == 2)
			{
				update_viaSBlock();
			} else
			{
				cout << "Unknown algo type" << endl;
				exit(0);
			}
		} else
		{
			update_unit();
		}
		gettimeofday(&endTime, NULL);
		bt = begTime.tv_sec;
		et = endTime.tv_sec;
		if (legacy)
		{
			cout << "algotype " << algotype << " iter:" << n_iter << " time: " << et - bt << "secs " <<
				endl;
		} else
		{
			cout << "algotype unit iter:" << n_iter << " time: " << et - bt << "secs " << endl;
		}

		string n_iter_string = to_string(n_iter + 1);
		write_test_files(n_iter_string);
		//Compute R and find error.
		double error = calculate_objective();
		//Find change in error
		double slope = (old_error - error) / old_error;
		reconstruction_err_->push_back(error);
		reconstruction_slope_->push_back(slope);
		if (verbose)
		{
			cout << "Itr " << n_iter + 1 << " error = " << error << ", slope = " << slope << endl;
		}
		//Test stopping criteria
		if (0 < slope && slope < tol)
		{
			num_converge++;
			if (num_converge > 0)
			{
				if (verbose)
				{
					cout << "Converged at iteration " << n_iter + 1 << endl;
				}
				break;
			}
		} else
		{
			old_error = error;
			old_slope = slope;
			num_converge = 0;
		}
	}
	return 0;
}


int
NMTF::fit_US(gsl_matrix *inputmat, gsl_matrix *W, gsl_matrix *H, gsl_matrix *D, gsl_matrix *O, gsl_matrix *L,
             gsl_matrix *Ris)
{
	// Used in iterate learning when V Does not need to be updated
	initialize_matrices(inputmat, W, H, D, O, L, Ris);

	reconstruction_err_->clear();
	reconstruction_slope_->clear();
	int num_converge = 0;

	if ((U->size1 != u_components) || (V->size1 != v_components))
	{
		cout <<
			"The first dimension of U and V (i.e. their number of rows) should equal the number of components specified when instantiating NMF."
			<< endl;
		return 1;
	}

	double old_error = calculate_objective();
	reconstruction_err_->push_back(old_error);
	double old_slope;

	for (int n_iter = 0; n_iter < max_iter; n_iter++)
	{
		update_US();
		double error = calculate_objective();
		double slope = (old_error - error) / old_error;
		reconstruction_err_->push_back(error);
		reconstruction_slope_->push_back(slope);
		if (verbose)
		{
			cout << "Itr " << n_iter + 1 << " error = " << error << ", slope = " << slope << endl;
		}
		if (0 < slope && slope < tol)
		{
			num_converge++;
			if (num_converge > 0)
			{
				if (verbose)
				{
					cout << "Converged at iteration " << n_iter + 1 << endl;
				}
				break;
			}
		} else
		{
			old_error = error;
			old_slope = slope;
			num_converge = 0;
		}
	}
	return 0;
}

int
NMTF::fit_SV(gsl_matrix *inputmat, gsl_matrix *W, gsl_matrix *H, gsl_matrix *D, gsl_matrix *O, gsl_matrix *L,
             gsl_matrix *Ris)
{
	// Used in iterated learning when U does not need to be updated.
	initialize_matrices(inputmat, W, H, D, O, L, Ris);

	reconstruction_err_->clear();
	reconstruction_slope_->clear();
	int num_converge = 0;

	if ((U->size1 != u_components) || (V->size1 != v_components))
	{
		cout <<
			"The first dimension of U and V (i.e. their number of rows) should equal the number of components specified when instantiating NMF."
			<< endl;
		return 1;
	}

	double old_error = calculate_objective();
	reconstruction_err_->push_back(old_error);
	double old_slope;

	for (int n_iter = 0; n_iter < max_iter; n_iter++)
	{
		update_SV();
		double error = calculate_objective();
		double slope = (old_error - error) / old_error;
		reconstruction_err_->push_back(error);
		reconstruction_slope_->push_back(slope);
		if (verbose)
		{
			cout << "Itr " << n_iter + 1 << " error = " << error << ", slope = " << slope << endl;
		}
		if (0 < slope && slope < tol)
		{
			num_converge++;
			if (num_converge > 0)
			{
				if (verbose)
				{
					cout << "Converged at iteration " << n_iter + 1 << endl;
				}
				break;
			}
		} else
		{
			old_error = error;
			old_slope = slope;
			num_converge = 0;
		}
	}
	return 0;
}


int
NMTF::increase_k1_fixed_k2(int k1, gsl_matrix *inputmat, gsl_matrix *W, gsl_matrix *H, gsl_matrix *D, gsl_matrix *O,
                           gsl_matrix *L, gsl_matrix *Ris, gsl_matrix *U_new, gsl_matrix *S_new, gsl_matrix *P_new,
                           gsl_matrix *Q_new, const gsl_rng *ri)
{
	//Iterated learning when k1 increases but k2 is fixed. Insure R is initialized correctly
	initialize_matrices(inputmat, W, H, D, O, L, Ris);
	//Number of factors to learn
	int k1_diff = k1 - u_components;
	//Initialize New matrices for partial learning on old Residual
	gsl_matrix *U_diff = gsl_matrix_calloc(k1_diff, n);
	gsl_matrix *P_diff = gsl_matrix_calloc(v_components, n);
	gsl_matrix *Q_diff = gsl_matrix_calloc(k1_diff, m);
	gsl_matrix *S_diff = gsl_matrix_calloc(k1_diff, v_components);
	//R_diff is used to store residual when running subNMTF problem.
	gsl_matrix *R_diff = gsl_matrix_calloc(n, m);
	gsl_matrix *R_abs = gsl_matrix_calloc(n, m);
	//Inforce positivity in R_abs matrix that will be used as the new data to fit.
	utils::matrix_abs(R, R_abs);

	init::random(U_diff, ri);
	init::random(S_diff, ri);
	string out_dir("temp/");

	//Fit U_diff V S_diff to R_abs;
	NMTF subNMTF = NMTF(k1_diff, v_components, init, max_iter, random_state, verbose, tol, reconstruction_err_,
	                    reconstruction_slope_, alphaU, lambdaU, alphaV, lambdaV);
	//Here we are trying to find components that capture features not captured by U V S
	subNMTF.fit_US(R_abs, U_diff, V, S_diff, P_diff, Q_diff, R_diff);

	// The following was used to remove new factors from old factors.
	//io::write_nmtf_output(U_diff, V, S_diff, R_abs, reconstruction_err_, reconstruction_slope_, out_dir);

	/*for(int i=0; i < k1_diff; i++){
                gsl_vector_view U_k = gsl_matrix_row(U_diff, i);
                subtract_factors(U, &U_k.vector);
        }*/


	//Concatinate the matrices
	utils::concat_matrix_rows(U, U_diff, U_new);
	utils::concat_matrix_rows(S, S_diff, S_new);
	//io::write_nmtf_output(U_new, V, S_new, R_abs, reconstruction_err_, reconstruction_slope_, out_dir);

	//Use new matrices to fit original data.
	u_components = k1;
	fit(X, U_new, V, S_new, P_new, Q_new, R);
	//io::write_nmtf_output(U_new, V, S_new, R, reconstruction_err_, reconstruction_slope_, out_dir);
	gsl_matrix_free(U_diff);
	gsl_matrix_free(P_diff);
	gsl_matrix_free(Q_diff);
	gsl_matrix_free(S_diff);
	gsl_matrix_free(R_diff);
	gsl_matrix_free(R_abs);
	return 0;
}

int
NMTF::increase_k2_fixed_k1(int k2, gsl_matrix *inputmat, gsl_matrix *W, gsl_matrix *H, gsl_matrix *D, gsl_matrix *O,
                           gsl_matrix *L, gsl_matrix *Ris, gsl_matrix *V_new, gsl_matrix *S_new, gsl_matrix *P_new,
                           gsl_matrix *Q_new, const gsl_rng *ri)
{
	//Iterated learning when k2 increases and k1 is fixed. Same as above.
	initialize_matrices(inputmat, W, H, D, O, L, Ris);
	int k2_diff = k2 - v_components;
	gsl_matrix *V_diff = gsl_matrix_calloc(k2_diff, m);
	gsl_matrix *P_diff = gsl_matrix_calloc(k2_diff, n);
	gsl_matrix *Q_diff = gsl_matrix_calloc(u_components, m);
	gsl_matrix *S_diff = gsl_matrix_calloc(u_components, k2_diff);
	gsl_matrix *R_diff = gsl_matrix_calloc(n, m);
	gsl_matrix *R_abs = gsl_matrix_calloc(n, m);
	utils::matrix_abs(R, R_abs);

	init::random(V_diff, ri);
	init::random(S_diff, ri);
	string out_dir("temp/");
	NMTF subNMTF = NMTF(u_components, k2_diff, init, max_iter, random_state, verbose, tol, reconstruction_err_,
	                    reconstruction_slope_, alphaU, lambdaU, alphaV, lambdaV);
	subNMTF.fit_SV(R_abs, U, V_diff, S_diff, P_diff, Q_diff, R_diff);

	// The following was used to remove new factors from old factors
	//io::write_nmtf_output(U, V_diff, S_diff, R_abs, reconstruction_err_, reconstruction_slope_, out_dir);

	/*for(int i=0; i < k2_diff; i++){
                gsl_vector_view V_k = gsl_matrix_row(V_diff, i);
                subtract_factors(V, &V_k.vector);
        }*/


	utils::concat_matrix_rows(V, V_diff, V_new);
	utils::concat_matrix_columns(S, S_diff, S_new);
	//io::write_nmtf_output(U, V_new, S_new, R, reconstruction_err_, reconstruction_slope_, out_dir);

	//Fit new matrices to orgianal data.
	v_components = k2;
	fit(X, U, V_new, S_new, P_new, Q_new, R);
	//io::write_nmtf_output(U, V_new, S_new, R, reconstruction_err_, reconstruction_slope_, out_dir);
	gsl_matrix_free(V_diff);
	gsl_matrix_free(S_diff);
	gsl_matrix_free(P_diff);
	gsl_matrix_free(Q_diff);
	gsl_matrix_free(R_diff);
	gsl_matrix_free(R_abs);
	return 0;
}

int
NMTF::increase_k1_k2(int k1, int k2, gsl_matrix *inputmat, gsl_matrix *W, gsl_matrix *H, gsl_matrix *D,
                     gsl_matrix *O, gsl_matrix *L, gsl_matrix *Ris, gsl_matrix *U_new, gsl_matrix *V_new,
                     gsl_matrix *S_new, gsl_matrix *P_new, gsl_matrix *Q_new, const gsl_rng *ri)
{
	//Used when both k1 and k2 incerases
	initialize_matrices(inputmat, W, H, D, O, L, Ris);

	int k1_diff = k1 - u_components;
	int k2_diff = k2 - v_components;

	//initialize all matrices: Here we have U_diff, V_diff, and S_diff
	gsl_matrix *U_diff = gsl_matrix_calloc(k1_diff, n);
	gsl_matrix *V_diff = gsl_matrix_calloc(k2_diff, m);
	gsl_matrix *P_diff = gsl_matrix_calloc(k2_diff, n);
	gsl_matrix *Q_diff = gsl_matrix_calloc(k1_diff, m);
	gsl_matrix *S_diff = gsl_matrix_calloc(k1_diff, k2_diff);
	gsl_matrix *R_diff = gsl_matrix_calloc(n, m);
	gsl_matrix *R_abs = gsl_matrix_calloc(n, m);;
	utils::matrix_abs(R, R_abs);

	init::random(U_diff, ri);
	init::random(V_diff, ri);
	init::random(S_diff, ri);

	//Fit U_diff, V_diff, S_diff to old matrices
	NMTF subNMTF = NMTF(k1_diff, k2_diff, init, max_iter, random_state, verbose, tol, reconstruction_err_,
	                    reconstruction_slope_, alphaU, lambdaU, alphaV, lambdaV);
	subNMTF.fit(R_abs, U_diff, V_diff, S_diff, P_diff, Q_diff, R_diff);

	// Used to remove learned factors from old factors
	//io::write_nmtf_output(U_diff, V_diff, S_diff, R_abs, reconstruction_err_, reconstruction_slope_, out_dir);

	/*for(int i=0; i < k1_diff; i++){
		gsl_vector_view U_k = gsl_matrix_row(U_diff, i);
		subtract_factors(U, &U_k.vector);
	}*/
	utils::concat_matrix_rows(U, U_diff, U_new);

	//Used to remove learned factors from old factors.
	/*for(int i=0; i < k2_diff; i++){
		gsl_vector_view V_k = gsl_matrix_row(V_diff, i);
		subtract_factors(V, &V_k.vector);
	}*/
	utils::concat_matrix_rows(V, V_diff, V_new);

	utils::concat_matrix_diagonal(S, S_diff, S_new);
	//io::write_nmtf_output(U_new, V_new, S_new, R, reconstruction_err_, reconstruction_slope_, out_dir);
	u_components = k1;
	v_components = k2;

	//Refitt new matrices to original matrices
	fit(X, U_new, V_new, S_new, P_new, Q_new, R);

	//io::write_nmtf_output(U_new, V_new, S_new, R, reconstruction_err_, reconstruction_slope_, out_dir);
	gsl_matrix_free(U_diff);
	gsl_matrix_free(V_diff);
	gsl_matrix_free(S_diff);
	gsl_matrix_free(R_diff);
	gsl_matrix_free(R_abs);
	return 0;
}

int
NMTF::subtract_factors(gsl_matrix *A, gsl_vector *b)
{
	//Function is used to subtract new factors from old factors in interative learning. The thought is that this would increase orthogonality between factors.
	int nFactors = A->size1;
	int nTerms = A->size2;
	for (int i = 0; i < nFactors; i++)
	{
		gsl_vector_view a_k = gsl_matrix_row(A, i);
		gsl_vector_sub(&a_k.vector, b);
		//inforce non_negativity constraint.∂
		for (int j = 0; j < nTerms; j++)
		{
			double *val = &((&a_k.vector)->data[j]);
			if (*val < 0)
			{
				*val = 0;
			}
		}
	}
	return 0;
}

int
NMTF::update_kth_block_of_U_unit(int k)
{
	// Update U_k  by taking the product R * q_k1.
	gsl_vector_view u_k = gsl_matrix_row(U, k);
	gsl_vector_view q_k = gsl_matrix_row(Q, k);
	gsl_blas_dgemv(CblasNoTrans, 1, R, &q_k.vector, 0, &u_k.vector);

	//Then enforce non-negativity and unit norm.
	enforce_min_val(&u_k.vector);
	unit_normalize(&u_k.vector);
	return 0;
}

int
NMTF::update_kth_block_of_V_unit(int k)
{
	//Update v_k by taking the product RT p
	gsl_vector_view p_k = gsl_matrix_row(P, k);
	gsl_vector_view v_k = gsl_matrix_row(V, k);
	gsl_blas_dgemv(CblasTrans, 1, R, &p_k.vector, 0, &v_k.vector);

	//Then enforce non-negativity and unit norm.
	enforce_min_val(&v_k.vector);
	unit_normalize(&v_k.vector);
	return 0;
}


int
NMTF::apply_orthog_u_unit(int k)
{
	if (lambdaU > 0)
	{
		gsl_vector_view u_k = gsl_matrix_row(U, k);
		gsl_vector *u_others = gsl_vector_calloc(n);
		for (int j = 0; j < u_components; j++)
		{
			if (j != k)
			{
				gsl_vector_view u_j = gsl_matrix_row(U, j);
				gsl_vector_add(u_others, &u_j.vector);
			}
		}
		unit_normalize(u_others);
		gsl_vector_scale(u_others, lambdaU);
		//subtract regularization term.
		gsl_vector_sub(&u_k.vector, u_others);
		gsl_vector_free(u_others);

		//Then enforce non-negativity and unit norm
		enforce_min_val(&u_k.vector);
		unit_normalize(&u_k.vector);
	}
	return 0;
}


int
NMTF::apply_sparsity_u_unit(int k)
{
	if (alphaU > 0)
	{
		gsl_vector_view u_k = gsl_matrix_row(U, k);
		enforce_min_val(&u_k.vector, alphaU);
		unit_normalize(&u_k.vector);
	}
	return 0;
}


int
NMTF::apply_orthog_v_unit(int k)
{
	if (lambdaV > 0)
	{
		gsl_vector_view v_k = gsl_matrix_row(V, k);
		gsl_vector *v_others = gsl_vector_calloc(m);
		for (int j = 0; j < v_components; j++)
		{
			if (j != k)
			{
				gsl_vector_view v_j = gsl_matrix_row(V, j);
				gsl_vector_add(v_others, &v_j.vector);
			}
		}
		unit_normalize(v_others);
		gsl_vector_scale(v_others, lambdaV);
		//subtract regularization term.
		gsl_vector_sub(&v_k.vector, v_others);
		gsl_vector_free(v_others);

		//Then enforce non-negativity and unit norm
		enforce_min_val(&v_k.vector);
		unit_normalize(&v_k.vector);
	}
	return 0;
}

int
NMTF::apply_sparsity_v_unit(int k)
{
	if (alphaV > 0)
	{
		gsl_vector_view v_k = gsl_matrix_row(V, k);
		enforce_min_val(&v_k.vector, alphaV);
		unit_normalize(&v_k.vector);
	}
	return 0;
}

int
NMTF::enforce_min_val(gsl_vector * x)
{
	double *val;
	for (int i = 0; i < x->size; i++)
	{
		val = gsl_vector_ptr(x, i);
		if (*val < 0)
		{
			*val = 0;
		}
	}
	return 0;
}


int
NMTF::enforce_min_val(gsl_vector *x, double alpha)
{
	double *val;
	for (int i = 0; i < x->size; i++)
	{
		val = gsl_vector_ptr(x, i);
		if (*val < alpha)
		{
			*val = 0;
		}
	}
	return 0;
}

int
NMTF::unit_normalize(gsl_vector *x)
{
	double norm = gsl_blas_dnrm2(x);
	if (norm > 0)
	{
		gsl_vector_scale(x, 1 / norm);
	} else
	//If norm is zero set all values to uniform val and normalize
	{
		gsl_vector_set_all(x, 1);
		norm = gsl_blas_dnrm2(x);
		gsl_vector_scale(x, 1 / norm);
	}
	return 0;
}


int
NMTF::update_U_unit()
{
	for (int k1 = 0; k1 < u_components; k1++)
	{
		gsl_vector_view u_k1 = gsl_matrix_row(U, k1);
		gsl_vector_view q_k1 = gsl_matrix_row(Q, k1);
		//Make R_i see equation for u_k update above for datail
		gsl_blas_dger(1, &u_k1.vector, &q_k1.vector, R);
		update_kth_block_of_U_unit(k1);
		apply_orthog_u_unit(k1);
		apply_sparsity_u_unit(k1);
		gsl_blas_dger(-1, &u_k1.vector, &q_k1.vector, R);
	}
	return 0;
}


int
NMTF::update_V_unit()
{
	for (int k2 = 0; k2 < v_components; k2++)
	{
		gsl_vector_view v_k2 = gsl_matrix_row(V, k2);
		gsl_vector_view p_k2 = gsl_matrix_row(P, k2);

		gsl_blas_dger(1, &p_k2.vector, &v_k2.vector, R);
		update_kth_block_of_V_unit(k2);
		apply_orthog_v_unit(k2);
		apply_sparsity_v_unit(k2);
		gsl_blas_dger(-1, &p_k2.vector, &v_k2.vector, R);
	}
	return 0;
}

int
NMTF::update_unit()
{
	update_U_unit();
	update_P();

	update_V_unit();
	update_Q();

	update_S_unit();
	update_P();
	update_Q();
	return 0;
}

int
NMTF::update_ith_jth_of_S_unit(int k1, int k2)
{
	//update the ith and jth element at a time. Picked this method because no computation of inverse needed.
	// R_ij = X - sum_{c ne i} sum_{d ne j} u_c s_{c,d} v_d^T \\rank 1 matrix corresponding to the the c and d vector.
	//Update equation s_i,j = u_i^T R_ij v_j / (|| u_i||_2^2 ||v_j||_2^2)
	gsl_vector *temp = gsl_vector_alloc(n);
	double *s_k1_k2 = gsl_matrix_ptr(S, k1, k2);
	gsl_vector_view u_k1 = gsl_matrix_row(U, k1);
	gsl_vector_view v_k2 = gsl_matrix_row(V, k2);

	gsl_blas_dgemv(CblasNoTrans, 1, R, &v_k2.vector, 0, temp);
	gsl_blas_ddot(&u_k1.vector, temp, s_k1_k2);
	if (*s_k1_k2 < 0)
	{
		*s_k1_k2 = 0;
	}

	gsl_vector_free(temp);
	return 0;
}


int
NMTF::update_S_unit()
{
	for (int k1 = 0; k1 < u_components; k1++)
	{
		for (int k2 = 0; k2 < v_components; k2++)
		{
			double *s_k1_k2 = gsl_matrix_ptr(S, k1, k2);
			gsl_vector_view u_k1 = gsl_matrix_row(U, k1);
			gsl_vector_view v_k2 = gsl_matrix_row(V, k2);
			//Make R_i,j See equation for s_ij update above for details
			gsl_blas_dger(*s_k1_k2, &u_k1.vector, &v_k2.vector, R);
			//Update S_ij
			update_ith_jth_of_S(k1, k2);
			//Make R
			gsl_blas_dger(-*s_k1_k2, &u_k1.vector, &v_k2.vector, R);
		}
		//Check that all rows are nonzero.
		gsl_vector_view s_k1 = gsl_matrix_row(S, k1);
		if (gsl_vector_isnull(&s_k1.vector))
		{
			gsl_vector_set_all(&s_k1.vector, 1);
			unit_normalize(&s_k1.vector);
		}
	}

	for (int k2 = 0; k2 < v_components; k2++)
	{
		//Check that all columns are nonzero
		gsl_vector_view s_k2 = gsl_matrix_column(S, k2);
		if (gsl_vector_isnull(&s_k2.vector))
		{
			gsl_vector_set_all(&s_k2.vector, 1);
			unit_normalize(&s_k2.vector);
		}
	}
	return 0;
}


int
NMTF::reset_k1_k2(int new_k1, int new_k2)
{
	//Used to change k1 and k2 for an NMTF object. Used in iterative learning
	u_components = new_k1;
	v_components = new_k2;
	return 0;
}

int
NMTF::set_size(int a, int b)
{
	n=a;
	m=b;
	return 0;
}


int
NMTF::setAlgotype(int val)
{
	algotype = val;
	return 0;
}

int
NMTF::setLegacy(bool leg)
{
	legacy = leg;
	return 0;
}








