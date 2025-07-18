#include <gsl/gsl_matrix.h>

#include <list>
#ifndef _nmtf_
#define _nmtf_
using namespace std;

enum init_method { nndsvd_init, random_init };

class NMTF
{
	public:
	NMTF();
	NMTF(int, init_method, int, int, bool, double, list<double> *, list<double> *);

	NMTF(int, int, init_method, int, int, bool, double, list<double> *, list<double> *);

	NMTF(int, int, init_method, int, int, bool, double, list<double> *, list<double> *, double, double);

	NMTF(int, int, init_method, int, int, bool, double, list<double> *, list<double> *, double, double, double,
	     double);

	~NMTF();

	int
	set_NMTF_params(int k1, int k2, init_method initMethod, int maxIter, int seed, bool verb, double termTol,
	                list<double> *err, list<double> *slope, double aU, double lU, double aV, double lV);

	int
	set_NMTF_params_python(int k1, int k2, int maxIter, int seed, bool verb, double termTol,
			list<double> *err, list<double> *slope, double aU, double lU, double aV, double lV);

	int
	set_NMTF_params(init_method initMethod, int maxIter, int seed, bool verb, double termTol,
			list<double> *err, list<double> *slope, double aU, double lU, double aV, double lV);


	int
	initialize_matrices(gsl_matrix *, gsl_matrix *, gsl_matrix *, gsl_matrix *, gsl_matrix *, gsl_matrix *,
	                    gsl_matrix *);

	int
	fit_US(gsl_matrix *, gsl_matrix *, gsl_matrix *, gsl_matrix *, gsl_matrix *, gsl_matrix *, gsl_matrix *);

	int
	fit_SV(gsl_matrix *, gsl_matrix *, gsl_matrix *, gsl_matrix *, gsl_matrix *, gsl_matrix *, gsl_matrix *);

	int
	fit(gsl_matrix *, gsl_matrix *, gsl_matrix *, gsl_matrix *, gsl_matrix *, gsl_matrix *, gsl_matrix *);

	int
	increase_k1_fixed_k2(int, gsl_matrix *, gsl_matrix *, gsl_matrix *, gsl_matrix *, gsl_matrix *, gsl_matrix *,
	                     gsl_matrix *, gsl_matrix *, gsl_matrix *, gsl_matrix *, gsl_matrix *, const gsl_rng *);

	int
	increase_k2_fixed_k1(int, gsl_matrix *, gsl_matrix *, gsl_matrix *, gsl_matrix *, gsl_matrix *, gsl_matrix *,
	                     gsl_matrix *, gsl_matrix *, gsl_matrix *, gsl_matrix *, gsl_matrix *, const gsl_rng *);

	int
	increase_k1_k2(int, int, gsl_matrix *, gsl_matrix *, gsl_matrix *, gsl_matrix *, gsl_matrix *, gsl_matrix *,
	               gsl_matrix *, gsl_matrix *, gsl_matrix *, gsl_matrix *, gsl_matrix *, gsl_matrix *, const gsl_rng *);

	int
	compute_R();

	int
	reset_k1_k2(int, int);

	int
	set_size(int, int);

	int
	setAlgotype(int);

	int
	setLegacy(bool);

	int
	get_algotype();

	int
	set_test(bool);


	gsl_matrix *X;
	gsl_matrix *U;
	gsl_matrix *V;
	gsl_matrix *S;
	gsl_matrix *R;
	gsl_matrix *P;
	gsl_matrix *Q;
	int u_components;
	int v_components;
	int n;
	int m;
	double lambdaU;
	double lambdaV;
	double alphaU;
	double alphaV;
	init_method init;
	int max_iter;
	int random_state;
	bool verbose;
	double tol;
	list<double> *reconstruction_err_;
	list<double> *reg_err_;
	list<double> *reconstruction_slope_;

	bool test;

	string outpath;
	private:
	int algotype;
	bool legacy;


	//Legacy update equations
	int
	update_P();

	int
	update_Q();

	int
	update_kth_block_of_U(int);

	int
	update_kth_block_of_V(int);

	int
	update_ith_jth_of_S(int, int);

	int
	update_kth_block_of_S(int);

	int
	update_US();

	int
	update_SV();

	int
	apply_orthog_u(int, double);

	int
	apply_sparsity_u(int, double);

	int
	apply_orthog_v(int, double);

	int
	apply_sparsity_v(int, double);

	int
	update_U();

	int
	update_V();

	int
	update_S();

	int
	update();


	//Updates with unit regularization
	int
	update_kth_block_of_U_unit(int);

	int
	update_kth_block_of_V_unit(int);

	int
	update_ith_jth_of_S_unit(int, int);

	int
	apply_orthog_u_unit(int);

	int
	apply_sparsity_u_unit(int);

	int
	apply_orthog_v_unit(int);

	int
	apply_sparsity_v_unit(int);

	int
	update_U_unit();

	int
	update_V_unit();

	int
	update_S_unit();

	int
	update_unit();

	int
	enforce_min_val(gsl_vector *);

	int
	enforce_min_val(gsl_vector *, double); //We add in alpha here just to save computations. Defualts to zero.


	int
	unit_normalize(gsl_vector *);


	int
	update_viaSBlock();

	//SR editted
	int
	update_ijblock();

	int
	normalize_and_scale_u();

	int
	normalize_and_scale_v();

	int
	write_test_files(string);

	int
	subtract_factors(gsl_matrix *, gsl_vector *);

	double
	calculate_objective();
};
#endif
