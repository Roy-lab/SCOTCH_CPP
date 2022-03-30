#include <gsl/gsl_matrix.h>
#include <list>
#ifndef _nmtf_
#define _nmtf_
using namespace std;

enum init_method {nndsvd_init, random_init};

class NMTF {
	public:
		NMTF(int, init_method, int, int, bool, double, list<double>*, list<double>*);
		NMTF(int, int, init_method, int, int, bool, double, list<double>*, list<double>*);
		NMTF(int, int, init_method, int, int, bool, double, list<double>*, list<double>*, double, double);
		NMTF(int, int, init_method, int, int, bool, double, list<double>*, list<double>*, double, double, double, double);
			
		~NMTF();
		int initialize_matrices(gsl_matrix*, gsl_matrix*, gsl_matrix*, gsl_matrix*, gsl_matrix*);
		int fit_US(gsl_matrix*, gsl_matrix*, gsl_matrix*, gsl_matrix*, gsl_matrix*);
		int fit_SV(gsl_matrix*, gsl_matrix*, gsl_matrix*, gsl_matrix*, gsl_matrix*);
		int fit(gsl_matrix*, gsl_matrix*, gsl_matrix*, gsl_matrix*, gsl_matrix*);
		int increase_k1_fixed_k2(int, gsl_matrix*, gsl_matrix*, gsl_matrix*, gsl_matrix*, gsl_matrix*, gsl_matrix*, gsl_matrix*, gsl_rng*);
		int increase_k2_fixed_k1(int, gsl_matrix*, gsl_matrix*, gsl_matrix*, gsl_matrix*, gsl_matrix*, gsl_matrix*, gsl_matrix*, gsl_rng*);
		int increase_k1_k2(int, int, gsl_matrix*, gsl_matrix*, gsl_matrix*, gsl_matrix*, gsl_matrix*, gsl_matrix*, gsl_matrix*, gsl_matrix*, gsl_rng*);
		int compute_R();
		int reset_k1_k2(int, int);
		int u_components;
		int v_components;
		int n;
		int m;
		int test;
		double lambdaU;
		double lambdaV;
		double alphaU;
		double alphaV;
		init_method init;
		int max_iter;
		int random_state;
		bool verbose;
		double tol;
		list<double>* reconstruction_err_;
		list<double>* reconstruction_slope_;
		
	private:
		gsl_matrix* X;
		gsl_matrix* U;
		gsl_matrix* V;
		gsl_matrix* S;
		gsl_matrix* R;
		gsl_matrix* P;
		gsl_matrix* Q;
		
		int update_P();
		int update_Q();
		int update_kth_block_of_U(int);
		int update_kth_block_of_V(int);
		int update_ith_jth_of_S(int, int);
		int update_US();
		int update_SV();
		int update();
		int normalize_and_scale_u();
		int normalize_and_scale_v();
		int write_test_files(string);
		int subtract_factors(gsl_matrix*, gsl_vector*);		
		double calculate_objective();
};
#endif
