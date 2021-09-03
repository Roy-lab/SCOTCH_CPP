#include <gsl/gsl_matrix.h>
#include <list>
#ifndef _nmtf_
#define _nmtf_
using namespace std;

enum init_method {nndsvd_init, random_init};

class NMTF {
	public:
		NMTF(int, init_method, int, int, bool, double, list<double>*);
		NMTF(int, int, init_method, int, int, bool, double, list<double>*);
		NMTF(int, int, init_method, int, int, bool, double, list<double>*, double, double);
		NMTF(int, int, init_method, int, int, bool, double, list<double>*, double, double, double, double);
		
		~NMTF();
		int fit(gsl_matrix*, gsl_matrix*, gsl_matrix*, gsl_matrix*);
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
		list<double>*  reconstruction_err_;

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
		int update();
		int normalize_and_scale_u();
		int normalize_and_scale_v();
		int write_test_files(string);
		double calculate_objective();
};
#endif
