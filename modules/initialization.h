#include <gsl/gsl_matrix.h>
#include <gsl/gsl_rng.h>
#ifndef _init_
#define _init_
using namespace std;

namespace init{
	int nndsvd(gsl_matrix*, gsl_matrix*, gsl_matrix*, int);
	int random(gsl_matrix*, gsl_matrix*, gsl_matrix*, int);
	int random(gsl_matrix*, gsl_matrix*, int);
	int random(gsl_matrix*, int);
	int random(gsl_matrix*, gsl_rng*);
};
#endif
