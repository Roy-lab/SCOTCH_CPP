#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_blas.h>
#include <vector>
#include <cmath>
#include <iostream>
#include "node.h"
#include "root.h"
#include "utils.h"

int Root::update() {
	gsl_matrix_set_zero(V);
	for (vector<Node*>::iterator itr=children.begin(); itr != children.end(); ++itr) {
		gsl_matrix_add(V, (*itr)->get_V());
	}	
	gsl_matrix_scale(V, 1.0/children.size());
	return 0;
}

double Root::calculate_objective() {
	return 0;
}

