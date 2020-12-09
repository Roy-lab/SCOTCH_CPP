#include <gsl/gsl_matrix.h>
#include <list>
#include <vector>
#include "node.h"
#ifndef _tmi_
#define _tmi_
using namespace std;

class TMI {
	public:
		TMI(int k, int nitr, int seed, bool verb, double t, double a, double l) : 
				n_components(k),
				max_iter(nitr),
				random_state(seed),
				verbose(verb),
				tol(t),
				alpha(a),
				lambda(l) {};

		~TMI() {
			for (vector<Node*>::iterator itr=tree.begin(); itr!=tree.end(); ++itr) {
				gsl_matrix_free((*itr)->get_V());
				if ((*itr)->is_leaf()) {
					gsl_matrix_free(((Leaf*)(*itr))->get_X());
					gsl_matrix_free(((Leaf*)(*itr))->get_U());
				}
			}
			int l = tree.size();
			for (int i = 0; i < l; i++) {
				delete tree[i];
			}
		};

		int fit();
		int make_tree(vector<int>&, vector<string>&, vector<string>&, vector<int>&, int);
		int make_tree_asymm(vector<int>&, vector<string>&, vector<string>&, vector<int>&, int);
		int print_factors(string);

		vector<Node*>* get_tree(){return &tree;};
		list<double>* get_errors(){return &reconstruction_err_;};

	private:
		int n_components;
		int max_iter;
		int random_state;
		bool verbose;
		double tol;
		double alpha;
		double lambda;
		vector<Node*> tree;
		list<double> reconstruction_err_;
};
#endif
