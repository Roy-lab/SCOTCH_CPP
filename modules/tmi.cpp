#include <gsl/gsl_matrix.h>
#include <list>
#include <iostream>
#include <string>
#include <vector>
#include <numeric>
#include "node.h"
#include "nmf.h"
#include "leaf.h"
#include "branch.h"
#include "root.h"
#include "io.h"
#include "utils.h"
#include "initialization.h"
#include "tmi.h"

int TMI::fit() {
	double old_error = 0;
	for(vector<Node*>::iterator node=tree.begin(); node!=tree.end(); ++node) {
		old_error += (*node)->calculate_objective();		
	}
	double old_slope;

	for (int n_iter =0; n_iter < max_iter; n_iter++){
		for(vector<Node*>::iterator node=tree.begin(); node!=tree.end(); ++node) {
			(*node)->update();
		}
		double error = 0;
		for(vector<Node*>::iterator node=tree.begin(); node!=tree.end(); ++node) {
			error += (*node)->calculate_objective();
		}
		reconstruction_err_.push_back(error);
		double slope = old_error - error;
		if (verbose) {
			cout << "Itr " << n_iter << " error = " << error << ", slope = " << slope << endl;
		}
		if (slope < tol) {
			if (verbose) {
				cout << "Converged at iteration " << n_iter << endl;	
			}
			break;
		} else {
			old_error = error;
			old_slope = slope;
		}
	}
	return 0;
}

int TMI::make_tree(vector<int>& parentIds, vector<string>& aliases, vector<string>& inputFiles, vector<int>& n, int m) {
	int nTasks = inputFiles.size();
	int nNodes = parentIds.size();

	int nTotal = accumulate(n.begin(), n.end(), 0);
	gsl_matrix* monsterX = gsl_matrix_calloc(nTotal, m);
	int offset = 0;
	for (int i = 0; i < nTasks; i++) {
		gsl_matrix_view currX = gsl_matrix_submatrix(monsterX, offset, 0, n[i], m);
		io::read_dense_matrix(inputFiles[i], &currX.matrix);
		offset += n[i];	
	}

	gsl_matrix* monsterU = gsl_matrix_alloc(n_components, nTotal);
	gsl_matrix* V1 = gsl_matrix_alloc(n_components, m);

	if (verbose) {
		cout << "Initializing with joint NMF:" << endl;
	}
	NMF nmf = NMF(n_components, nndsvd_init, 100, random_state, true, 1);
	nmf.fit(monsterX, monsterU, V1);
	if (verbose) {
		cout << "Initialization done." << endl;
	}

	offset = 0;
	for (int i = 0; i < nNodes; i++) {
		gsl_matrix* V = gsl_matrix_alloc(n_components, m);
		gsl_matrix_memcpy(V, V1);
		if (i < nTasks) {
			gsl_matrix_view currX = gsl_matrix_submatrix(monsterX, offset, 0, n[i], m);
			gsl_matrix_view currU = gsl_matrix_submatrix(monsterU, 0, offset, n_components, n[i]);
			gsl_matrix* X = gsl_matrix_alloc(n[i], m);
			gsl_matrix_memcpy(X, &currX.matrix);
			gsl_matrix* U = gsl_matrix_alloc(n_components, n[i]);	
			gsl_matrix_memcpy(U, &currU.matrix);
			Leaf* leaf = new Leaf(X, U, V, n_components, alpha, lambda, aliases[i]);
			tree.push_back(leaf);
			offset += n[i];		
		} else {
			if (parentIds[i] == -1) {
				Root* root = new Root(V, n_components, alpha, aliases[i]);
				tree.push_back(root);	
			}	else {
				Branch* branch = new Branch(V, n_components, alpha, aliases[i]);
				tree.push_back(branch);
			}
		}		
	}

	gsl_matrix_free(monsterX);
	gsl_matrix_free(V1);
	gsl_matrix_free(monsterU);

	//link parents and children:
	for (int i =0; i < nNodes; i++) {
		int parentId = parentIds[i]-1;
		if (parentId > 0) {
			Node* child = tree[i];
			Node* parent = tree[parentId]; 
			child->set_parent(parent);	
			parent->add_child(child);
		}
	}
	return 0;
}

 
int 
TMI::make_tree_asymm(vector<int>& parentIds, vector<string>& aliases, vector<string>& inputFiles, vector<int>& n, int m) {
	int nTasks = inputFiles.size();
	int nNodes = parentIds.size();

	//make first leaf node:
	gsl_matrix* X1 = gsl_matrix_calloc(n[0],m);
	//io::read_sparse_matrix(inputFiles[0], X1);
	io::read_dense_matrix(inputFiles[0], X1);
	gsl_matrix* U1 = gsl_matrix_alloc(n_components, n[0]);
	gsl_matrix* V1 = gsl_matrix_alloc(n_components, m);
	NMF nmf = NMF(n_components, nndsvd_init, 200, random_state, false, 1);
	nmf.fit(X1, U1, V1);
	Leaf* leaf1 = new Leaf(X1, U1, V1, n_components, alpha, lambda, aliases[0]);
	tree.push_back(leaf1); 

	//make rest of nodes:
	// Vs initialized to Vs of first leaf node.
	// Us initialized to random
	for (int i = 1; i < nNodes; i++) {
		gsl_matrix* V = gsl_matrix_alloc(n_components, m);
		gsl_matrix_memcpy(V, V1);
		if (i < nTasks) {
			gsl_matrix* X = gsl_matrix_calloc(n[i],m);
			//io::read_sparse_matrix(inputFiles[i], X);
			io::read_dense_matrix(inputFiles[i], X);
			gsl_matrix* U = gsl_matrix_alloc(n_components, n[i]);
			//SR did this
			init::random(U,random_state);
			//gsl_matrix_memcpy(U, U1);
			Leaf* leaf = new Leaf(X, U, V, n_components, alpha, lambda, aliases[i]);
			tree.push_back(leaf);
		} else {
			if (parentIds[i] == -1) {
				Root* root = new Root(V, n_components, alpha, aliases[i]);
				tree.push_back(root);	
			}	else {
				Branch* branch = new Branch(V, n_components, alpha, aliases[i]);
				tree.push_back(branch);
			}
		}		
	}

	//link parents and children:
	for (int i =0; i < nNodes; i++) {
		int parentId = parentIds[i]-1;
		if (parentId > 0) {
			Node* child = tree[i];
			Node* parent = tree[parentId]; 
			child->set_parent(parent);	
			parent->add_child(child);
		}
	}
	return 0;
} 

int TMI::print_factors(string prefix) {
	for (vector<Node*>::iterator itr=tree.begin(); itr!=tree.end(); ++itr) {
		(*itr)->write_factors_to_file(prefix);
	}
}
