#include <iostream>
#include <fstream>
#include <gsl/gsl_matrix.h>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <list>
#include <math.h>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include "io.h"

int io::print_usage(const string inputFile) {
	ifstream f(inputFile.c_str());
	string line;
	while(getline(f, line)) {
		cout << line << endl;
	}
	f.close();
	return 0;
}

int io::write_dense_matrix(const string outputFile, gsl_matrix* X) {
	int rowNum = X-> size1;
	int colNum = X-> size2;
	ofstream ofs;
	ofs.open(outputFile.c_str());
	for (int i = 0; i < rowNum; i++) {
		for (int j = 0; j < colNum; j++) {
			ofs << X->data[i * X->tda + j];
			if (j < colNum - 1) {
				ofs << "\t";
			}
		}	
		ofs << endl;
	}
	ofs.close();
	return 0;
}

int io::read_sparse_matrix(const string inputFile, gsl_matrix* X) {
	int rowNum = X->size1;
	int colNum = X->size2;
	ifstream input(inputFile.c_str());
	int i, j;
	double val;
	while (input >> i >> j >> val) {
		//val = log2(val+1);
		gsl_matrix_set(X, i, j, val);
		gsl_matrix_set(X, j, i, val);
	}
	input.close();
	return 0;
}


int io::read_dense_matrix(const string inputFile, gsl_matrix* X) {
	int rowNum = X->size1;
	int colNum = X->size2;
	ifstream input(inputFile.c_str());
	char* buff=NULL;
	string buffstr;
	int bufflen=0;
	int i, j;
	double val;
	int rowid=0;
	while(input.good())
	{
	//while (input >> i >> j >> val) {
		//val = log2(val+1);
		getline(input, buffstr);
		if(buffstr.length()>=bufflen)
		{
			if(buff!=NULL)
			{
				delete[] buff;
			}
			bufflen=buffstr.length()+1;
			buff=new char[bufflen];
		}
		strcpy(buff,buffstr.c_str());
		int colid=0;
		char* tok=strtok(buff,"\t");
		while(tok!=NULL)
		{
			if(colid>=0)
			{
				val=atof(tok);
				gsl_matrix_set(X, rowid, colid, val);
				
			}	
			tok=strtok(NULL,"\t");
			colid++;
		}
		rowid++;
	}
	input.close();

	// Free the memory after you're done
	if (buff != NULL) {
		delete[] buff;  // Free memory before function ends
	}

	return 0;
}


int io::write_list(const string outputFile, list<double>& err) {
	ofstream ofs;
	ofs.open(outputFile.c_str());
	for (list<double>::iterator itr = err.begin(); itr != err.end(); ++itr) {
		ofs << *itr << endl;
	}
	ofs.close();
	return 0;
}

int io::read_tree(const string inputFile,
	vector<int>& parentIds, vector<string>& aliases, vector<string>& fileNames, vector<int>& numSamples) {
	ifstream input(inputFile.c_str());
	int id, pid;
	string alias, filename, n;
	while (input >> id >> pid >> alias >> filename >> n) {
		parentIds.push_back(pid);
		aliases.push_back(alias);
		if (filename != "N/A") {
			fileNames.push_back(filename);
			stringstream nTemp(n);
			int numSample = 0;
			nTemp >> numSample;
			numSamples.push_back(numSample);
		}
	}
	input.close();
	return 0;
}


int io::read_k1_k2_list(const string inputFile, vector<int>& k1_vec, vector<int>& k2_vec){
	ifstream input(inputFile.c_str());
	int k1, k2;
	while(input >> k1 >> k2){
		k1_vec.push_back(k1);
		k2_vec.push_back(k2);
	}
	return 0;
}

int io::read_k1_k2_list(const string inputFile, vector<pair<int, int>>& k_vec){
	ifstream input(inputFile.c_str());
	int k1, k2;
	while(input >> k1 >> k2){
		pair<int, int> p(k1, k2);
		k_vec.push_back(p);
	}
	return 0;
}

int io::write_nmtf_output(gsl_matrix *U, gsl_matrix *V, gsl_matrix *S, gsl_matrix *R,list<double> *err, list<double> *slope, string& out_dir){
        io::write_list(out_dir + "err.txt", *err);
        io::write_list(out_dir + "slope.txt", *slope);
        io::write_dense_matrix(out_dir + "U.txt", U);
        io::write_dense_matrix(out_dir + "V.txt", V);
        io::write_dense_matrix(out_dir + "S.txt", S);
	//io::write_dense_matrix(out_dir + "R.txt", R);
        return 0;
}

int io::write_mem_and_time(const string out_file, unsigned long int time_diff, unsigned long int mem_diff){
	ofstream ofs;
	ofs.open(out_file.c_str());
	ofs << "time:\t" << time_diff << endl;
	ofs << "mem:\t" << mem_diff << endl;
	ofs.close();
	return 0;
}

int io::read_prev_results(const string in_dir, gsl_matrix *U, gsl_matrix *V, gsl_matrix *S){
	io::read_dense_matrix(in_dir + "U.txt", U);
	io::read_dense_matrix(in_dir + "V.txt", V);
	io::read_dense_matrix(in_dir + "S.txt", S);
	return 0;		
}
