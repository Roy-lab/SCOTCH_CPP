#include <iostream>
#include <fstream>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_vector.h>
#include <list>
#include <vector>
#include <math.h>
#include <string>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <unistd.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <sys/resource.h>
#include "modules/io.h"
#include "modules/utils.h"
#include "modules/initialization.h"
#include "modules/nmtf.h"

int main(int argc, char **argv)
{
	struct timeval beginTime;
	gettimeofday(&beginTime,NULL);

	struct rusage bUsage;
	getrusage(RUSAGE_SELF,&bUsage);

	const char* matrixFile;
	int nSamples = -1;
	int nFeatures = -1;
	int k1 = -1;
	int k2 = -1;
	
	string outputPrefix = string("");
	string k_file=string("");
	bool multK=false;
	int seed = 1010;
	int verbose = true;
	int maxIter = 300;
	double tol = 1e-5;
	double alphaU = 0;
	double lambdaU = 0;
	double alphaV = 0;
	double lambdaV = 0;
	list<double> *err= new list<double>;
	list<double> *slope = new list<double>;
	string usage = string("usage_nmtf.txt");

	int c;
	while((c = getopt(argc, argv, "o:r:s:m:t:a:l:b:k:h:i:")) != -1)
		switch (c) {
			case 'o':
				outputPrefix = string(optarg);
				break;
			case 'r':
				seed = atoi(optarg);
				break;
			case 's':
				verbose = false;
				break;
			case 'm':
				maxIter = atoi(optarg);
				break;
			case 't':
				tol = atof(optarg);
				break;
			case 'a':
				alphaV = atof(optarg);
				break;
			case 'l':
				lambdaV = atof(optarg);
				break;
			case 'b':
				alphaU = atof(optarg);
				break;
			case 'k':
				lambdaU = atof(optarg);
				break;
			case 'i':
				k_file = string(optarg);
				multK=true;
				break;			
			case 'h':
				io::print_usage(usage);
				return 0;
			case '?':
				io::print_usage(usage);
				return 0;
			default:
				io::print_usage(usage);
				return 0;
		}	

	if ((argc - optind) < 5) {
		io::print_usage(usage);
		return 1;
	} else {
		matrixFile = argv[optind];
		nSamples = atoi(argv[optind+1]);
		nFeatures = atoi(argv[optind+2]);
		k1 = atoi(argv[optind+3]);
		k2 = atoi(argv[optind+4]);
	}
	
	//Initialize Random Vector 
	const gsl_rng_type* T;
        gsl_rng* ri;
        gsl_rng_env_setup();
        T = gsl_rng_default;
        ri = gsl_rng_alloc(T);
        gsl_rng_set(ri, seed);
	
	//Read Initial Matrix	
	string matrixFileName(matrixFile);
	gsl_matrix* X = gsl_matrix_calloc(nSamples, nFeatures);
	gsl_matrix* R = gsl_matrix_calloc(nSamples, nFeatures);
	io::read_dense_matrix(matrixFileName, X);
	
	//U - U Factors
	//P = U * S
	//V = V Factors
	//Q = S * V^T
	//S = S co-factor matrix 
	gsl_matrix* U = gsl_matrix_calloc(k1, nSamples);
	gsl_matrix* P = gsl_matrix_calloc(k2, nSamples); 
	gsl_matrix* V = gsl_matrix_calloc(k2, nFeatures);
        gsl_matrix* Q = gsl_matrix_calloc(k1, nFeatures);
	gsl_matrix* S = gsl_matrix_calloc(k1, k2);
	
	//Initialize timers 
	struct timeval factorTime;
	struct timeval endTime;
	struct rusage eUsage;	
	unsigned long int bt;
	unsigned long int ft;
	unsigned long int et;
	unsigned long int bu;
	unsigned long int eu;
			
	//if multiple K, See if any outputs exist already
	if(multK){
		//Load in K_list, note we initialize list with the first k1 and k2. 
		vector<int> k1_vec, k2_vec;
                k1_vec.push_back(k1);
                k2_vec.push_back(k2);
                io::read_k1_k2_list(k_file ,k1_vec, k2_vec);
		int completed = -99;
		int prev_k1;
		int prev_k2;
		int new_k1;
		int new_k2;

	
		string out_dir; 
		ifstream Ufile;
		ifstream Vfile;
		ifstream Sfile;
		NMTF nmtf = NMTF(k1, k2, random_init,  maxIter, seed, verbose, tol, err, slope, alphaU, lambdaU, alphaV, lambdaV);
		
		//See which are completed. 
		for(int i = 0; i < k1_vec.size(); i++){
			stringstream out_dir_str;
			out_dir_str << outputPrefix << "/k1_" << k1_vec[i] << "_k2_" << k2_vec[i] << "/";
			out_dir=out_dir_str.str();	
			Ufile.open((out_dir + "U.txt").c_str());
			Vfile.open((out_dir + "V.txt").c_str());
			Sfile.open((out_dir + "S.txt").c_str());
			if(Ufile && Vfile && Sfile){
				completed = i;
			}
			Ufile.close();
			Vfile.close();
			Sfile.close();					
		}
		//If completed version exists, load in U V and S
		if(completed >=  0){
			stringstream in_dir_str;
			string in_dir;

			prev_k1 = k1_vec[completed];
			prev_k2 = k2_vec[completed];
			
			in_dir_str << outputPrefix << "/k1_" << k1_vec[completed] << "_k2_" << k2_vec[completed] << "/";
			in_dir = in_dir_str.str();
				
			gsl_matrix_free(U);
                        gsl_matrix_free(V);
                        gsl_matrix_free(S);	
			gsl_matrix_free(P);
			gsl_matrix_free(Q);
		
			gsl_matrix* U_old = gsl_matrix_calloc(prev_k1, nSamples);
                        gsl_matrix* V_old = gsl_matrix_calloc(prev_k2, nFeatures);
                       	gsl_matrix* S_old = gsl_matrix_calloc(prev_k1, prev_k2);
                        gsl_matrix* P_old = gsl_matrix_calloc(prev_k2, nSamples);
			gsl_matrix* Q_old = gsl_matrix_calloc(prev_k1, nFeatures);
			nmtf.reset_k1_k2(prev_k1, prev_k2);
                        io::read_prev_results(in_dir, U_old, V_old, S_old);
                        U = U_old;
                        V = V_old;
                        S = S_old;
			P = P_old;
			Q = Q_old;
		}else{
			//Do dry start 
        		//randomly initialize U V S
			init::random(U, ri);
        		init::random(V, ri);
        		init::random(S, ri);

        		gettimeofday(&factorTime, NULL);
			
			//Fit data 
        		nmtf.fit(X, U, V, S, P, Q, R);
        		mkdir(outputPrefix.c_str(), 0766);
        		stringstream out_dir_str;
        		out_dir_str << outputPrefix << '/' << "k1_" << k1 << "_k2_" << k2 << "/";
        		string out_dir=out_dir_str.str();
        		mkdir(out_dir.c_str(), 0766);

        		gettimeofday(&endTime,NULL);

        		getrusage(RUSAGE_SELF,&eUsage);

        		bt = beginTime.tv_sec;
       			ft = factorTime.tv_sec;
        		et = endTime.tv_sec;

       			cout << "Total time elapsed: " << et - bt << " seconds" << endl;

        		bu = bUsage.ru_maxrss;
        		unsigned long int eu = eUsage.ru_maxrss;

        		cout << "Memory usage: " << (eu - bu)/1000 << "MB" << endl;
			
			//write data 
        		io::write_mem_and_time(out_dir + "usage.txt", et - ft, (eu - bu)/1000);
        		io::write_nmtf_output(U, V, S, R, err, slope, out_dir);
			string tar_dir = "tar czf " + outputPrefix + ".tgz " + outputPrefix;
                        system(tar_dir.c_str());
                        sleep(10);


			prev_k1 = k1;
			prev_k2 = k2;
			completed = 0;
		}
	        
		// Continue MultK after initialization
                int i = completed + 1;
		string old_dir;
		//Set success equal to true to enter loops. 
		//success is used to make sure that any gene list can be used as long as it is the orginal k1 and k2 are the smallest factors learned 
                bool success= true;
                while(i < k1_vec.size()){
                        if(success){
                                gettimeofday(&factorTime, NULL);
                                new_k1=k1_vec[i];
                                new_k2=k2_vec[i];
                                err->clear();
                                slope->clear();
                        }
			//If we increment k1 and k2
                        if( new_k1 > prev_k1 && new_k2 > prev_k2){
				//Initialize new size matrices
                                gsl_matrix* U_new = gsl_matrix_calloc(new_k1, nSamples);
				gsl_matrix* P_new = gsl_matrix_calloc(new_k2, nSamples);
                                gsl_matrix* V_new = gsl_matrix_calloc(new_k2, nFeatures);
                                gsl_matrix* Q_new = gsl_matrix_calloc(new_k1, nFeatures);
				gsl_matrix* S_new = gsl_matrix_calloc(new_k1, new_k2);
                                nmtf.increase_k1_k2(new_k1, new_k2, X, U, V, S, P, Q, R, U_new, V_new, S_new, P_new, Q_new, ri);
                                gsl_matrix_free(U);
                                gsl_matrix_free(V);
                                gsl_matrix_free(S);
				gsl_matrix_free(P);
				gsl_matrix_free(Q);
                                U = U_new;
                                V = V_new;
                                S = S_new;
				P = P_new;
				Q = Q_new;
                                success = true;
                        //If we increase k1 and leave k2 the same 
                        //When this happens we fit with the old V matrix and new U matrix
			}else if (new_k1 > prev_k1 && new_k2 == prev_k2){
                                gsl_matrix* U_new = gsl_matrix_calloc(new_k1, nSamples);
                                gsl_matrix* P_new = gsl_matrix_calloc(new_k2, nSamples);
                                gsl_matrix* Q_new = gsl_matrix_calloc(new_k1, nFeatures);
                                gsl_matrix* S_new = gsl_matrix_calloc(new_k1, new_k2);
				nmtf.increase_k1_fixed_k2(new_k1, X, U, V, S, P, Q, R, U_new, S_new, P_new, Q_new, ri);
                                gsl_matrix_free(U);
                                gsl_matrix_free(S);
                                gsl_matrix_free(P);
				gsl_matrix_free(Q);
				U = U_new;
                                S = S_new;
				P = P_new;
				Q = Q_new;
                                success = true;
                        //If we increase k2 and leave k1 the same
                        //When this happens we fit with the old U and new V matrix
			}else if (new_k1 == prev_k1 && new_k2 > prev_k2){
                                gsl_matrix* P_new = gsl_matrix_calloc(new_k2, nSamples);
                                gsl_matrix* V_new = gsl_matrix_calloc(new_k2, nFeatures);
                                gsl_matrix* Q_new = gsl_matrix_calloc(new_k1, nFeatures);
                                gsl_matrix* S_new = gsl_matrix_calloc(new_k1, new_k2);
				nmtf.increase_k2_fixed_k1(new_k2, X, U, V, S, P, Q, R, V_new, S_new, P_new, Q_new, ri);
                                gsl_matrix_free(V);
                                gsl_matrix_free(S);
                                gsl_matrix_free(P);
				gsl_matrix_free(Q);
				V = V_new;
                                S = S_new;
				P = P_new;
				Q = Q_new;
                                success = true;
                        }else{
			//If the next element on the k1 and k2 is decreasing with respect to the previous, 
			//an old matrix is loaded that has smaller k1 or K2
                                gsl_matrix_free(U);
                                gsl_matrix_free(V);
                                gsl_matrix_free(S);
				gsl_matrix_free(P);
                                gsl_matrix_free(Q);

				int j=i-1;
                                while(k1_vec[j] > new_k1 ||  k2_vec[j] > new_k2){
                                        j--;
                                }
                                prev_k1=k1_vec[j];
                                prev_k2=k2_vec[j];
                                stringstream old_dir_str;
                                old_dir_str << outputPrefix << "/k1_" << prev_k1 << "_k2_" << prev_k2 << "/";
                                old_dir=old_dir_str.str();
                                gsl_matrix* U_old = gsl_matrix_calloc(prev_k1, nSamples);
                                gsl_matrix* V_old = gsl_matrix_calloc(prev_k2, nFeatures);
                                gsl_matrix* S_old = gsl_matrix_calloc(prev_k1, prev_k2);
                                gsl_matrix* P = gsl_matrix_calloc(prev_k2, nSamples);
				gsl_matrix* Q = gsl_matrix_calloc(prev_k1, nFeatures);
				nmtf.reset_k1_k2(prev_k1, prev_k2);
                                io::read_prev_results(old_dir, U_old, V_old, S_old);
                                U = U_old;
                                V = V_old;
                                S = S_old;
                                //setting success to false prevents selecting new k1 and k2 at top of loop and stops writing. 
				success = false;
                        }
                        if(success){
                                //Write the matrix
				gettimeofday(&endTime, NULL);
                                getrusage(RUSAGE_SELF,&eUsage);
                                ft = factorTime.tv_sec;
                                et = endTime.tv_sec;
                                eu = eUsage.ru_maxrss;
                                stringstream out_dir_str;
				out_dir_str << outputPrefix << "/k1_" << new_k1 << "_k2_" << new_k2 << "/";
                                out_dir=out_dir_str.str();
                                mkdir(out_dir.c_str(), 0766);
				io::write_mem_and_time(out_dir + "usage.txt", (et - ft), (eu - bu)/1000);
                                io::write_nmtf_output(U, V, S, R, err, slope, out_dir);
                                string tar_dir = "tar czf " + outputPrefix + ".tgz " + outputPrefix;
				system(tar_dir.c_str());
				sleep(10);
				prev_k1=new_k1;
                                prev_k2=new_k2;
                                i++;
                        }
		}
	}else{	
		// This is for single use. We fit the matrix.
		init::random(U, ri);
		init::random(V, ri);
		init::random(S, ri);

		NMTF nmtf = NMTF(k1, k2, random_init,  maxIter, seed, verbose, tol, err, slope, alphaU, lambdaU, alphaV, lambdaV);
		gettimeofday(&factorTime, NULL);
	
		nmtf.fit(X, U, V, S, P, Q, R);
		mkdir(outputPrefix.c_str(), 0766);
		stringstream out_dir_str;
		out_dir_str << outputPrefix << "/k1_" << k1 << "_k2_" << k2 << "/";
		string out_dir=out_dir_str.str();
		mkdir(out_dir.c_str(), 0766);
	
				
		gettimeofday(&endTime,NULL);

		getrusage(RUSAGE_SELF,&eUsage);

		unsigned long int bt = beginTime.tv_sec;
		unsigned long int ft = factorTime.tv_sec;
		unsigned long int et = endTime.tv_sec;

		cout << "Total time elapsed: " << et - bt << " seconds" << endl;
	
		unsigned long int bu = bUsage.ru_maxrss;
		unsigned long int eu = eUsage.ru_maxrss;
	
		cout << "Memory usage: " << (eu - bu)/1000 << "MB" << endl;
	
		io::write_mem_and_time(out_dir + "usage.txt", et - ft, (eu - bu)/1000);
		io::write_nmtf_output(U, V, S, R, err, slope, out_dir);
	}
	//Free all memory. 
	gsl_matrix_free(X);
	gsl_matrix_free(U);
	gsl_matrix_free(V);
	gsl_matrix_free(S);
	gsl_matrix_free(R);
	gsl_matrix_free(P);
	gsl_matrix_free(Q);
	gsl_rng_free(ri);
	delete err;
	delete slope;	
	return 0;
}

