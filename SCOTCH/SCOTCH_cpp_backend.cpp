//
// Created by Halberg, Spencer on 4/12/25.
//

#include "SCOTCH_cpp_backend.h"
typedef vector<pair<int, int>> k_vec_t;


gsl_rng* SCOTCH_cpp_backend::initialize_random_generator(int seed) {
	const gsl_rng_type* T;
	gsl_rng* ri;
	gsl_rng_env_setup();
	T = gsl_rng_default;
	ri = gsl_rng_alloc(T);
	gsl_rng_set(ri, seed);
	return ri;
}

int SCOTCH_cpp_backend::initialize_factors(int k1, int k2, int nSamples, int nFeatures) {
	U = gsl_matrix_calloc(k1, nSamples);
	V = gsl_matrix_calloc(k2, nFeatures);
	S = gsl_matrix_calloc(k1, k2);
	P = gsl_matrix_calloc(k2, nSamples);
	Q = gsl_matrix_calloc(k1, nFeatures);
	return 0;
}

//Frees all matrices except X and R
int SCOTCH_cpp_backend::free_factors()
{
	gsl_matrix_free(U);
	gsl_matrix_free(V);
	gsl_matrix_free(S);
	gsl_matrix_free(P);
	gsl_matrix_free(Q);
	return 0;
}

int SCOTCH_cpp_backend::free_factors_fixed_k1()
{
	gsl_matrix_free(V);
	gsl_matrix_free(S);
	gsl_matrix_free(P);
	gsl_matrix_free(Q);
	return 0;
}

int SCOTCH_cpp_backend::free_factors_fixed_k2()
{
	gsl_matrix_free(U);
	gsl_matrix_free(S);
	gsl_matrix_free(P);
	gsl_matrix_free(Q);
	return 0;
}

int SCOTCH_cpp_backend::free_matrices()
{
	free_factors();
	gsl_matrix_free(R);
	gsl_matrix_free(X);
	return 0;
}


string SCOTCH_cpp_backend::build_directory_path(const string& outputPrefix, int k1, int k2)
{
	stringstream out_dir_str;
	out_dir_str << outputPrefix << '/' << "k1_" << k1 << "_k2_" << k2 << "/";
	return out_dir_str.str();
}

int SCOTCH_cpp_backend::load_completed_run(const string& outputPrefix, int index, k_vec_t k_list, int &prev_k1, int &prev_k2)
{

	prev_k1 = k_list[index].first;
	prev_k2 = k_list[index].second;

	string in_dir = build_directory_path(outputPrefix, prev_k1, prev_k2);

	free_factors();

	U = gsl_matrix_calloc(prev_k1, n);
	V = gsl_matrix_calloc(prev_k2, m);
	S = gsl_matrix_calloc(prev_k1, prev_k2);
	P = gsl_matrix_calloc(prev_k2, n);
	Q = gsl_matrix_calloc(prev_k1, m);
	reset_k1_k2(prev_k1, prev_k2);
	io::read_prev_results(in_dir, U, V, S);
	return 0;
}


int SCOTCH_cpp_backend::processRun(int k1, int k2, const gsl_rng* rng) {


	//Set up run params
	reset_k1_k2(k1, k2);
	initialize_factors(k1, k2, n, m);

	//Initialize_matrices(Random only supported option)
	init::random(U, rng);
	init::random(V, rng);
	init::random(S, rng);


	NMTF::fit(X, U, V, S, P, Q, R);

	return 0;
}


int
SCOTCH_cpp_backend::increaseRun(const gsl_rng *rng, int new_k1, int new_k2)
{
	//Add handels so we don't accentently replace with uninitialized pointers
	gsl_matrix *U_new = U;
	gsl_matrix *V_new = V;
	gsl_matrix *S_new = S;
	gsl_matrix *P_new = P;
	gsl_matrix *Q_new = Q;

	if (new_k1 > u_components){
		U_new = gsl_matrix_calloc(new_k1, X->size1);
	}

	if (new_k2 > v_components)
	{
		V_new = gsl_matrix_calloc(new_k2, X->size2);
	}

	P_new = gsl_matrix_calloc(new_k2, X->size1);
	Q_new = gsl_matrix_calloc(new_k1, X->size2);
	S_new = gsl_matrix_calloc(new_k1, new_k2);

	if (new_k1 > u_components && new_k2 > v_components)
	{
		increase_k1_k2(new_k1, new_k2, X, U, V, S, P, Q, R, U_new, V_new, S_new, P_new, Q_new, rng);
		free_factors();
	}else if (new_k1 > u_components && new_k2 == v_components)
	{
		increase_k1_fixed_k2(new_k1, X, U, V, S, P, Q, R, U_new, S_new, P_new, Q_new, rng);
		free_factors_fixed_k2();
	}else if (new_k1 == u_components && new_k2 > v_components)
	{
		increase_k2_fixed_k1(new_k2, X, U, V, S, P, Q, R, V_new, S_new, P_new, Q_new, rng);
		free_factors_fixed_k1();
	}


	U = U_new;
	V = V_new;
	S = S_new;
	P = P_new;
	Q = Q_new;

}

int
SCOTCH_cpp_backend::decreaseRun(const gsl_rng *rng, int new_k1, int new_k2, k_vec_t k_list, int curr_i)
{
	//If the next element on the k1 and k2 is decreasing with respect to the previous,
	//an old matrix is loaded that has smaller k1 or K2 or we run from scratch.
	//free_factors();
	int prev_k1;
	int prev_k2;
	string in_path;

	int j;
	//Find the previous element in the k1 and k2 list that is smaller than the current element.
	for (j = curr_i-1; j >= 0; j--) // Go through index if J backwards
	{
		int test_k1 = k_list[j].first;
		int test_k2 = k_list[j].second;
		if (test_k1 <= new_k1 && test_k2 <= new_k2)
		{
			prev_k1 = test_k1;
			prev_k2 = test_k2;
			break;
		}
	}

	if (j == -1)
	{
		processRun(new_k1, new_k2, rng);
	}else
	{
		//in_path = build_directory_path(outputPrefix, prev_k1, prev_k2);
		load_completed_run(in_path, j, k_list, prev_k1, prev_k2);
		increaseRun(rng, new_k1, new_k2);
	}
	return 0;
}


int SCOTCH_cpp_backend::fit()
{
	//*************** SET ALL DEFAULTS *********************************
	//initialize factor and matrix size.
	string matrixFile;
	int k1 = -1, k2 = -1;

	//initialize file paths
	string outputPrefix = "./", k_file, inPrefix;

	//initialize_fit options
	bool multK=false, legacy = false, verbose = false;


	// Initialize converge tracking elements
	list<double> *err= new list<double>;
	list<double> *slope = new list<double>;

	//Usage file
	string usage = string("nmtf_usage.txt");


	// ******************************** PARSE ARGS ******************************************
	// Example validation
	if (X == nullptr) {
		std::cerr << "Error: Missing matrix file. Add data to SCOTCH\n";
		io::print_usage(usage);
		return 1;  // Exit with an error
	}

	if (k1<=0 || k2<=0)
	{
		if (k_file.empty()) {
			std::cerr << "Error: Missing factor. Please run \n";
			io::print_usage(usage);
			return 1;  // Exit with an error
		}
	}

	if (n<=0|| m<=0) {
		std::cerr << "Error: Missing matrix sizes. Please provide --n_samples or --n_features.\n";
		io::print_usage(usage);
		return 1;  // Exit with an error
	}

	//Initialize Random generator:
	// Extracted initialization logic for RNG
	gsl_rng* rng = initialize_random_generator(random_state);
	processRun(k1, k2, rng);
	free_matrices();

	return 0;
}

int
SCOTCH_cpp_backend::add_data_to_scotch(pybind11::array_t<double> x)
{
	int n_rows = x.shape(0);
	int n_cols = x.shape(1);
	X = numpy_to_gsl_matrix(x);
	R = gsl_matrix_calloc(n_rows, n_cols);
	set_size(n_rows, n_cols);
	return 0;
}

gsl_matrix*
SCOTCH_cpp_backend::numpy_to_gsl_matrix(pybind11::array_t<double> numpy_array)
{
	pybind11::buffer_info buf = numpy_array.request();
	if (buf.ndim != 2)
	{
		throw std::invalid_argument("numpy_to_gsl_matrix: array must be 2-dimensional");
	}

	int rows = buf.shape[0];
	int cols = buf.shape[0];

	gsl_matrix* gsl_mat = gsl_matrix_calloc(rows, cols);

	double* numpy_data = static_cast<double *>(buf.ptr);

	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			gsl_matrix_set(gsl_mat, i, j, numpy_data[i * cols + j]);
		}
	}

	return gsl_mat;
}


pybind11::array_t<double>
SCOTCH_cpp_backend::gsl_matrix_to_numpy(gsl_matrix* gsl_mat) {
	// Validate that the input pointer is not null
	if (!gsl_mat) {
		throw std::invalid_argument("gsl_matrix* is null.");
	}

	// Get the dimensions of the gsl_matrix
	size_t rows = gsl_mat->size1;
	size_t cols = gsl_mat->size2;

	// Create a NumPy array of the same shape
	auto numpy_array = pybind11::array_t<double>({rows, cols});

	// Access the buffer of the NumPy array
	pybind11::buffer_info buf = numpy_array.request();
	double* numpy_data = static_cast<double*>(buf.ptr);

	// Fill the NumPy array with data from the gsl_matrix
	for (size_t i = 0; i < rows; i++) {
		for (size_t j = 0; j < cols; j++) {
			numpy_data[i * cols + j] = gsl_matrix_get(gsl_mat, i, j);
		}
	}

	return numpy_array;
}
