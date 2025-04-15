//
// Created by Halberg, Spencer on 4/12/25.
//

#ifndef SCOTCH_CPP_BACKEND_H
#define SCOTCH_CPP_BACKEND_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "modules/nmtf.h"

namespace py = pybind11;



class  SCOTCH_cpp_backend: public NMTF{
	public:
	~SCOTCH_cpp_backend()
	{
		free_matrices();
	}
	gsl_matrix* get_U() { return U; }
	gsl_matrix* get_S() { return S; }
	gsl_matrix* get_V() { return V; }
	gsl_matrix* get_P() { return P; }
	gsl_matrix* get_Q() { return Q; }
	gsl_matrix* get_X() { return X; }




	using NMTF::set_NMTF_params;
	gsl_rng *
	initialize_random_generator(int seed);

	int
	initialize_factors(int k1, int k2, int nSamples, int nFeatures);

	int
	free_factors();

	int
	free_factors_fixed_k1();

	int
	free_factors_fixed_k2();

	int
	free_matrices();

	string
	build_directory_path(const std::string &outputPrefix, int k1, int k2);

	int
	load_completed_run(const std::string &outputPrefix, int index, vector<pair<int, int>> k_list, int &prev_k1,
	                   int &prev_k2);

	int
	processRun(int k1, int k2, const gsl_rng *rng);

	int
	increaseRun(const gsl_rng *rng, int new_k1, int new_k2);


	int
	decreaseRun(const gsl_rng *rng, int new_k1, int new_k2,
	            vector<pair<int, int>> k_list,
	            int curr_i);

	int
	run_multiple_NMTF_runs(const std::string &k_file, const std::string &outputPrefix, const gsl_rng *rng);

	int fit();

	pybind11::array_t<double> gsl_matrix_to_numpy(gsl_matrix *);
	int add_data_to_scotch(pybind11::array_t<double> X);
	gsl_matrix* numpy_to_gsl_matrix (pybind11::array_t<double> numpy_array);
};

PYBIND11_MODULE(SCOTCH_cpp_backend, m) {
	m.doc() = "Python bindings for the C++ NMTF class";
	// Bind the NMTF class
	py::class_<SCOTCH_cpp_backend>(m, "SCOTCH_cpp_backend")
	.def(py::init([](int k1, int k2, int maxIter, int seed,
	double lU = 0.0, double lV = 0.0, double aU = 0.0, double aV = 0.0){
	 // Create a new SCOTCH_cpp_backend instance
	 auto instance = new SCOTCH_cpp_backend();

	 // Default values for additional parameters of set_NMTF_params
	 init_method defaultInitMethod = random_init; // Example initialization
	 bool verb = false; // Default verbosity
	 double termTol = 1e-5; // Example termination tolerance
	 std::list<double>* err = nullptr; // No error list provided
	 std::list<double>* slope = nullptr; // No slope list provided

	 // Call set_NMTF_params to initialize NMTF parameters
	 int status = instance->set_NMTF_params(k1, k2, defaultInitMethod, maxIter, seed, verb, termTol,
						err, slope, aU, lU, aV, lV);

	 if (status != 0) {
	     throw std::runtime_error("Failed to set NMTF parameters");
	 }

	 return instance; // Return the initialized instance
	}))

	.def("gsl_matrix_to_numpy", &SCOTCH_cpp_backend::gsl_matrix_to_numpy, "Converts a gsl_matrix to a numpy array")
	.def("numpy_to_gsl_matrix", &SCOTCH_cpp_backend::numpy_to_gsl_matrix, "Converts a numpy array to a gsl_matrix")
	.def("add_data_to_scotch", &SCOTCH_cpp_backend::add_data_to_scotch,  "Adds data to the scotch matrix")
	.def("fit", &SCOTCH_cpp_backend::fit, "Performs the NMTF factorization on a matrix")
	.def("get_U", &SCOTCH_cpp_backend::get_U, "Gets pointer to U matrix")
	.def("get_V", &SCOTCH_cpp_backend::get_V, "Gets pointer to V matrix")
	.def("get_S", &SCOTCH_cpp_backend::get_S, "Gets pointer to S matrix")
	.def("get_P", &SCOTCH_cpp_backend::get_P, "Gets pointer to P matrix")
	.def("get_Q", &SCOTCH_cpp_backend::get_Q, "Gets pointer to Q matrix")
	.def("get_X", &SCOTCH_cpp_backend::get_X, "Gets pointer to X matrix");
}




#endif //SCOTCH_CPP_BACKEND_H
