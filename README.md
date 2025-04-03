# SCOTCH in C++

## Documentation
This is the SCOTCH/NMTF implementation in C++


## Description 
A C++ implementation of the regularized non-negative matrix tri-factorization algorithm. 

## Methods/Outputs
The goal of this function is to take a matrix X of dimension n x m and factor it into three matrices, U in n x k1, S in k1 x k2, and V in k2 x m, such that the difference between X and the product U x S x V is minimized. Formally this is resolved by minimizing the objective || X - U S V ||2^2. The program outputs the three product matrices. 

### Orthoganilty regularization
The model allows for the addition of orthoganility regularization on the U and/or the V matrix. The idea of this regularization is to generate unique factors. It is particularly useful in resolving unique clusters from the lower dimensional embedding. It is recommend to put orthoganility reg. on either or both factors such that they are unique. If both are selected, then the S matrix will have more overlap. 

### Sparsity regularization. 
This works simililarly to the ortho regularization. In this case, the number of non-zero entries in each factor in penalized. It again will result in more unique factors although. 

## Arguments 

| Argument | is Required? | Description | additional Info                               |
| ---------|--------------|-------------|-----------------------------------------------|
| --in_file: | required | The file containing the tab delimited X matrix if full form. | example: test/A.txt                           |
| --k1:		   | required |	The lower dimension of the U matrix.                         | example 2                                     | 
| --k2:		   | required | The lower dimension of the V matrix.                         | example 3                                     |
| --lU:		   | optional	|	strength of ortho reg on U.                                  | example 0.1 (value should be between [0, 1])  |
| --lV:		   | optional	|	strength of ortho reg on V.                                  | example 0.1 (values should be between [0, 1]) |
| --aU:		   | optional	|	strength of sparsity reg on U. | example 0.1 (values should be between [0, 1]) |
| --aV:		   | optional	|	strength of sparsity reg on V. | example 0.1 (values should be between [0, 1]) |
| --verbose	 | optional |		if included, time and convergence info will be printed to terminal |
| --seed		 | optional	|	random seed used to initialize U, S, and V. | Default 1010                                  |
| --max_iter | optional	|	number of epochs to attempt prior to output.| Default 100                                   |
| --term_tol | optional	|	the relative change in error prior to completion. | Default 1e-5                                  |	
| --out_dir	 | optional	|	output directory to print U, S, and V.  |
| --cpu		   | optional	|	if included, defuaults to using the CPU | 

## Running the example 
**./run_example.sh 1**.  Defaults to the CPU 
**./run_example.sh 0**.  Runs on GPU if available. 
