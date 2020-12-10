### Tree-guided multi-task matrix factorization in C++

- [Installation instruction](#step-1-install)
- [Basic usage](#basic-usage)
- [Input file formats](#input-tree-file-format)
- [Parameters](#parameters)
- [Program outputs](#output-files)

### \[Step 1\] Install 

Installation instructions below were tested in Linux Centos 7 distribution. [GSL (GNU Scientific Library) 2.6](https://www.gnu.org/software/gsl/doc/html/index.html) is used to handle matrix- and vector-related operations. For matrix inversion, one of the newer functions in GSL 2.6 is used, so the code may not run if you have an older GSL.

1. __If you already have GSL 2.6 installed__, edit the first few lines of the Makefile to point to the correct include and shared library directory, then jump to step 3.
```
#CHANGE PATHS AS NEEDED:
INCLUDE_PATH = ${CONDA_PREFIX}/include
LIBRARY_PATH = ${CONDA_PREFIX}/lib
```
2. __If you do not have GSL 2.6 installed, or you are not sure__, one way to get it installed is to use [conda](https://anaconda.org/conda-forge/gsl/):
```
conda install -c conda-forge gsl
```
3. Make sure to add the location of the installed shared library to where the compiler/linker will be looking. If you used conda to install GSL to the default location in step 2, run the following command after activating the correct environment, or add the appropriate path if you already have GSL installed:
```
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CONDA_PREFIX}/lib
```
4. And let's install! In the same directory you downloaded the code/Makefile (either by cloning the repository or by downloading a release), run:
```
make run_tmi
```
5. If all went well, you won't get any alarming messages, and you will see an executable named `run_tmi` created in the same directory. A quick test below will print the manual (UNDER CONSTRUCTION):
```
./run_tmi -h
```

Note: in order to implement NNDSVD initialization of factors, a fast randomized SVD algorithm from [RSVDPACK](https://github.com/sergeyvoronin/LowRankMatrixDecompositionCodes) was used. A minor modification to allow random seed specification was made to the original code from [RSVDPACK](https://github.com/sergeyvoronin/LowRankMatrixDecompositionCodes/tree/master/single_core_gsl_code). This updated code is included under modules/random_svd directory. Compilation of this code is part of the included Makefile; no additional step is necessary for installation.

### \[Step 2\] Run

#### Basic usage
See [Parameters](#parameters) for more details.
```
./run_tmi input/toy_tree.txt 120 2 -o output/ -a 10 -l 200
```
- `input/toy_tree.txt` specifies the tree file, which contains file locations to individual task matrices (paths are relative to location of run_tmi executable location). 
- `120` is the number of features/columns in each task matrix, which has to be be the same across all tasks. 
- `2` = k, the smaller dimensions of U and V. 
-	[Optional] `-o output/` will put all output files to output/ directory. Check out the example output directory in the repo. By default output will be saved to current directory.
-	[Optional] `-a 10` will set the alpha (strength of regularization to parent node) to be 10. Default is alpha = 10.
- [Optional] `-l 200` will set lambda (strength of sparsity constraint) to be 200. By default there is no sparsity constraint, i.e., lambda = 0.

#### Input tree file format
See example in input/toy_tree.txt.
```
1 3 A input/toy/A.txt 95
2 3 B input/toy/B.txt 80
3 -1 root N/A N/A
```
- Column 1: **node ID**; start from 1 and move up.
- Column 2: **parent node ID**; right now the implementation will only work correctly if you ID all children nodes before a parent node (so start from the lowest depth of tree, move to next level, till you hit the root, which should be the last node ID.)
- Column 3: **node alias**, used as prefix to U and V output files.
- Column 4: **location of input matrix file for leaf nodes**, relative to where the `run_tmi` executable is. Set as N/A for non-leaf nodes.
- Column 5: **number of rows/data points in each input matrix**. Set as N/A for non-leaf nodes.

#### Input matrix format
- Tab-delimited "dense" matrix text files.
- Each column represents a feature (e.g. gene); each row represents a data point (e.g. cell).
- All input matrix files listed in the tree file should have the same number of features/columns (they can have different number of rows or data points).

#### Parameters


| Position | Parameter | Description | Default Value/Behavior | 
| :---    | :---        | :--- | :--- |
| 1     | input tree file      | The tree file, which contains file locations to individual task matrices (paths are relative to location of run_tmi executable location). See above for tree file format | N/A | 
| 2     | number of features   | Number of features/columns in each task matrix, which has to be be the same across all tasks. | N/A | 
| 3     | k     | The lower dimension of the factors (i.e. number of clusters) | N/A | 
| optional | -o \<output_file_prefix\>    | Ouput file path. Note: will NOT create a directory if the specified directory does not exist. | Output files will save to current directory. | 
| optional | -a \<alpha\>  | Strenth of tree regularization, higher value enfoces higher similarity to parent node. |  10 | 
| optional | -l \<lambda\>  | Strength of the sparsity constraint. Larger value results in sparser leaf node (i.e. task-specific) factor V. | 0, i.e. no sparsity regularization. |  
| optional | -s  | Run in slient mode, nothing printed to stdout. | Error, total run time, max memory usage printed to stdout. | 
| optional | -r \<random_state\> | Random state/seed used for rNNDSVD initialization. | 1010|
| optional | -t \<tol\> | Determines convergence and the termination of iterations. If \<tol\> = 10, the algorithm will keep iterating until the absolute difference between the previous iteration's error and current iteration's error is less than 10.| 1 |
| optional | -m \<max_iter\> | If \<max_iter\> = 200, the algorithm will terminate at 200 iterations if it has not coverged based on the tolerance (tol) parameter by then. | 300 |

#### Output files:
- `leaf`\_U.txt and `leaf`\_V.txt for each leaf node; `leaf` will bereplaced with the node's alias.
- `node`\_V.txt for each internal node and the root node; `node` will be replaced with the node's alias.

#### Difference from TGIF
- No graph regularization
- Initialization via joint NMF
- Sparsity regularization for leaf node (i.e. task-specific) factor V.

#### TODO
- [x] Update documentation
- [ ] Upload derivation for sparisty regularization on task-specific Vs
- [x] Test sparsity regularization
- [ ] Try to reduce matrix copying after initialization via joint NMF (somehow force matrix views to stick around in the heap??)
