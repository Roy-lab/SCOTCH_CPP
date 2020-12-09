### Tree-structured, Graph-regularized Integrated Factorization (TGIF) in C++

### [Step 1] Install 

Installation instructions below were tested in Linux Centos 7 distribution. [GSL (GNU Scientific Library) 2.6](https://www.gnu.org/software/gsl/doc/html/index.html) is used to handle matrix- and vector-related operations. For matrix inversion, one of the newer functions in GSL 2.6 is used, so the code may not run if you have an older GSL.

1. __If you already have GSL 2.6 installed__, edit the first few lines of the Makefile to point to the correct include and shared library directory, then jump to step 3.
```
#CHANGE PATHS AS NEEDED:
INCLUDE_PATH = ${CONDA_PREFIX}/include
LIBRARY_PATH = ${CONDA_PREFIX}/lib
```
2. __If you do not have GSL 2.6 installed, or you are not sure__, the easiest way to get it installed is to use [conda](https://anaconda.org/conda-forge/gsl/):
```
conda install -c conda-forge gsl
```
3. Make sure to add the location of the installed shared library to where the compiler/linker will be looking. If you used conda to install GSL to the default location in step 2, run the following command (or add the appropriate path if you already have GSL installed):
```
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CONDA_PREFIX}/lib
```
4. And let's install! In the same directory you downloaded the code/Makefile (either by cloning the repository or by downloading a release), run:
```
make
```
5. If all went well, you won't get any alarming messages, and you will see an executable named `grinch` created in the same directory. A quick test below will print the manual for grinch:
```
./grinch -h
```

Note: in order to implement NNDSVD initialization of factors, a fast randomized SVD algorithm from [RSVDPACK](https://github.com/sergeyvoronin/LowRankMatrixDecompositionCodes) was used. A minor modification to allow random seed specification was made to the original code from [RSVDPACK](https://github.com/sergeyvoronin/LowRankMatrixDecompositionCodes/tree/master/single_core_gsl_code). This updated code is included under modules/random_svd directory. Compilation of this code is part of the included Makefile; no additional step is necessary for installation.

### [Step 2] Run

#### Basic usage
```
./run_tgif input/rhie_2019_tree.txt 196 10 -o output/
```
- `input/rhie_2019_tree.txt` specifies the tree file, which contains file locations to individual task matrices (paths are relative to location of run_tgif executable location). 
- `196` is the number of features/columns in each task matrix, which has to be be the same across all tasks. This current version assume a symmetric matrix.
- `10` = k, the smaller dimensions of U and V. 
-	`-o output/` will put all output files to output/ directory. Check out the example output directory in the repo.
-	`-a 10` will set the alpha (strength of regularization to parent node) to be 10. Default is alpha = 1.


#### Input tree file format
See example in [input/rhie_2019_tree.txt](https://github.com/Roy-lab/tgif-c/blob/master/input/rhie_2019_tree.txt)
```
1 5 RWPE1 input/rhie_2019/RWPE1.txt 196
2 4 C42B input/rhie_2019/C42B.txt 196
3 4 22Rv1 input/rhie_2019/22Rv1.txt 196
4 5 cancer N/A N/A
5 -1 root N/A N/A
```
- Column 1: **node ID**; start from 1 and move up.
- Column 2: **parent node ID**; right now the implementation will only work correctly if you ID all children nodes before a parent node (so start from the lowest depth of tree, move to next level, till you hit the root, which should be the last node ID.)
- Column 3: **node alias**, used as prefix to U and V output files.
- Column 4: **location of input matrix file for leaf nodes**, relative to where the run_tgif executable is. Set as N/A for non-leaf nodes.
- Column 5: **number of samples/data points/rows in each input matrix**; should be the same as the number of features/columns for now. Set as N/A for non-leaf nodes.

#### Input matrix file format
- Matrix file, 0-indexed, tab-delimited, sparse-matrix format, no header
- See examples in [input/rhie_2019/22Rv1.txt](https://github.com/Roy-lab/tgif-c/blob/master/input/rhie_2019/22Rv1.txt)
```
0	0	1000.2
0	1	1201.78
10	1	200.7
...
```

### TODO
Need to handle asymmetric case: the easiest way right now is to just keep V the same across all tasks (this is fine since the number of features are the same), then just randomly initialize U. This should work well since the code already updates U before V. Code that needs to be updated or can be reused in this case:

- [ ] Replace this line in TGIF::make_tree where U is the copied from the first leaf node: https://github.com/Roy-lab/tgif-c/blob/81936118ecddd6066a673978f68a28bce8e039d9/modules/tgif.cpp#L71 
- [ ]	Use init::random, which currently fills in two factors with random values, to create a function that only fills in one factor: https://github.com/Roy-lab/tgif-c/blob/81936118ecddd6066a673978f68a28bce8e039d9/modules/initialization.cpp#L124 
- [ ] Update io::read_sparse_matrix to handle asymmetric matrices: https://github.com/Roy-lab/tgif-c/blob/81936118ecddd6066a673978f68a28bce8e039d9/modules/io.cpp#L40

Note: all .cpp file has a corresponding .h file.
 
FYI, We discussed three ways for intialization:
- do NMF individually, then align with hungarian algorithm -> lots of NMFs, also I don't have a hungarian algo implemented/tested
- joint nmf -> NMF on a potentially very large matrix.
- nmf on first one, use it across all tasks -> doesn't work in asymmetric case mentioned
