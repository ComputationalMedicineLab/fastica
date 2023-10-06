fastica
=======

This is an implementation of the fixed-point FastICA algorithm presented by
Aapo Hyvärinen and Erkki Oja in [Independent Component Analysis: Algorithms and
Applications](https://www.cs.helsinki.fi/u/ahyvarin/papers/NN00new.pdf) with an
emphasis on performance, machine resource utilization, and portability of data
format.


Installation
------------

This is a Cython project which requires a C++ build toolchain. The easiest way
to install is to use Conda (or preferably Mamba) to install a dedicated
environment:

```sh
micromamba create --name=fastica python=3.11
micromamba activate fastica
micromamba install 'numpy<1.26'
micromamba install 'Cython<3.0'
micromamba install scipy scikit-learn matplotlib jupyter ipython
micromamba install mkl mkl-include mkl-service gxx gcc
make && make install
```

`make` by itself will run the basic build and tests. If the build fails, you
might start by checking that you have the correct versions of `gcc` and `g++`.
`make install` will install the fastica library and command line tool into the
activated Python environment. Running the jupyter notebook `Visual Test` will
provide an additional check that the fastica algorithm implementation is
working correctly.


Usage Notes
-----------

`fastica` is both a library and a command line tool.
**TODO** - elaborate, provide documentation and usage guide.
