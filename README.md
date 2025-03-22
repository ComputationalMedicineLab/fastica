fastica
=======

This is an implementation of the fixed-point FastICA algorithm presented by
Aapo Hyvärinen and Erkki Oja in [Independent Component Analysis: Algorithms and
Applications](https://www.cs.helsinki.fi/u/ahyvarin/papers/NN00new.pdf) with an
emphasis on performance, machine resource utilization, and portability of data
format.

`fastica` requires certain versions of MKL, Intel-OpenMP, Cython, and numpy,
which are as of this writing (March 2025) somewhat out of date.  There is a
torch backed implementation in development at [cml tools
v2](https://github.com/ComputationalMedicineLab/cml_tools)). Nevertheless, on
Intel machines, especially multi-node NUMA machines, the implementation here is
still 2x faster than the best torch or numpy backed pure-Python implementation
we have been able to develop. Since the package provides a CLI tool which reads
and writes numpy ndarray data files (e.g., `file.npy`), pre and post processing
may easily be done in any manner you choose. Our recommendation is to run
`fastica` as a standalone CLI tool from a dedicated Conda environment.


Installation
------------

This is a Cython project which requires a C++ build toolchain in a conda
environment. The easiest way to install is to make sure you have micromamba on
your path, and run:

```sh
./build_env.sh  # (or `source build_env.sh`)
make && make install
```

If running `$ ./build_env.sh` fails because bash cannot find micromamba, you
can try `source build_env.sh` instead. `build_env.sh` runs micromamba to
install from `environment.yaml`, and additionally both adds a few hooks with
useful environment variables and checks the installed versions.

`make` by itself will run the basic build and tests. If the build fails, you
might start by checking that you have the correct versions of `gcc` and `g++`.
`make install` will install the fastica library and command line tool into the
activated Python environment. Running the jupyter notebook `Visual Test` will
provide an additional check that the fastica algorithm implementation is
working correctly.


Usage Notes
-----------

`fastica` is both a library and a command line tool. After installation run
`fastica --help` to begin exploring the CLI. There are four subcommands:
`whiten`, `run`, `recover`, and `project`. `whiten` preprocesses a data matrix by
PCA projection and a whitening transform. `run` performs ICA analysis of a
whitened data matrix to produce a matrix `W`, from which `recover` will produce
the matrices `A` and `S` of the ICA formula `X = AS`. `project` will apply the
learned preprocessing steps (centering and whitening) and the ICA mixing matrix
to produce projections of new data through the ICA model. See also the example
script `example_fastica.sh`.
