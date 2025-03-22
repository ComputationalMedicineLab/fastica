fastica
=======

This is an implementation of the fixed-point FastICA algorithm presented by
Aapo Hyvärinen and Erkki Oja in [Independent Component Analysis: Algorithms and
Applications](https://www.cs.helsinki.fi/u/ahyvarin/papers/NN00new.pdf) with an
emphasis on performance, machine resource utilization, and portability of data
format.


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

`fastica` is both a library and a command line tool.
**TODO** - elaborate, provide documentation and usage guide.
