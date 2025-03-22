#!/bin/bash
micromamba create -f ./environment.yaml
# activate sets CONDA_PREFIX correctly
micromamba activate fastica

# Make sure that we have MKL BLAS pinned for later installs
echo 'libblas=*=*mkl' >> "$CONDA_PREFIX/conda-meta/pinned"

# Automatically set some environment variables on activate
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d/"
mkdir -p "$CONDA_PREFIX/etc/conda/deactivate.d/"

cat >"$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh" << EOF
#!/bin/bash
export KMP_AFFINITY='granularity=fine,compact,1,0'
export KMP_BLOCKTIME=1
export LD_PRELOAD="$CONDA_PREFIX/lib/libiomp5.so"
EOF

cat >"$CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh" << EOF
unset LD_PRELOAD
EOF

# Run the above env var scripts
source "$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh"

# Post-install version checks
python --version
pip --version
gcc --version
c++ --version
jupyter --version
ipython --version

MKL_VERBOSE=1 python << EOF
import cython, numpy, scipy, sklearn
sep = ("-"*80)+"\n"
print(f'{sep}Numpy version: {numpy.__version__}')
print(numpy.show_config())
print(f'{sep}Cython version: {cython.__version__}')
print(f'{sep}scipy version: {scipy.__version__}')
print(f'{sep}sklearn version: {sklearn.__version__}')
print(sep)
EOF
