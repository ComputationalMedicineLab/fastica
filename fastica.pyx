# # cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
# # distutils: libraries = ['mkl_intel_ilp64', 'mkl_intel_thread', 'mkl_core', 'iomp5', 'pthread', 'm', 'dl']
# # distutils: extra_link_args = ['-fopenmp', '-Wl,--no-as-needed']
# # distutils: extra_compile_args = ['-m64', '-O3', '-fopenmp', '-Wno-unused']
# # distutils: define_macros=MKL_ILP64=1
"""fastica.pyx - a performant FastICA implementation
"""
DEF CACHE_ALIGN = 64
DEF PAGE_ALIGN = 4096

cimport cython
from cython cimport view
from libc.math cimport fabs, sqrt
from mkl_wrapper cimport *

import logging
import os
import shutil
import unittest
from pathlib import Path

import numpy as np


cdef double EPS = np.finfo(np.double).eps
cdef double TINY = np.finfo(np.double).tiny
cdef double ONE = 1.0

__version__ = '0.1'


###############################################################################
#                                                                             #
#   UTILITY AND GENERAL FUNCTIONS                                             #
#                                                                             #
###############################################################################
def format_nbytes(Py_ssize_t num):
    """Format bytes in human readable SI units"""
    cdef double n = <double>num
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(n) < 1024.0:
            return "%3.1f%sB" % (n, unit)
        n /= 1024.0
    return "%.1f%sB" % (n, 'Yi')


def estimate_base_memory(Py_ssize_t n, Py_ssize_t m, Py_ssize_t c):
    """Gives a low-ball estimate of how much memory in bytes will be needed to
    run the base FastICA function on a dataset projected down to `n` components
    and `m` samples, with `c` extra rows of working space in `apply_g`.

    Does **not** include buffers which may be allocated internally by MKL BLAS
    and LAPACK functions.

    Cf. fastica.fastica
    """
    cdef:
        Py_ssize_t s = sizeof(double)
        Py_ssize_t nbytes = 0
    nbytes += (n*m*s)*2     # X and gX
    nbytes += (n*s)         # gdv
    nbytes += (c*m*s)       # tmp_m
    nbytes += (n*n*s)*3     # W, W1, and tmp_W
    nbytes += (n*n*s)       # F
    nbytes += (n*s)         # D
    nbytes += (m*s)         # ones
    return nbytes


def estimate_whiten_memory(Py_ssize_t n, Py_ssize_t m, Py_ssize_t c):
    """Gives a minimum estimate of how much memory in bytes will be needed to
    run the whitening algorithm on a data matrix of doubles of shape [n, m]
    seeking to project down to c components.

    Does **not** include buffers which may be allocated internally by MKL BLAS
    and LAPACK functions.

    Cf. fastica.whiten
    """
    cdef:
        Py_ssize_t s = sizeof(double)
        Py_ssize_t nbytes = 0
    nbytes += (n*m*s)   # X
    nbytes += (c*m*s)   # X1
    nbytes += (n*s)     # X_mean
    nbytes += (c*c*s)   # K
    nbytes += (n*n*s)   # F
    nbytes += (n*s)     # D
    return nbytes


cdef void* malloc_or_raise(size_t nbytes, int align):
    # raise a MemoryError if mkl_malloc fails
    cdef void* ptr = mkl_malloc(nbytes, align)
    if not ptr:
        raise MemoryError(f'{format_nbytes(nbytes)} allocation failed')
    return ptr


cdef void safe_mkl_free(void *buff):
    # Check if buff is NULL before calling mkl_free
    if buff: mkl_free(buff)


cdef view.array aligned_doubles(tuple shape, int align):
    # Wraps an aligned buffer in a cython.view.array object with shape.
    # Caller is responsible for making sure shape is a tuple of nonzero
    # dimensions (in particular, cython memoryviews and slices are always
    # 8-elem lists with trailing zeros after the last "real" dimension).
    cdef:
        view.array vx = view.array(shape=shape,
                                   itemsize=sizeof(double),
                                   format='d',
                                   allocate_buffer=False)
        Py_ssize_t i = 0
        Py_ssize_t n = len(shape)
        Py_ssize_t d = 0
        size_t nbytes = sizeof(double)
    for i in range(n):
        d = shape[i]
        nbytes *= d
    vx.data = <char *>malloc_or_raise(nbytes, align)
    vx.callback_free_data = safe_mkl_free
    return vx


cdef void fast_copy_vector(double[::1] v,
                           double[::1] w
                           ) nogil:
    cdef Py_ssize_t n = v.shape[0]
    mkl_domatcopy(b'R', b'N', 1, n, 1.0, &v[0], n, &w[0], n)


cdef void fast_copy_matrix(double[:,::1] A,
                           double[:,::1] B,
                           bint transpose
                           ) nogil:
    cdef:
        Py_ssize_t n = A.shape[0]
        Py_ssize_t m = A.shape[1]
        Py_ssize_t ldb = n if transpose else m
        char trans = b'T' if transpose else b'N'
    mkl_domatcopy(b'R', trans, n, m, 1.0, &A[0,0], m, &B[0,0], ldb)


cdef view.array aligned_copy(arr,
                             int align = PAGE_ALIGN,
                             bint transpose = False):
    # Cython memoryview shapes are always 8 elements; the trailing elements
    # after the number of real dimensions are all zero. So first we need to
    # transpose as needed and get the real shape of the input.
    out_shape = arr.T.shape if transpose else arr.shape
    out_shape = tuple(x for x in out_shape if x != 0)
    out = aligned_doubles(out_shape, align)

    if arr.ndim == 1:
        fast_copy_vector(arr, out)
    elif arr.ndim == 2:
        fast_copy_matrix(arr, out, transpose)
    else:
        # This handy bit of syntax copies all data from arr to out across dims
        # (i.e., we don't need to know how many dimensions arr has) - *BUT* it
        # is *very, very slow* for large objects.
        out[...] = arr[...]
    return out


def align_ndarray(ndarr, align=PAGE_ALIGN, transpose=False):
    """
    Copies `ndarr` into an aligned buffer, optionally transposing it, and
    returns the buffer wrapped in a numpy.ndarray
    """
    return np.asarray(aligned_copy(ndarr, align, transpose))


# TODO: have dsyevr / dsyevd as option?
cdef inline int c_sym_eig(double* D,        # Eigenvalue output
                          double* F,        # Eigenvector output
                          double* A,        # Matrix input
                          Py_ssize_t n_row, # Num Rows of A
                          Py_ssize_t n_col, # Num Cols of A
                          int transpose     # 0 = AA^T; 1 = A^TA
                          ) nogil noexcept:
    # Computes the eigenvalues and eigenvectors of `A @ A.T` or `A.T @ A`
    cdef:
        CBLAS_TRANSPOSE tr = CblasTrans if transpose else CblasNoTrans
        lapack_int n = n_col if transpose else n_row
        lapack_int k = n_row if transpose else n_col
        lapack_int lda = n_col  # Leading (aka Memory-layout) dimension of A
        lapack_int ldc = n      # dimension of square output matrix
        lapack_int status = 0
    # `dsyrk` only puts its results into either the upper or lower triangular
    # part of the output, but `dsyevr/dsyevd` only *need* the upper or lower
    # part - keep them in sync with CblasLower and b'L' and it works out fine.
    cblas_dsyrk(CblasRowMajor, CblasLower, tr, n, k, 1.0, A, lda, 0.0, F, ldc)
    status = LAPACKE_dsyevd(CblasRowMajor, b'V', b'L', n, F, n, D)
    return status


cdef inline void c_center(double[:,::1] X,
                          double[::1] X_mean,
                          Py_ssize_t n,
                          Py_ssize_t m
                          ) nogil noexcept:
    cdef Py_ssize_t i = 0
    for i in range(n):
        X_mean[i] = (c_dsum(m, &X[i,0]) / <double>m)
        vdSubI(m, &X[i,0], 1, &X_mean[i], 0, &X[i,0], 1)


cdef inline double c_dsum(Py_ssize_t n, double *X) nogil noexcept:
    # The compiler should be able to pipeline & vectorize this implementation
    # if given the proper flags (-march=native, -mavx, -O2/3, etc.)
    cdef:
        Py_ssize_t i = 0
        Py_ssize_t r = n % 4
        Py_ssize_t m = n - r
        double total = 0.0
        double *acc = [0.0, 0.0, 0.0, 0.0]
    for i in range(0, m, 4):
        acc[0] += X[i]
        acc[1] += X[i+1]
        acc[2] += X[i+2]
        acc[3] += X[i+3]
    total = (acc[0]+acc[1]) + (acc[2]+acc[3])
    for i in range(r):
        total += X[m+i]
    return total



###############################################################################
#                                                                             #
#   PRE AND POST PROCESSING                                                   #
#                                                                             #
###############################################################################
def center(double[:,::1] X not None):
    """Center the rows of X inplace and return the means"""
    # Centers X inplace, and fills X_mean with the mean values used
    cdef double[::1] X_mean = aligned_doubles((X.shape[0],), PAGE_ALIGN)
    c_center(X, X_mean, X.shape[0], X.shape[1])
    return np.asarray(X_mean)


def whiten(double[:,::1] X not None,
           Py_ssize_t n_components = 0,
           double component_thresh = 1e-6,
           *,
           int align = PAGE_ALIGN):
    """Whitens X.

    X *must* be of shape [n_features, n_samples].

    `X1`, `K`, and `X_mean` are returned such that the formula `X1 = K(X -
    X_mean)` is true (`X` in the equation is not centered). The input `X` is
    centered inplace to avoid making a third copy of size `X`.
    """
    cdef:
        int status = 0
        Py_ssize_t i = 0
        Py_ssize_t j = 0
        Py_ssize_t n = X.shape[0]
        Py_ssize_t m = X.shape[1]
        double[::1] X_mean
        double *F
        double *D
        double eigv = 0.0
        double[:,::1] X1
        double[:,::1] K

    logging.info('Running whiten on dataset of shape [%d, %d]', n, m)
    logging.info('FastICA whiten estimated required RAM: %s',
                 format_nbytes(estimate_whiten_memory(n, m, n_components or n)))

    # Centers X inplace, and fills X_mean with the mean values used
    logging.info('Whiten calculating X_mean and centering X')
    X_mean = aligned_doubles((n,), align)
    c_center(X, X_mean, n, m)

    # compute eigens of `X @ X.T`, of shape [features, features]
    F = <double *>malloc_or_raise(n*n*sizeof(double), align)
    D = <double *>malloc_or_raise(n*sizeof(double), align)
    try:
        logging.info('Whiten seeking the eigenvalues of X@X.T')
        status = c_sym_eig(D, F, &X[0,0], n, m, 0)
        if status != 0:
            raise RuntimeError(f'whiten->dsyevd failed: status={status}')

        # If a specific n_components is not requested, then we calculate the
        # number of components to project X down to as the number of components
        # above `component_thresh` rounded up to the nearest multiple of 64,
        # capped at n.  N.B.: this algo is run on the raw eigenvalues, not
        # their sqrts!
        if n_components < 1:
            n_components = 0
            for i in range(n):
                if D[i] > component_thresh:
                    n_components += 1
            logging.info(f'Whiten: {n_components} eigvals > {component_thresh}')
            # round n_components DOWN to multiple of 64
            #n_components -= (n_components % 64)
            # round n_components UP to multiple of 64
            n_components += 64 - (n_components % 64)
            # We may have overshot and b above the possible number of components,
            # in which case we just set n_components to the max possible
            if n_components > n:
                n_components = n
            logging.info(f'Whiten: final n_components = {n_components}')

        X1 = aligned_doubles((n_components, m), align)
        K = aligned_doubles((n_components, n), align)
        # Clip at EPS for stability of sqrt
        vdFmaxI(n, D, 1, &EPS, 0, D, 1)
        vdSqrt(n, D, D)

        # The eigenvalues and corresponding eigenvectors returned by c_sym_eig are
        # in *ascending* order. We choose the top `n_components` values (and
        # corresponding column vectors), divide the vectors by the values, and
        # multiply by the sign of the leading element (so that the largest
        # eigenval's vector's leading element is always positive). The scaled
        # vectors are placed in the *rows* of K. Equivalent to the following code
        # using Python and numpy:
        #   d = d[::-1]
        #   u = u[:, ::-1]
        #   u *= np.sign(u[0])
        #   K = (u/d).T[:n_components]
        for i in range(n_components):
            # NB: Array subscripting does pointer dereferencing in Cython as in C
            eigv = D[(n-1)-i]
            if F[(n-1)-i] < 0:
                eigv *= -1.0
            for j in range(n):
                K[i, j] = F[(n*j)+(n-1)-i] / eigv
    finally:
        safe_mkl_free(F)
        safe_mkl_free(D)

    # whiten and project X via K:
    # X1 = np.dot(K, X) * np.sqrt(m)
    logging.info('Whiten projecting X1 = K(X - X_mean)')
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                n_components, m, n, sqrt(<double>m), &K[0,0], n,
                &X[0,0], m, 0.0, &X1[0,0], m)

    logging.info('Whiten completed: X1 is of shape [%d, %d]',
                 X1.shape[0], X1.shape[1])
    return np.asarray(X1), np.asarray(K), np.asarray(X_mean)


def apply_whitening(double[:,::1] Y, double[:,::1] K, double[::1] X_mean):
    """
    Returns Y1 = K(Y - X_mean), where K and X_mean are the whitening
    parameters determined by `whiten`. Y is centered by X_mean *inplace*!
    """
    # Equivalent: Y -= X_mean.reshape(-1, 1); Y1 = np.dot(K, Y)
    assert K.shape[1] == Y.shape[0] and Y.shape[0] == X_mean.shape[0]
    cdef:
        Py_ssize_t i = 0
        Py_ssize_t c = K.shape[0]
        Py_ssize_t n = Y.shape[0]
        Py_ssize_t m = Y.shape[1]
        double[:,::1] Y1 = aligned_doubles((c, m), PAGE_ALIGN)

    # Center Y by X_mean *inplace!*
    for i in range(n):
        vdSubI(m, &Y[i,0], 1, &X_mean[i], 0, &Y[i,0], 1)

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, c, m, n,
                1.0, &K[0,0], n, &Y[0,0], m, 0.0, &Y1[0,0], m)
    return np.asarray(Y1)


def _recover_S(double[:,::1] W, double[:,::1] X, bint scale_by_m = True):
    # Equivalent: S = np.dot(W, X)
    # If scale_by_m, S /= np.sqrt(X.shape[1])
    assert W.shape[0] == W.shape[1] and W.shape[1] == X.shape[0]
    cdef:
        Py_ssize_t n = W.shape[0]
        Py_ssize_t m = X.shape[1]
        double[:,::1] S = aligned_doubles((n, m), PAGE_ALIGN)
        double scale = 1.0
    # Whitened X contains an extra factor: sqrt(m), which isn't present in
    # freshly projected data, so we have to divide that out via the alpha param
    # to dgemm if this is the whitened data used in the actual ICA estimation.
    if scale_by_m:
        scale = 1.0/sqrt(<double>m)
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, m, n,
                scale, &W[0,0], n, &X[0,0], m, 0.0, &S[0,0], m)
    return S


def recover_S_from_WX1(double[:,::1] W, double[:,::1] X1):
    """Returns S = WX1 = WK(X-m); i.e. X1 is X centered and whitened"""
    return np.asarray(_recover_S(W, X1, scale_by_m=True))


def recover_A_from_WK(double[:,::1] W, double[:,::1] K):
    """Returns A = (WK)^{-1}; i.e. A is the pseudo-inverse of WK"""
    # Equivalent: A = scipy.linalg.pinv(np.dot(W, K))
    assert W.shape[0] == W.shape[1] and W.shape[1] == K.shape[0]
    # TODO: keep or replace this dependency on scipy?
    from scipy import linalg
    cdef:
        Py_ssize_t n = W.shape[0]
        Py_ssize_t m = K.shape[1]
        double[:,::1] A = aligned_doubles((n, m), PAGE_ALIGN)
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, m, n,
                1.0, &W[0,0], n, &K[0,0], m, 0.0, &A[0,0], m)
    return linalg.pinv(np.asarray(A), check_finite=False)


def apply_model(double[:,::1] W, double[:,::1] K,
                double[:,::1] Y, double[::1] X_mean,
                double[::1] scale_factors = None):
    """Applies the ICA model defined by S = WK(X-m) to arbitrary data Y; i.e.,
    produces the projection of Y into the given model.

    Y will be shifted *in-place* by X_mean. If scale_factors is not None, then
    the output will be scaled row-by-row by the given scale factors.
    """
    # Requires extra space eqaul to two copies of Y
    assert W.shape[0] == W.shape[1] and W.shape[1] == K.shape[0]
    assert K.shape[1] == Y.shape[0] and Y.shape[0] == X_mean.shape[0]
    cdef:
        Py_ssize_t i = 0
        double[:,::1] Y1 = apply_whitening(Y, K, X_mean)
        double[:,::1] SY = _recover_S(W, Y1, scale_by_m=False)
    if scale_factors is not None:
        assert SY.shape[0] == scale_factors.shape[0]
        for i in range(SY.shape[0]):
            cblas_dscal(SY.shape[1], 1.0/scale_factors[i], &SY[i,0], 1)
    return np.asarray(SY)


def scale_to_unit_variance(double[:,::1] S,
                           double[:,::1] AT,
                           double alpha = 1.0,
                           bint sign_flip = False):
    """Returns S /= (alpha * sd(S)) and AT *= (alpha * sd(S))

    S is shape [n_components, n_samples]
    A is shape [n_features, n_components], so that X = AS; hence AS (this
    function's input) must be [n_components, n_features].

    Since `A = pinv(dot(W, K))` and the output of `pinv` is F-contiguous, AT (A
    transpose) is C-contiguous. If A is copied or memory-transposed or
    otherwise *not* F-contiguous, care must be taken to ensure AT is
    C-contiguous as is expected by the present function.

    Returns the scaling factors applied to S and A.
    """
    assert S.shape[0] == AT.shape[0], 'N comp. should be both S and AT 1st dim'
    cdef:
        Py_ssize_t i = 0
        Py_ssize_t n = S.shape[0]
        Py_ssize_t m = S.shape[1]
        Py_ssize_t k = AT.shape[1]
        size_t amax_idx = 0
        double mu = 0.0
        double sd = 0.0
        double *tmp = <double *>malloc_or_raise(m*sizeof(double), PAGE_ALIGN)
        double[::1] factors = aligned_doubles((n,), PAGE_ALIGN)
    for i in range(n):
        # sd(S)
        mu = (c_dsum(m, &S[i,0]) / <double>m)
        vdSubI(m, &S[i,0], 1, &mu, 0, tmp, 1)
        vdSqr(m, tmp, tmp)
        sd = alpha * sqrt(c_dsum(m, tmp) / <double>m)
        if sign_flip:
            amax_idx = cblas_idamax(k, &AT[i,0], 1)
            if AT[i, amax_idx] < 0.0:
                sd *= -1.0
        # Scale: S/sd(S), but AT*sd(S)
        cblas_dscal(m, 1.0/sd, &S[i,0], 1)
        cblas_dscal(k, sd, &AT[i,0], 1)
        factors[i] = sd
    safe_mkl_free(tmp)
    return np.asarray(factors)


def inplace_vec_norm(double[:,::1] X):
    """Inplace divides the rows of X by the L2 norm"""
    cdef:
        Py_ssize_t i = 0
        Py_ssize_t n = X.shape[0]
        Py_ssize_t m = X.shape[1]
        double *tmp = <double *>malloc_or_raise(m*sizeof(double), PAGE_ALIGN)
        double norm = 0.0
    for i in range(n):
        vdSqr(m, &X[i,0], tmp)
        norm = 1.0 / sqrt(c_dsum(m, tmp))
        cblas_dscal(m, norm, &X[i,0], 1)
    safe_mkl_free(tmp)


def cosine_similarity(double[:,::1] X,
                      double[:,::1] Y,
                      bint norm_X = True,
                      bint norm_Y = True,
                      bint X_is_Y = False):
    """Returns the cosine similarity matrix between X and Y.

    If X and Y are L2 normalized this reduces to the matrix product XY.T;
    thus X and Y are L2 normalized - inplace, to avoid extraneous copies.
    """
    assert X.shape[1] == Y.shape[1], 'X & Y 2nd dim must match'
    cdef:
        Py_ssize_t i = 0
        Py_ssize_t n = X.shape[0]
        Py_ssize_t k = X.shape[1]
        Py_ssize_t m = Y.shape[0]
        double[:,::1] C = aligned_doubles((n, m), PAGE_ALIGN)

    if norm_X:
        inplace_vec_norm(X)

    if norm_Y and not X_is_Y:
        inplace_vec_norm(Y)

    # Now that X and Y are unit vectors, the cosine similarity between them is
    # just the matrix product XY^T
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n, m, k,
                1.0, &X[0,0], k, &Y[0,0], k, 0.0, &C[0,0], m)

    return np.asarray(C)


###############################################################################
#                                                                             #
#   FastICA Algorithm Implementation                                          #
#                                                                             #
###############################################################################
def fastica(double[:,::1] X not None,
            double[:,::1] W not None,
            *,
            long max_iter = 200,
            double tol = 1e-4,
            long checkpoint_iter = 0,
            str checkpoint_dir = './checkpoints',
            str checkpoint_iter_format = 'd',
            long start_iter = 0,
            long c = 1,
            long blockwidth = -1,
            long align = PAGE_ALIGN,
            str log_timing_format = '.4f'
            ):
    cdef:
        Py_ssize_t n = X.shape[0]
        Py_ssize_t m = X.shape[1]
        Py_ssize_t ldx = X.shape[1]
        Py_ssize_t offset = 0
        double *Xptr = &X[0,0]

    if 0 < blockwidth and blockwidth < ldx:
        m = blockwidth

    if not (n == W.shape[0] and n == W.shape[1]):
        raise ValueError('W must be square and of dim X.shape[0]')

    if start_iter > 0:
        logging.info(f'FastICA: resume training from iteration: {start_iter}')

    if c < 1: c = 1
    if c > n: c = n

    logging.info('FastICA: estimated required RAM %s',
                 format_nbytes(estimate_base_memory(n, m, c)))

    # Equivalent to `mkdir -p $checkpoint_dir` if we're checkpointing
    if checkpoint_iter > 0:
        logging.info('Checkpoints every %d iterations in %s',
                     checkpoint_iter, checkpoint_dir)
        Path(checkpoint_dir).mkdir(exist_ok=True, parents=True)

    cdef:
        int status = 0
        Py_ssize_t it = 0
        Py_ssize_t s = sizeof(double)
        # Loop control and time logging variables
        double lim, t_start, t0, t1, t2, t3, t4, t5, t6
        # Wrapped in a Python object so we can return it
        double[:,::1] W1 = aligned_doubles((n, n), align)
        # Internal buffers which will be released on function end
        double *gX = <double *>malloc_or_raise(n*m*s, align)
        double *gdv = <double *>malloc_or_raise(n*s, align)
        double *tmp_m = <double *>malloc_or_raise(c*m*s, align)
        double *tmp_W = <double *>malloc_or_raise(n*n*s, align)
        double *tmp_F = <double *>malloc_or_raise(n*n*s, align)
        double *tmp_D = <double *>malloc_or_raise(n*s, align)
        double *ones = <double *>malloc_or_raise(m*s, align)

    try:
        for it in range(m):
            ones[it] = 1.0
        # dummy call to warm up dsecnd (first call to CPU clocks is pricy)
        dsecnd()

        logging.info('Initial decorrelation of W')
        decorr_W(n, &W[0,0], tmp_D, tmp_F, tmp_W)

        logging.info('Starting main loop of FastICA algorithm')
        logging.info('it|gX|apply_g|W1|update_W|decorr_W|max_change|total|lim')
        # Used by logging calls in the main loop body
        f = lambda t: format(t, log_timing_format)
        g = lambda i: format(i, checkpoint_iter_format)

        t_start = dsecnd()
        offset = 0
        with nogil:
            for it in range(start_iter, start_iter+max_iter):
                if 0 < blockwidth and blockwidth < ldx:
                    offset = (it * m) % (ldx - (ldx % m))

                # gX = np.dot(W, X)
                t0 = dsecnd()
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            n, m, n, 1.0, &W[0,0], n,
                            Xptr+offset, ldx, 0.0, gX, m)

                # gX = np.tanh(gX); gdv = np.mean(1 - gX**2, axis=1)
                t1 = dsecnd()
                apply_g(n, m, c, gX, gdv, tmp_m, ones)

                # W1 = np.dot(gX, X.T) # ie. np.dot(np.tanh(np.dot(W, X)), X.T)
                t2 = dsecnd()
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                            n, n, m, (1.0 / <double>m), gX, m,
                            Xptr+(offset*n), ldx, 0.0, &W1[0,0], n)

                # W1 -= W*gdv[:,np.newaxis]  # row operation
                t3 = dsecnd()
                update_W(n, m, &W1[0,0], &W[0,0], gdv, tmp_W)

                # symmetric decorrelation of W1; i.e.  W1 = (WW^T)^{-1}W
                t4 = dsecnd()
                status = decorr_W(n, &W1[0,0], tmp_D, tmp_F, tmp_W)
                if status != 0:
                    with gil:
                        msg = f'decorr_W->dsyevd failed: status={status}'
                        raise RuntimeError(msg)

                # lim = max(abs(abs(np.einsum("ij,ij->i", W1, W)) - 1))
                t5 = dsecnd()
                lim = max_change(n, &W[0,0], &W1[0,0])

                t6 = dsecnd()
                with gil:
                    if 0 < blockwidth and blockwidth < ldx:
                        it_str = f'{it}\\{offset//m}'
                    else:
                        it_str = str(it)
                    logging.info('%s|%s|%s|%s|%s|%s|%s|%s|%.8f', it_str,
                                 f(t1-t0), f(t2-t1), f(t3-t2), f(t4-t3),
                                 f(t5-t4), f(t6-t5), f(t6-t0), lim)

                W, W1 = W1, W
                if lim < tol:
                    break

                if (checkpoint_iter > 0) and (it % checkpoint_iter == 0):
                    with gil:
                        np.save(f'{checkpoint_dir}/W.{g(it)}.npy', np.asarray(W))

        logging.info(f'FastICA: {(it-start_iter)+1} iterations'
                     f' run in {dsecnd()-t_start:.4f} seconds')
        return np.asarray(W), it+1
    finally:
        safe_mkl_free(gX)
        safe_mkl_free(gdv)
        safe_mkl_free(tmp_m)
        safe_mkl_free(tmp_W)
        safe_mkl_free(tmp_F)
        safe_mkl_free(tmp_D)
        safe_mkl_free(ones)


cdef void apply_g(Py_ssize_t n, Py_ssize_t m, Py_ssize_t c,
                  double *gX, double *gdv,
                  double *tmp_m, double *ones
                  ) nogil:
    cdef:
        Py_ssize_t i = 0
        Py_ssize_t j = 0
        Py_ssize_t n_batches = n // c
        Py_ssize_t remainder = n % c
    # gX = tanh(WX)
    vdTanh(n*m, gX, gX)

    # gdv = mean(1 - gX^2, axis=1)  # axis=1, i.e. component means
    # The following uses tmp_m as a holding place for rows of gX
    # pointwise squared. Then we set gdv to 1.0, alpha to -1/m, and compute
    # y = alphaAx + y with dgemv: since gdv = y, alpha = -1/m, and x is a
    # vector of all ones (it must be a real vector too, zero increments are
    # not allowed in lapack funcs), the effect is to output into gdv the
    # mean value of 1 - gX^2.
    cblas_dcopy(n, &ONE, 0, gdv, 1)

    for i in range(n_batches):
        vdSqr(c*m, gX+(i*m*c), tmp_m)
        cblas_dgemv(CblasRowMajor, CblasNoTrans, c, m,
                    (-1.0 / <double>m), tmp_m, m,
                    ones, 1, 1, gdv+(i*c), 1)

    if remainder > 0:
        vdSqr(m*remainder, gX+(n_batches*m*c), tmp_m)
        cblas_dgemv(CblasRowMajor, CblasNoTrans, remainder, m,
                    (-1.0 / <double>m), tmp_m, m,
                    ones, 1, 1, gdv+(n_batches*c), 1)


cdef void update_W(Py_ssize_t n, Py_ssize_t m,
                   double *W1, double* W,
                   double *gdv, double *tmp_W
                   ) nogil noexcept:
    # (row by row): W1 -= gdv*W
    cdef Py_ssize_t i = 0
    mkl_domatcopy(b'R', b'N', n, n, 1.0, W, n, tmp_W, n)
    for i in range(n):
        cblas_dscal(n, gdv[i], tmp_W+(i*n), 1)
    vdSub(n*n, W1, tmp_W, W1)


cdef int decorr_W(Py_ssize_t n, double *W,
                  double *D, double *F, double *tmp_W
                  ) nogil noexcept:
    cdef Py_ssize_t i = 0
    cdef int status = 0

    # D, F := (FDF^T) == WW^T
    status = c_sym_eig(D, F, W, n, n, 0)
    if status != 0:
        return status

    # Clipping D at TINY for numeric stability of InvSqrt
    vdFmaxI(n, D, 1, &TINY, 0, D, 1)
    vdInvSqrt(n, D, D)

    # Three matrix multiplications to get W = FDF^TW_0
    # This pattern of calls will (a) use only one working space matrix
    # additional to the three components F, D, W; and (b) ensure each
    # BLAS call has safe output matrices (BLAS is not in-place safe)

    # Start by copying W into W-shaped working space
    mkl_domatcopy(b'R', b'N', n, n, 1.0, W, n, tmp_W, n)

    # W := F^TW_0 ; starting from the right
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                n, n, n, 1.0, F, n, tmp_W, n, 0, W, n)

    # W_0 := DW
    # dscal is much faster than vdMulI on our expected scale of input (even
    # with the copy, which must happen behind the scenes with vdMulI
    # anyways) - but at *large* scale the time difference evaporates

    # NB: This copy *is necessary* = tmp_W is apparently partially
    # ovewritten by the above cblas_dgemm call (XXX: verify or disprove!)
    mkl_domatcopy(b'R', b'N', n, n, 1.0, W, n, tmp_W, n)
    for i in range(n):
        cblas_dscal(n, D[i], tmp_W+(i*n), 1)

    # W := FW_0 ; Voila, W is now decorrelated in-place
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                n, n, n, 1.0, F, n, tmp_W, n, 0, W, n)


cdef double max_change(Py_ssize_t n, double *W, double *W1) nogil noexcept:
    #Python: lim = max(abs(abs(np.einsum("ij,ij->i", W1, W)) - 1))
    cdef:
        Py_ssize_t i = 0
        double x = 0.0
        double x_max = 0.0
    for i in range(n):
        x = cblas_ddot(n, W+(i*n), 1, W1+(i*n), 1)
        x = fabs(fabs(x) - 1.0)
        if x > x_max:
            x_max = x
    return x_max


###############################################################################
#                                                                             #
#   COMMAND LINE TOOL                                                         #
#                                                                             #
###############################################################################
def pprint_param_ns(ns, prefix=''):
    nsdict = vars(ns)
    fmt = f'>{max(len(k) for k in nsdict)}s'
    for key, val in sorted(nsdict.items()):
        logging.info('%s %s: %s', prefix, format(key, fmt), val)


def load_aligned(file, align=PAGE_ALIGN, transpose=False):
    X = np.load(file, mmap_mode='c')
    X = align_ndarray(X, align=align, transpose=transpose)
    return X


def cli():
    """A performance-oriented implementation of FastICA"""
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='command')
    all_parsers = [parser]

    # whiten subcommand whitens data
    whiten_parser = subparsers.add_parser('whiten')
    all_parsers.append(whiten_parser)
    f = whiten_parser.add_argument
    f('data', type=Path)
    f('-X', '--X1-file', type=Path, default='X1.npy')
    f('-K', '--K-file', type=Path, default='K.npy')
    f('-M', '--X-mean-file', type=Path, default='X_mean.npy')
    f('-c', '--n-components', type=int, default=0)
    f('-t', '--component-thresh', type=float, default=1e-6)
    f('--transpose', action='store_true')

    # run subcommand runs the fastica estimation algorithm
    run_parser = subparsers.add_parser('run')
    all_parsers.append(run_parser)
    f = run_parser.add_argument
    f('data', type=Path)
    f('-W', '--W-file', type=Path, default='W.npy')
    f('--w-init', type=Path)
    f('--max-iter', type=int, default=200)
    f('--tol', type=float, default=1e-4)
    f('--checkpoint-iter', type=int, default=0)
    f('--checkpoint-dir', type=Path, default='./checkpoints')
    f('--checkpoint-iter-format', type=str, default='d')
    f('--start-iter', type=int, default=0)
    f('--cwork', type=int, default=1)
    f('--retry', type=int, default=1)

    # recover subcommand produces the S & A which correspond to the X data used
    # in FastICA estimation
    recover_parser = subparsers.add_parser('recover')
    all_parsers.append(recover_parser)
    f = recover_parser.add_argument
    f('-X', '--X1-file', type=Path, default='X1.npy')
    f('-W', '--W-file', type=Path, default='W.npy')
    f('-K', '--K-file', type=Path, default='K.npy')
    f('-S', '--S-file', type=Path, default='S.npy')
    f('-A', '--A-file', type=Path, default='A.npy')
    # u for unit scaling
    f('-u', '--scale', action='store_true')
    f('-a', '--alpha', type=float, default=1.0)
    # p for "positive" (sign flipping of major components)
    f('-p', '--sign-flip', action='store_true')
    f('-f', '--factors-file', type=Path, default='factors.npy')

    # project subcommand projects new data Y through the model defined by W, K,
    # and X_mean, with optional additional scaling factors
    project_parser = subparsers.add_parser('project')
    all_parsers.append(project_parser)
    f = project_parser.add_argument
    f('-Y', '--Y-file', type=Path, default='Y.npy')
    f('-W', '--W-file', type=Path, default='W.npy')
    f('-K', '--K-file', type=Path, default='K.npy')
    f('-M', '--X-mean-file', type=Path, default='X_mean.npy')
    f('-f', '--factors-file', type=Path, default='factors.npy')
    f('-S', '--S-file', type=Path, default='S.npy')

    # Common arguments: stupid argparse is set up such that arguments added to
    # the parent parsers apparently don't propagate down to subcommand parsers
    for p in all_parsers:
        f = p.add_argument
        f('-v', '--verbose', action='count', default=0)
        f('--log-file', type=Path, default=None)
        f('--log-timing-format', type=str, default='.4f')
        f('--memalign', type=int, default=PAGE_ALIGN)
    del f
    args = parser.parse_args()

    if args.verbose > 0 or args.log_file:
        level = logging.INFO
        if args.verbose > 1:
            level = logging.DEBUG
        logging.basicConfig(filename=args.log_file,
                            level=level,
                            format='%(asctime)s %(name)s %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S %p')

    logging.info('FastICA %s (pid=%d): %s with parameter namespace = ',
                 __version__, os.getpid(), args.command)
    pprint_param_ns(args)

    if args.memalign == 'cache':
        args.memalign = CACHE_ALIGN
    elif args.memalign == 'page':
        args.memalign = PAGE_ALIGN
    else:
        args.memalign = int(args.memalign)

    if (args.memalign % 8) > 0:
        raise ValueError('memalign must be divisible by 8 (64 recommended)')

    if args.command == 'whiten':
        logging.info('Loading data file %s with%s transpose',
                     args.data, '' if args.transpose else 'out')
        XT = load_aligned(args.data, args.memalign, args.transpose)
        # whiten handles logging start/stop/etc internally
        X1, K, X_mean = whiten(XT, args.n_components, args.component_thresh,
                               align=args.memalign)
        logging.info('Saving X1 to %s', args.X1_file)
        np.save(args.X1_file, X1)
        logging.info('Saving K to %s', args.K_file)
        np.save(args.K_file, K)
        logging.info('Saving X_mean to %s', args.X_mean_file)
        np.save(args.X_mean_file, X_mean)
        logging.info('FastICA whiten DONE')

    elif args.command == 'run':
        logging.info('Loading X1 from %s', args.data)
        X1 = load_aligned(args.data, args.memalign)
        n = X1.shape[0]
        for n_try in range(1, args.retry+1):
            logging.info('FastICA attempt %d: generating W_init', n_try)
            W_init = np.random.normal(size=(n, n))
            W_init = align_ndarray(W_init, args.memalign)
            W, it = fastica(X1, W_init,
                            max_iter=args.max_iter,
                            tol=args.tol,
                            checkpoint_iter=args.checkpoint_iter,
                            checkpoint_dir=str(args.checkpoint_dir),
                            checkpoint_iter_format=args.checkpoint_iter_format,
                            start_iter=args.start_iter,
                            c=args.cwork,
                            align=args.memalign,
                            log_timing_format=args.log_timing_format)
            if it < args.max_iter:
                logging.info('Attempt %d success in %d iterations', n_try, it)
                break
            # If we need to retry the algorithm, remove any checkpoints
            if args.checkpoint_iter > 0:
                shutil.rmtree(args.checkpoint_dir)
        else:
            logging.info('No convergence in %d attempts', args.retry)
        logging.info('Persisting W_init to %s', args.w_init)
        np.save(args.w_init, W_init)
        logging.info('Persisting W to %s', args.W_file)
        np.save(args.W_file, W)
        logging.info('FastICA main algorithm DONE')

    elif args.command == 'recover':
        logging.info('Loading X1 data from %s', args.X1_file)
        X1 = load_aligned(args.X1_file, args.memalign)
        logging.info('Loading est. W from %s', args.W_file)
        W = load_aligned(args.W_file, args.memalign)
        logging.info('Loading K from %s', args.K_file)
        K = load_aligned(args.K_file, args.memalign)
        logging.info('Recovering sources S')
        S = recover_S_from_WX1(W, X1)
        logging.info('Recovering components A')
        A = recover_A_from_WK(W, K)
        if args.scale:
            logging.info('Scaling A & S')
            factors = scale_to_unit_variance(S, A.T, alpha=args.alpha,
                                             sign_flip=args.sign_flip)
            logging.info('Saving scale factors to %s', args.factors_file)
            np.save(args.factors_file, factors)
        logging.info('Persisting S to %s', args.S_file)
        np.save(args.S_file, S)
        logging.info('Persisting A to %s', args.A_file)
        np.save(args.A_file, A)
        logging.info('FastICA recovering A and S DONE')

    elif args.command == 'project':
        logging.info('Loading Y data from %s', args.Y_file)
        Y = load_aligned(args.Y_file, args.memalign)
        logging.info('Loading est. W from %s', args.W_file)
        W = load_aligned(args.W_file, args.memalign)
        logging.info('Loading K from %s', args.K_file)
        K = load_aligned(args.K_file, args.memalign)
        logging.info('Loading X_mean from %s', args.X_mean_file)
        X_mean = load_aligned(args.X_mean_file, args.memalign)
        factors = None
        if args.factors_file is not None:
            logging.info('Loading optional factors from %s', args.factors_file)
            factors = load_aligned(args.factors_file, args.memalign)
        logging.info('Projecting new data Y through ICA Model')
        SY = apply_model(W, K, Y, X_mean, factors)
        logging.info('Persisting projections S(Y) to %s', args.S_file)
        np.save(args.S_file, SY)
        logging.info('Projecting new data DONE')

    else:
        parser.print_usage()


###############################################################################
#                                                                             #
#   UNIT TESTS                                                                #
#                                                                             #
###############################################################################
def setUpModule():
    # Unittest module-level setup: import libraries used in tests but that the
    # fastica code itself shouldn't depen on so that they're cached in
    # sys.modules and ready to go; also if any import fails we just won't run
    # the tests.
    try:
        import scipy
        import scipy.linalg
        import scipy.signal
    except ImportError as err:
        raise unittest.SkipTest('cannot find scipy') from err
    try:
        import sklearn
        import sklearn.decomposition
    except ImportError as err:
        raise unittest.SkipTest('cannot find sklearn') from err


class FastICATestBase(unittest.TestCase):
    def setUp(self):
        # Always make sure mkl's caches are clear - some tests rely on this
        # being the case, it isn't for performance.
        mkl_free_buffers()


class TestSignalReconstruction(FastICATestBase):
    @staticmethod
    def gen_test_signals(seed=0):
        from scipy import signal

        np.random.seed(seed)
        n_samples = 2000
        time = np.linspace(0, 8, n_samples)

        s1 = np.sin(2 * time)
        s2 = np.sign(np.sin(3 * time))
        s3 = signal.sawtooth(2 * np.pi * time)

        S = np.row_stack((s1, s2, s3))
        S += 0.2 * np.random.normal(size=S.shape)

        S /= S.std(axis=1, keepdims=True)
        A = np.array([[1.0, 0.5, 1.5],
                      [1.0, 2.0, 1.0],
                      [1.0, 1.0, 2.0]])
        # With X properly defined as [features, samples], the ICA equation is
        # straightforwardly calculated: X = AS
        X = np.dot(A, S)
        return X, A, S

    def setUp(self):
        super().setUp()
        from sklearn.decomposition import _fastica as sk_fastica
        self.X, self.A, self.S = self.gen_test_signals()
        self.W = np.random.normal(size=self.A.shape)
        self.sk_ica = sk_fastica.FastICA(n_components=3,
                                         w_init=np.copy(self.W),
                                         whiten_solver='eigh',
                                         whiten='unit-variance',
                                         # defaults
                                         max_iter=200,
                                         tol=1e-04)

    def test_signal_reconstruction(self):
        # Automatically tests the reconstruction error, but this really needs
        # to be checked visually well. Should this be two or more tests? Maybe
        # check against sklearn and against the original separately.
        cdef:
            Py_ssize_t n = self.X.shape[0]
            Py_ssize_t m = self.X.shape[1]

        # Get the reconstruction parts from scikit-learn. Bear in mind that out
        # interface specifies X as [features, samples], so there are certain
        # transposes that must be made to work with scikit-learn's API
        sk_S = self.sk_ica.fit_transform(np.copy(self.X.T)).T
        sk_A = self.sk_ica.mixing_
        sk_X = np.dot(sk_A, sk_S) + self.sk_ica.mean_[:,np.newaxis]
        sk_check = np.allclose(self.X, sk_X)
        self.assertTrue(sk_check, 'Scikit-learn reconstruction should work')

        # Get the reconstruction parts from our code
        X0 = np.copy(self.X)
        X1, K, mean = whiten(X0, 3)
        W, _ = fastica(X1, np.copy(self.W))

        S = recover_S_from_WX1(W, X1)
        A = recover_A_from_WK(W, K)
        cy_X = np.dot(A, S) + mean[:, np.newaxis]

        sk_vs_cy = np.allclose(cy_X, sk_X)
        self.assertTrue(sk_vs_cy, 'Reconstruction should match sklearn recon')

        cy_check = np.allclose(self.X, cy_X)
        self.assertTrue(cy_check, 'Reconstruction should match actual input')

    def test_apply_whitening(self):
        # XXX / TODO - what to do about the sqrt(m) factor in whiten?
        X1, K, X_mean = whiten(np.copy(self.X), 3)
        Y = np.random.normal(size=self.X.shape)
        Y_copy = np.copy(Y)
        K_copy = np.copy(K)
        m_copy = np.copy(X_mean)
        cy_Y1 = apply_whitening(Y_copy, K, X_mean)

        self.assertTrue(np.all(K == K_copy), 'should not mutate K')
        self.assertTrue(np.all(X_mean == m_copy), 'should not mutate X_mean')

        check = np.allclose(Y-X_mean[:,np.newaxis], Y_copy)
        self.assertTrue(check, 'Y should be centered by X_mean inplace')

        np_Y1 = np.dot(K, Y-X_mean[:, np.newaxis])
        check = np.allclose(cy_Y1, np_Y1)
        self.assertTrue(check, 'cython and numpy results should match')

    def test_recover_S_from_WX(self):
        X1, K, mean = whiten(np.copy(self.X), 3)

        check = np.allclose(X1,
            np.dot(K, self.X-mean[:,np.newaxis])*np.sqrt(self.X.shape[1])
        )
        self.assertTrue(check, 'X1 should be K(X-mean)*sqrt(m)')

        W, _ = fastica(X1, np.copy(self.W))

        W_orig = np.copy(W)
        X1_orig = np.copy(X1)
        cy_S = recover_S_from_WX1(W, X1)
        self.assertTrue(np.all(W == W_orig), 'should not mutate W')
        self.assertTrue(np.all(X1 == X1_orig), 'should not mutate X1')

        np_S = np.dot(np.dot(W, K), self.X-mean[:,np.newaxis])
        self.assertTrue(np.allclose(cy_S, np_S), 'np & cy should match')

    def test_recover_A_from_WK(self):
        from scipy import linalg
        X1, K, mean = whiten(np.copy(self.X), 3)
        W, _ = fastica(X1, np.copy(self.W))

        W_orig = np.copy(W)
        K_orig = np.copy(K)
        cy_A = recover_A_from_WK(W, K)
        self.assertTrue(np.all(W == W_orig), 'should not mutate W')
        self.assertTrue(np.all(K == K_orig), 'should not mutate X1')

        np_A = linalg.pinv(np.dot(W, K), check_finite=False)
        self.assertTrue(np.allclose(cy_A, np_A), 'np & cy should match')

    def test_scale_to_unit_variance(self):
        from scipy import linalg
        X1, K, mean = whiten(np.copy(self.X), 3)
        W, _ = fastica(X1, np.copy(self.W))

        # Get unit-variance scaled S and A via numpy
        np_S = np.dot(np.dot(W, K), self.X - mean[:,np.newaxis])
        sd_S = np.std(np_S, axis=1, keepdims=True)
        np_S /= sd_S
        np_A = linalg.pinv(np.dot(W/sd_S.T, K), check_finite=False)

        # Get S by the fast-path from W and X1 directly
        cy_S = recover_S_from_WX1(W, X1)
        # NB: cy_A is F-contiguous! If we change that, downstream scaling to
        # unit or any other variance will need to be adjusted accordingly.
        cy_A = recover_A_from_WK(W, K)

        S_orig = np.copy(cy_S)
        A_orig = np.copy(cy_A)
        cy_sd_S = np.std(cy_S, axis=1, keepdims=1)

        scale_to_unit_variance(cy_S, cy_A.T)
        check_S = np.allclose(cy_S, S_orig/cy_sd_S)
        self.assertTrue(check_S, 'S should be scaled in place')
        check_A = np.allclose(cy_A.T, A_orig.T*cy_sd_S)
        self.assertTrue(check_A, 'A should be scaled in place')

        self.assertTrue(np.allclose(cy_A, np_A), 'np & cy A should match')
        self.assertTrue(np.allclose(cy_S, np_S), 'np & cy S should match')


class TestDriverCdefFunctions(FastICATestBase):
    def test_apply_g(self):
        cdef:
            Py_ssize_t s = sizeof(double)
            Py_ssize_t n = 100
            Py_ssize_t m = 1000
            Py_ssize_t c = 0
            Py_ssize_t i = 0
            double[:,::1] W, X, gX
            double[::1] gdv
            double *tmp_m
            double *ones = <double *>malloc_or_raise(m*s, PAGE_ALIGN)

        for i in range(m):
            ones[i] = 1.0

        for c in (1, 10, 32, 47, 100):
            # We have to make W *small* - in apply_g, gX and gdv are equivalent
            # to:
            #   gX = np.tanh(np.dot(W, X))
            #   gdv = np.mean(1 - gX**2, axis=1)
            # Now: the first iteration through with default random_sample
            # values, np.dot(W, X) will all be > 1.0 - hence it is true that
            # np.allclose(1.0, gX). In turn, np.allclose(0, gdv) is true.
            # `gdv` is *also* (almost always) np.allclose(0, gdv), because
            # random noise from malloc is almost always close to zero.
            # Therefore, these tests will almost always pass without ever
            # checking that apply_g does *anything* to `gdv` unless we make
            # sure that the output of tanh(WX) isn't all 1's. We do this by
            # dividing W by the number of samples.
            W = np.random.random_sample((n, n)) / m
            X = np.random.random_sample((n, m))

            W_orig = np.copy(W)
            X_orig = np.copy(X)
            np_gX = np.tanh(np.dot(W_orig, X_orig))
            np_gdv = np.mean(1 - np_gX**2, axis=1)

            gX = np.dot(W, X)
            gdv = np.zeros(n)

            tmp_m = <double *>malloc_or_raise(c*m*s, PAGE_ALIGN)
            apply_g(n, m, c, &gX[0,0], &gdv[0], tmp_m, ones)
            safe_mkl_free(tmp_m)

            with self.subTest(c=c):
                check_gX = np.allclose(np_gX, np.asarray(gX))
                self.assertTrue(check_gX, 'gX should match numpy')

                check_gdv = np.allclose(np_gdv, np.asarray(gdv))
                self.assertTrue(check_gdv, 'mean(g`X) should match numpy')
        safe_mkl_free(ones)

    def test_update_W(self):
        cdef:
            Py_ssize_t n = 100
            Py_ssize_t m = 1000
            Py_ssize_t s = sizeof(double)
            double[:,::1] W1 = np.zeros((n, n))
            double[:,::1] W = np.random.random_sample((n, n))
            double[:,::1] X = np.random.random_sample((n, m))
            double[:,::1] gX = np.tanh(np.dot(W, X))
            double[::1] gdv = np.mean(1 - gX.base**2, axis=1)
            double *tmp_W = <double *>malloc_or_raise(n*n*s, PAGE_ALIGN)

        W_orig = np.copy(W)
        X_orig = np.copy(X)
        gX_orig = np.copy(gX)
        gdv_orig = np.copy(gdv)

        W1 = np.dot(gX, X.T) / <double>m
        update_W(n, m, &W1[0,0], &W[0,0], &gdv[0], tmp_W)
        safe_mkl_free(tmp_W)

        self.assertTrue(np.all(W.base == W_orig), 'update_W ought not mutate W')
        self.assertTrue(np.all(X.base == X_orig), 'update_W ought not mutate X')
        self.assertTrue(np.all(gX.base == gX_orig),
                        'update_W ought not mutate gX')
        self.assertTrue(np.all(gdv.base == gdv_orig),
                        'update_W ought not mutate gdv')

        np_W1 = np.dot(gX, X.T)/m - gdv.base[:,np.newaxis]*W
        self.assertTrue(np.allclose(np_W1, W1.base), 'W1 should match numpy')

    def test_max_change(self):
        cdef:
            Py_ssize_t n = 1000
            Py_ssize_t m = 1000
            double[:,::1] X = np.random.random_sample((n, n))
            double[:,::1] Y = np.random.random_sample((n, n))
            double cy_max = 0.0

        with self.subTest("Max deviation should match ref implementation"):
            py_max = max(abs(abs(np.einsum("ij,ij->i", X, Y)) - 1))
            cy_max = max_change(n, &X[0,0], &Y[0,0])
            check = np.allclose(py_max, cy_max)
            self.assertTrue(check, "Max change should match")

        with self.subTest("max_change : f(X, Y) should equal f(Y, X)"):
            cy_max_yx = max_change(n, &Y[0,0], &X[0,0])
            check = np.allclose(cy_max, cy_max_yx)
            self.assertTrue(check, "max_change should be associative")

    def _check_eigs(self, A, eigenvals, eigenvecs, msg='',
                   atol=1000*np.finfo(np.double).eps, rtol=0.0):
        """Utility to check that the eigensystems are in fact eigensystem."""
        A = np.asarray(A)
        eigenvals = np.asarray(eigenvals)
        eigenvecs = np.asarray(eigenvecs)
        # Equivalent to but more explicit than: (A@w - v*w)
        A0 = np.dot(A, eigenvecs) - np.multiply(eigenvals, eigenvecs)
        self.assertTrue(np.allclose(A0, 0.0, atol=atol, rtol=rtol), msg)

    def test_c_sym_eig(self):
        # This set of tests is mostly to verify that I, the programmer, have not
        # made any (more) stupid usage errors passing arguments to LAPACK or BLAS
        from scipy.linalg import eigh

        # The division by sqrt(n) is necessary to keep the dot products A@A.T
        # and A.T@A in the right *scale* for our eigenvalue test against eps to
        # be meaningful.
        cdef:
            double[:,::1] A = np.random.random_sample((100, 100)) / 10
            double[:,::1] F = np.empty((100, 100))
            double[::1] D = np.empty(100)
            double[:,::1] A_orig = np.copy(A)

        ATA = np.dot(A.T, A)
        with self.subTest(matrix='A.T @ A', lib='scipy'):
            # w is lambdas (eigenvalues); v is the eigenvectors
            # scipy.linalg.eigh returns (values, vectors)
            w, v = eigh(np.copy(ATA))
            self._check_eigs(ATA, w, v, "You're so bad even scipy's eigh fails")

        with self.subTest(matrix='A.T @ A', lib='fastica'):
            c_sym_eig(&D[0], &F[0,0], &A[0,0], 100, 100, 1)
            asc_check = np.all(np.sort(D.base) == D.base)
            self.assertTrue(asc_check, 'eigvals should be in ascending order')
            self._check_eigs(ATA, D, F, 'c_sym_eig(..., transpose=True) failed')

        A = np.copy(A_orig)
        AAT = np.dot(A, A.T)
        with self.subTest(matrix='A @ A.T', lib='scipy'):
            w, v = eigh(np.copy(AAT))
            self._check_eigs(AAT, w, v, "You're so bad even scipy's eigh fails")

        with self.subTest(matrix='A @ A.T', lib='fastica'):
            c_sym_eig(&D[0], &F[0,0], &A[0,0], 100, 100, 1)
            asc_check = np.all(np.sort(D.base) == D.base)
            self.assertTrue(asc_check, 'eigvals should be in ascending order')
            self._check_eigs(AAT, w, v, 'c_sym_eig(..., transpose=False) failed')


class TestSymmetricDecorrelation(FastICATestBase):
    def setUp(self):
        super().setUp()
        cdef:
            Py_ssize_t n = 50
            Py_ssize_t s = sizeof(double)
            double[:,::1] w_init = np.random.random_sample((n, n))
            double[:,::1] W = np.copy(w_init)
            double *F = <double *>malloc_or_raise(n*n*s, PAGE_ALIGN)
            double *D = <double *>malloc_or_raise(n*s, PAGE_ALIGN)
            double *tmp_W = <double *>malloc_or_raise(n*n*s, PAGE_ALIGN)
        # Run the decorrelation
        decorr_W(n, &W[0,0], D, F, tmp_W)
        safe_mkl_free(F)
        safe_mkl_free(D)
        safe_mkl_free(tmp_W)
        self.n = n
        self.w_init = np.asarray(w_init)
        self.W = np.asarray(W)

    def test_shape(self):
        self.assertEqual(self.w_init.shape, self.W.shape)

    def test_sym_decorrelation_against_sk(self):
        # N.B. - This test uses scikit-learn's non-public _sym_decorrelation
        # function as a reference implementation to test against. If sklearn
        # changes this may break.
        from sklearn.decomposition import _fastica as sk_fastica
        sk_W = sk_fastica._sym_decorrelation(self.w_init)
        check = np.allclose(sk_W, self.W)
        self.assertTrue(check, 'fails to match the reference implementation')

    def test_sym_decorrelation_eigenvalues(self):
        # If W has been correctly symmetrically decorrelated then the
        # eigenvalues of WW^T must all be identically one. This property leads
        # also to idempotence of the operation (tested next).
        #   LaTeX illustration of this property:
        # $$
        # \begin{align}
        # \mathbf{W}_{next}
        #       &= (\mathbf{W}\mathbf{W}^T)^{-1/2}\mathbf{W} \\
        #       &= (\mathbf{F}\mathbf{D}^{-1/2}\mathbf{F}^T)\mathbf{W} \\
        #       &= (\mathbf{F}\mathbf{I}\mathbf{F}^T)\mathbf{W} \\
        #       &= (\mathbf{F}\mathbf{F}^T)\mathbf{W} \\
        #       &= \mathbf{I}\mathbf{W} \\
        #       &= \mathbf{W} \\
        # \end{align}
        # $$
        from scipy.linalg import eigh
        WWT = np.dot(self.W, self.W.T)
        d, _ = eigh(WWT)
        check = np.allclose(d, 1.0)
        self.assertTrue(check, 'eigenvalues of W should be identically 1')

    def test_idempotence(self):
        cdef:
            Py_ssize_t n = self.n
            Py_ssize_t s = sizeof(double)
            double[:,::1] W1 = np.copy(self.W)
            double *F = <double *>malloc_or_raise(n*n*s, PAGE_ALIGN)
            double *D = <double *>malloc_or_raise(n*s, PAGE_ALIGN)
            double *tmp_W = <double *>malloc_or_raise(n*n*s, PAGE_ALIGN)
        # Run the decorrelation
        decorr_W(n, &W1[0,0], D, F, tmp_W)
        safe_mkl_free(F)
        safe_mkl_free(D)
        safe_mkl_free(tmp_W)
        check = np.allclose(W1.base, self.W)
        self.assertTrue(check, 'f(W) == f(f(W)) should be True')


class TestWhitening(unittest.TestCase):
    def setUp(self):
        self.n_samples = 50
        self.n_features = 10
        self.n_components = 5
        self.X = np.random.random_sample((self.n_features, self.n_samples))
        self.X_orig = np.copy(self.X)
        (
            self.X1,
            self.K,
            self.X_mean
        ) = whiten(self.X, self.n_components)

    def test_mean(self):
        feature_mean = np.mean(self.X_orig, axis=1)
        self.assertEqual(self.X_mean.shape, (self.n_features,),
                         "Mean should be over features")
        check = np.allclose(feature_mean, self.X_mean)
        self.assertTrue(check, 'Mean does not match mean of pre-centered X')

    def test_X_centered_inplace(self):
        post_whiten_mean = np.mean(self.X, axis=1)
        check = np.allclose(post_whiten_mean, 0.0)
        self.assertTrue(check, 'X should be centered after whiten(X)')

    def test_X1_shape(self):
        n_features, n_samples = self.X.shape
        self.assertEqual(self.X1.shape, (self.n_components, self.n_samples),
                         'X1 should have shape [n_components, n_samples]')

    def test_K_shape(self):
        self.assertEqual(self.K.shape, (self.n_components, self.n_features),
                         'K should have shape [n_components, n_features]')

    def test_X1_is_centered(self):
        X1_mean = np.mean(self.X1, axis=1)
        check = np.allclose(X1_mean, 0.0)
        self.assertTrue(check, 'X1 features should be centered')

    def test_X1_is_whitened(self):
        # ddof=0 is essential!
        X1_cov = np.cov(self.X1, ddof=0)
        check = np.allclose(X1_cov, np.eye(X1_cov.shape[0]))
        self.assertTrue(check, 'X1 cov matrix should be identity matrix')

    def test_adaptive_n_components(self):
        # Relatively expensive test
        r = 700
        n = 1000
        A = np.random.random_sample((r, n))
        # X is a 1000 by 1000 matrix with rank (probably) 700
        X = np.dot(A.T, A)

        X1, K, X_mean = whiten(X, n_components=-1, component_thresh=1e-6)
        c = X1.shape[0]
        self.assertEqual((c, n), K.shape, 'K and X1 shapes must match')

        # this test depends on A having rank at most 700
        self.assertEqual(c, 704, 'c should be rounded up to multiple of 64')

        # This check should always pass no matter what
        check = (c % 64 == 0) or (c == n)
        self.assertTrue(check, 'Num components must be multiple of 64 or n')


class TestUtilities(FastICATestBase):
    # NB: the correctness of many of these tests implicitly require
    # FastICATestBase.setUp(), which frees MKL's internal buffers

    def test_aligned_copy(self):
        cdef:
            size_t allocated_bytes
            int allocated_buffs
        allocated_bytes = mkl_mem_stat(&allocated_buffs)
        self.assertEqual(allocated_bytes, 0, 'allocated_bytes should be zero')
        self.assertEqual(allocated_buffs, 0, 'allocated_buffs should be zero')

        # Get random samples until we have a mis-aligned buffer
        offset = 0
        attempts = []
        while offset == 0:
            X_orig = np.random.random_sample(1_000_000)
            # I think a reference must be kept else np will re-use the memory
            # buffer and the alignment of X_orig won't change
            attempts.append(X_orig)
            offset = X_orig.__array_interface__['data'][0] % 64
        self.assertNotEqual(offset, 0, 'X_orig should not be 64 byte aligned')
        # clean up the prior attempts
        del attempts

        # Must cast to an array to get the array interface
        X_alig = np.asarray(aligned_copy(X_orig))
        offset = X_alig.__array_interface__['data'][0] % 64
        self.assertEqual(offset, 0, 'copy of X should be 64 byte aligned')

        allocated_bytes = mkl_mem_stat(&allocated_buffs)
        self.assertGreaterEqual(allocated_bytes, X_alig.nbytes,
                               'mkl_mem_stat should show bytes allocated')
        self.assertEqual(allocated_buffs, 1,
                         'mkl_mem_stat should show one buffer allocated')

        # *Should* free the underlying buffer, since this is the only reference
        # XXX: don't add other references to X_alig!
        del X_alig

        allocated_bytes = mkl_mem_stat(&allocated_buffs)
        self.assertEqual(allocated_buffs, 0,
                         'mkl_mem_stat should show the buffer has been freed')

    @unittest.skipUnless(os.getenv('FASTICA_TEST_ALL'), 'test is *expensive*')
    def test_ILP64_enabled(self):
        # This is an expensive test - it takes about 1min and up to 40+ GiB.
        # We make sure that MKL's VML functions won't explode or silently fail
        # with realistic working size inputs. An even more realistic size would
        # be 8k by 1.8 million but we don't want to be here all day: 3968*600k
        # is sufficiently greater than signed 32-bit int max 2^31-1 for tests.
        cdef:
            Py_ssize_t n = 3968
            Py_ssize_t m = 600000
            double[:,::1] X = aligned_copy(np.random.random_sample((n, m)))
            double[:,::1] Y = aligned_copy(np.zeros((n, m)))
        # any VML func would do but we actually use tanh in the algo
        vdTanh(n*m, &X[0,0], &Y[0,0])
        # Checks that the indices higher than 2^32-1 are actually processed
        # NB: n-1, not -1: wraparound is OFF for this file's contents
        check = np.allclose(np.tanh(X[n-1]), Y[n-1])
        self.assertTrue(check, 'VML functions should process n_elem > (2^31)-1')

    def test_inplace_vec_norm(self):
        X = np.random.random_sample((100, 1000))
        Y = np.copy(X)
        norms = np.einsum("ij,ij->i", X, X)
        np.sqrt(norms, out=norms)
        X /= norms.reshape(-1, 1)
        inplace_vec_norm(Y)
        self.assertTrue(np.allclose(Y, X), 'normalization should match')

    def test_cosine_similarity(self):
        from sklearn.metrics.pairwise import cosine_similarity as sk_csim
        X = np.random.random_sample((100, 1000))
        Y = np.random.random_sample((100, 1000))
        X0 = np.copy(X)
        Y0 = np.copy(Y)
        C0 = sk_csim(X, Y)
        C = cosine_similarity(X, Y)
        self.assertTrue(np.allclose(C0, C), 'similarities should match')
