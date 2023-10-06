# cython: language_level=3

# Set and get the MXCSR flags for
#   - FTZ (Flush to Zero - subnormals outputs are set to zero)
#   - DAZ (Denormals are Zero - subnormal inputs are set to zero)
cdef extern from *:
    """
#include <xmmintrin.h>
#include <pmmintrin.h>

void set_daz(void) {
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
}

void set_ftz(void) {
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
}

int test_ftz(void) {
    if (_MM_GET_FLUSH_ZERO_MODE())
        return 1;
    return 0;
}

int test_daz(void) {
    if (_MM_GET_DENORMALS_ZERO_MODE())
        return 1;
    return 0;
}
    """
    # Make the functions visible to Cython
    void set_daz()
    void set_ftz()
    int get_daz()
    int get_ftz()


cdef extern from "mkl.h" nogil:
    # Double precision time - use for wall-timing (logging, etc)
    double dsecnd()

    # Needs to be turned on before it can report; degrades
    # performance. Arguments enable/disable/reset/etc.
    long long mkl_peak_mem_usage(int mode)

    # outputs the number of buffers into the buffers
    # arg and returns the number of bytes allocated
    long long mkl_mem_stat(int * buffers)

    # alloc_size is nbytes; alignment should usually be 64
    void* mkl_malloc(size_t alloc_size, int alignment)

    # no op if a_ptr is NULL
    void mkl_free(void *a_ptr)

    # Frees internal buffers (caches and workspaces) across all threads
    void mkl_free_buffers()

    # The number of max threads MKL is willing to use - this should *usually*
    # be set to the number of physical cores (so we can use it as a proxy for
    # number of physical cores)
    int mkl_get_max_threads()

    # Vector Math Functions and Types
    # 32-bit signed int unless linked against ILP64 interface - then its a
    # 64-bit signed int. Thus we should *always* be linked against the ILP64
    # interface or many of the vml funcs below will break under real scale.
    ctypedef long long MKL_INT

    # Control flags - we especially want VML_FTZDAZ_ON
    unsigned int vmlSetMode(const unsigned int flags)
    # More flag defs in "mkl_vml_defines.h"
    unsigned int VML_FTZDAZ_ON = 0x00280000
    unsigned int VML_DOUBLE_CONSISTENT = 0x00000020
    # Can only use *ONE* at a time
    unsigned int VML_LA = 0x00000001
    unsigned int VML_HA = 0x00000002
    unsigned int VML_EP = 0x00000003

    void vdInvSqrt(MKL_INT n, double *a, double *y)
    void vdMul(MKL_INT n, double *a, double *b, double *y)
    void vdSqr(MKL_INT n, double *a, double *y)
    void vdSqrt(MKL_INT n, double *a, double *y)
    void vdSub(MKL_INT n, double *a, double *b, double *y)
    void vdTanh(MKL_INT n, double* a, double *y)

    void vdDivI(MKL_INT n,  # y = a / b
                double* a, MKL_INT inca,
                double* b, MKL_INT incb,
                double* y, MKL_INT incy)
    void vdFmaxI(MKL_INT n,
                 double* a, MKL_INT inca,
                 double* b, MKL_INT incb,
                 double* y, MKL_INT incy)
    void vdMulI(MKL_INT n,
                double* a, MKL_INT inca,
                double* b, MKL_INT incb,
                double* y, MKL_INT incy)
    void vdSubI(MKL_INT n,  # y = a - b
                double* a, MKL_INT inca,
                double* b, MKL_INT incb,
                double* y, MKL_INT incy)


    # Vector Summary Statistics (VSLSS)
    ctypedef void* VSLSSTaskPtr
    MKL_INT VSL_SS_MATRIX_STORAGE_ROWS = 0x00010000
    MKL_INT VSL_SS_MATRIX_STORAGE_COLS = 0x00020000
    MKL_INT VSL_SS_ED_MEAN = 7
    MKL_INT VSL_SS_METHOD_FAST = 0x00000001
    MKL_INT VSL_SS_MEAN = 0x0000000000000001

    int vsldSSNewTask(VSLSSTaskPtr* task,
                      const MKL_INT* p,
                      const MKL_INT* n,
                      const MKL_INT* xstorage,
                      const double * x,
                      const double * w,
                      const MKL_INT* indices)
    int vsldSSEditTask(VSLSSTaskPtr task,
                       const MKL_INT par,
                       const double* par_addr)
    int vsldSSCompute(VSLSSTaskPtr task,
                      const MKL_INT estimates,
                      const MKL_INT method)
    int vslSSDeleteTask(VSLSSTaskPtr* task)


    # CBLAS API
    ctypedef enum CBLAS_LAYOUT:
        CblasRowMajor=101
        CblasColMajor=102

    ctypedef enum CBLAS_TRANSPOSE:
        CblasNoTrans=111
        CblasTrans=112
        CblasConjTrans=113

    ctypedef enum CBLAS_UPLO:
        CblasUpper=121  # uplo ='U'
        CblasLower=122  # uplo ='L'

    # Returns the index of the absolute maximum value element of x
    # Equivalent to np.argmax(np.abs(x)) for 1d `x`
    size_t cblas_idamax(const MKL_INT n,
                        const double *x,
                        const MKL_INT incx)

    # y := x
    void cblas_dcopy(const MKL_INT n,
                     const double *x,
                     const MKL_INT incx,
                     double *y,
                     const MKL_INT incy)

    # rval = xy (dot product)
    double cblas_ddot(const MKL_INT n,
                      const double *sx, const MKL_INT incx,
                      const double *sy, const MKL_INT incy)

    # Inplace scale vector x by scalar a: x = a*x
    void cblas_dscal(const MKL_INT n,
                     const double a,
                     double *x,
                     const MKL_INT incx)

    # y := (alpha * op(A) * x) + (beta * y)
    void cblas_dgemv(const CBLAS_LAYOUT Layout,
                     const CBLAS_TRANSPOSE trans,   # A or A.T?
                     const MKL_INT m,               # rows of op(A)
                     const MKL_INT n,               # cols of op(A)
                     const double alpha,
                     const double *a,
                     const MKL_INT lda,
                     const double *x,
                     const MKL_INT incx,            # *NOT* ZERO!
                     const double beta,             # Zero to overwrite y
                     double *y,
                     const MKL_INT incy)            # Also not zero

    # C := (alpha * op(A) * op(B)) + (beta * C); op(X) = X or X^T depending on `trans`
    void cblas_dgemm(const CBLAS_LAYOUT Layout,     # Row vs Col ordering
                     const CBLAS_TRANSPOSE transa,  # A or A.T ?
                     const CBLAS_TRANSPOSE transb,  # B or B.T ?
                     const MKL_INT m,               # rows of op(A) (and therefore C)
                     const MKL_INT n,               # cols of op(B) (and therefore C)
                     const MKL_INT k,               # inner dimension
                     const double alpha,            # scaling factor for AB
                     const double *a,               # matrix A
                     const MKL_INT lda,             # leading dim of A
                     const double *b,               # matrix B
                     const MKL_INT ldb,             # leading dim of B
                     const double beta,             # scaling factor of C
                     double *c,                     # matrix C (output)
                     const MKL_INT ldc)             # leading dim of C

    # C := (alpha * A * A^T) + (beta * C)
    void cblas_dsyrk(const CBLAS_LAYOUT Layout,     # Row vs Col Major
                     const CBLAS_UPLO uplo,         # U or L - which part to overwrite
                     const CBLAS_TRANSPOSE trans,   # if no trans AA^T, else A^TA
                     const MKL_INT n,               # the order of C (i.e. n = C is n by n)
                     const MKL_INT k,               # the other dimension of A
                     const double alpha,            # scales AA^T
                     const double *a,               # matrix A
                     const MKL_INT lda,             # leading dimension of A
                     const double beta,             # scales C (set to zero for no effect)
                     double *c,                     # matrix C - overwritten with output
                     const MKL_INT ldc)             # leading dimension of C

    # LAPACK Extended C interface
    # 0 Turns NaN checking off, any other int turns it on
    void LAPACKE_set_nancheck(int flag)
    int LAPACKE_get_nancheck()

    ctypedef MKL_INT lapack_int
    int LAPACK_ROW_MAJOR = 101
    int LAPACK_COL_MAJOR = 102

    lapack_int LAPACKE_dsyevr(int matrix_layout,
                              char jobz,
                              char range,
                              char uplo,
                              lapack_int n,
                              double* a,
                              lapack_int lda,
                              double vl,
                              double vu,
                              lapack_int il,
                              lapack_int iu,
                              double abstol,
                              lapack_int* m,
                              double* w,
                              double* z,
                              lapack_int ldz,
                              lapack_int* isuppz)
    # Use with b'S' as argument to dsyevr `abstol` parameter
    double LAPACKE_dlamch(char cmach)

    lapack_int LAPACKE_dsyevd(int matrix_layout,
                              char jobz,
                              char uplo,
                              lapack_int n,
                              double* a,
                              lapack_int lda,
                              double* w)

    # MKL BLAS-like extensions
    # From the docs: For threading to be active in mkl_?imatcopy, the pointer
    # AB must be aligned on the 64-byte boundary. This requirement can be met
    # by allocating AB with mkl_malloc.
    void mkl_dimatcopy(const char ordering,         # b'R' or b'C' for Row or Col major
                       const char trans,            # b'T' to transpose
                       size_t rows,                 # rows before transpose
                       size_t cols,                 # also before transpose
                       const double alpha,          # optional scale factor
                       double *AB,                  # input/output matrix
                       size_t lda,                  # lda/ldb both distance between
                       size_t ldb)                  # elements in adj. rows/cols

    # B := alpha*op(A) - "out of memory alpha transpose" copy
    # Note the type of the ordering and trans args: char, not enums
    void mkl_domatcopy(char ordering,
                       char trans,
                       size_t rows,
                       size_t cols,
                       const double alpha,
                       const double *A,
                       size_t lda,
                       double *B,
                       size_t ldb)
