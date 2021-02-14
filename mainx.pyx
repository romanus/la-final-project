#cython: language_level=3

# python-related
import sys
import numpy as np
import scipy.linalg

# cython-related
cimport scipy.linalg.cython_lapack as lapack
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy, memset
from libc.math cimport log2

DBL_EPSILON = sys.float_info.epsilon

def solve_lss(int alg_id, long long A, long long b, int m, int n, int nrhs, int lda, int ldb):
    cdef unsigned long long A_size = m * n * sizeof(double)
    cdef unsigned long long b_size = m * nrhs * sizeof(double)
    cdef double* A_ptr = <double*>malloc(A_size)
    cdef double* b_ptr = <double*>malloc(b_size)
    cdef double* work
    cdef double* s
    cdef int lwork, info, mn = min(m,n), optimal_lwork, rank, liwork, smlsiz, nlvl
    cdef int* jpvt
    cdef double rcond = DBL_EPSILON
    cdef int* iwork

    # since the input buffers are overwritten with the algorithm results,
    # we need to copy them; this is not expensive at all
    # (w.r.t. the overall algorithm complexity)
    memcpy(A_ptr, <void*>A, A_size)
    memcpy(b_ptr, <void*>b, b_size)

    if alg_id == 0: # dgels
        lwork = max(1, mn + max(mn, nrhs))
        work = <double*>malloc(lwork * sizeof(double))
        lapack.dgels("N", &m, &n, &nrhs, A_ptr, &lda, b_ptr, &ldb, work, &lwork, &info)
        optimal_lwork = <int>work[0]
        free(work)
    elif alg_id == 1: # dgelsy
        jpvt = <int*> malloc(n * sizeof(int))
        memset(jpvt, 0, n * sizeof(int))
        lwork = max(mn+3*n+1, 2*mn+nrhs)
        work = <double*>malloc(lwork * sizeof(double))
        lapack.dgelsy(&m, &n, &nrhs, A_ptr, &lda, b_ptr, &ldb, jpvt, &rcond, &rank, work, &lwork, &info)
        optimal_lwork = <int>work[0]
        free(work)
        free(jpvt)
    elif alg_id == 2: # dgelss
        s = <double*>malloc(mn * sizeof(double))
        lwork = 3*mn + max(2*mn, max(m,n), nrhs)
        work = <double*>malloc(lwork * sizeof(double))
        lapack.dgelss(&m, &n, &nrhs, A_ptr, &lda, b_ptr, &ldb, s, &rcond, &rank, work, &lwork, &info)
        optimal_lwork = <int>work[0]
        free(work)
        free(s)
    elif alg_id == 3: # dgelsd
        s = <double*>malloc(mn * sizeof(double))
        smlsiz = 30
        nlvl = max(0, <int>log2(mn/(smlsiz+1.)) + 1)
        lwork = 12*n + 2*n*smlsiz + 8*n*nlvl + n*nrhs + (smlsiz+1)**2
        liwork = max(1, 3 * mn * nlvl + 11*mn)
        work = <double*>malloc(lwork * sizeof(double))
        iwork = <int*>malloc(liwork * sizeof(int))
        lapack.dgelsd(&m, &n, &nrhs, A_ptr, &lda, b_ptr, &ldb, s, &rcond, &rank, work, &lwork, iwork, &info)
        optimal_lwork = <int>work[0]
        free(iwork)
        free(work)
        free(s)
    else:
        raise Exception("unknown alg_id")

    free(A_ptr)
    free(b_ptr)

    return info, optimal_lwork

def generate_matrices(int seed, int type_id, int m, int n, int nrhs):
    cdef double* A = <double*>malloc(m * n * sizeof(double))
    cdef double* b = <double*>malloc(m * nrhs * sizeof(double))

    if not A or not b:
        raise MemoryError()

    np.random.seed(seed)

    dtype = np.double

    U = scipy.linalg.orth(np.random.randn(m, m))
    S = np.zeros((m,n))
    VT = scipy.linalg.orth(np.random.randn(n, n))

    bc = np.random.randn(m, nrhs)

    mn = min(m,n)

    if type_id == 0:
        # Singular values distributed arithmetically from eps up to 1
        diag_vals = np.linspace(DBL_EPSILON, 1, mn)
    elif type_id == 1:
        # Singular values distributed geometrically from eps up to 1
        diag_vals = np.logspace(DBL_EPSILON, 1, mn, base=2)-1
    elif type_id == 2:
        # 1 singular value at 1 and the other clustered at eps
        diag_vals = DBL_EPSILON * np.ones((mn))
        diag_vals[0] = 1
    else:
        raise Exception("unknown type_id")

    S[:mn, :mn] = np.diag(diag_vals)

    Ac = U @ S @ VT

    bf = bc.ravel(order='F')
    Af = Ac.ravel(order='F')

    for i in range(m*n):
        A[i] = Af[i]

    for i in range(m*nrhs):
        b[i] = bf[i]

    return <long long>A, <long long>b

def free_matrices(long long A, long long b):
    free(<void*>A)
    free(<void*>b)