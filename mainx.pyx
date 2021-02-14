#cython: language_level=3

# python-related
import sys
import numpy as np
import scipy.linalg

# cython-related
cimport scipy.linalg.cython_lapack as lapack
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy, memset

DBL_EPSILON = sys.float_info.epsilon

def solve_lss(int alg_id, long long A, long long b, int m, int n, int nrhs, int lda, int ldb):
    cdef unsigned long long A_size = m * n * sizeof(double)
    cdef unsigned long long b_size = m * nrhs * sizeof(double)
    cdef double* A_ptr = <double*>malloc(A_size)
    cdef double* b_ptr = <double*>malloc(b_size)
    cdef double* work
    cdef int lwork, info, mn = min(m,n), optimal_lwork, rank
    cdef int* jpvt
    cdef double rcond = DBL_EPSILON

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
        free(work)
        free(jpvt)
    else:
        pass

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
        diag_vals = np.logspace(DBL_EPSILON, 1, mn, base=2) / 2
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