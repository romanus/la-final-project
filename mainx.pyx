#cython: language_level=3

# python-related
import sys
import numpy as np
import scipy.linalg

# cython-related
cimport scipy.linalg.cython_lapack as lapack
# from cpython cimport array
# import array
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy

# def example_func():
#     cdef array.array a = array.array('f', [
#         1.44, -9.96, -7.55,  8.34,  7.08, -5.45,
#         -7.84, -0.28,  3.24,  8.09,  2.52, -5.70,
#         -4.39, -3.24,  6.27,  5.28,  0.74, -1.19,
#         4.53,  3.83, -6.64,  2.06, -2.47,  4.70])
#
#     cdef array.array b = array.array('f', [
#         8.58,  8.26,  8.48, -5.28,  5.72,  8.93,
#         9.35, -4.43, -0.70, -0.26, -7.36, -2.52])
#
#     cdef int m = 6, n = 4, nrhs = 2
#     cdef int lda = m, ldb = m, lwork = min(m, n) + max(1,m,n,nrhs), info
#     cdef float* work = <float*>malloc(lwork * sizeof(float))
#
#     if not work:
#         raise MemoryError()
#
#     try:
#         lapack.sgels("N", &m, &n, &nrhs, a.data.as_floats, &lda, b.data.as_floats, &ldb, work, &lwork, &info)
#
#         print("INFO", info)
#
#         if info == 0:
#             for i in range(n * nrhs):
#                 print(b[i])
#             print()
#             print("LWORK", work[0])
#     finally:
#         free(work)

def solve_lss(int alg_id, long long A, long long b, int m, int n, int nrhs, int lda, int ldb):
    cdef unsigned long long A_size = m * n * sizeof(double)
    cdef unsigned long long b_size = m * nrhs * sizeof(double)
    cdef double* A_ptr = <double*>malloc(A_size)
    cdef double* b_ptr = <double*>malloc(b_size)
    cdef double* work
    cdef int lwork, info, mn = min(m,n)

    memcpy(A_ptr, <void*>A, A_size)
    memcpy(b_ptr, <void*>b, b_size)

    if alg_id == 0: # dgels
        lwork = max(1, mn + max(mn, nrhs))
        work = <double*>malloc(lwork * sizeof(double))
        lapack.dgels("N", &m, &n, &nrhs, A_ptr, &lda, b_ptr, &ldb, work, &lwork, &info)
        free(work)
    else:
        pass

    free(A_ptr)
    free(b_ptr)

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
        diag_vals = np.linspace(sys.float_info.epsilon, 1, mn)
    elif type_id == 1:
        # Singular values distributed geometrically from eps up to 1
        diag_vals = np.logspace(sys.float_info.epsilon, 1, mn, base=2) / 2
    elif type_id == 2:
        # 1 singular value at 1 and the other clustered at eps
        diag_vals = sys.float_info.epsilon * np.ones((mn))
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