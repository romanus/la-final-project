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

def solve_lss(int alg_id, long long A, long long b, int m, int n, int nrhs, int lda, int ldb, workspace):
    cdef unsigned long long A_size
    cdef unsigned long long b_size
    cdef double* A_ptr
    cdef double* b_ptr
    cdef double* work
    cdef double* s
    cdef int lwork, info, rank, liwork
    cdef int* jpvt
    cdef double rcond
    cdef int* iwork

    rcond = DBL_EPSILON

    A_size = m * n * sizeof(double)
    b_size = m * nrhs * sizeof(double)

    estimated_parameters, allocated_matrices = workspace

    A_ptr = <double*><long long>allocated_matrices[0]
    b_ptr = <double*><long long>allocated_matrices[1]

    # since the input buffers are overwritten with the algorithm results,
    # we need to copy them; this is not expensive at all
    # (w.r.t. the overall algorithm complexity)
    memcpy(A_ptr, <void*>A, A_size)
    memcpy(b_ptr, <void*>b, b_size)

    if alg_id == 0: # dgels
        lwork = estimated_parameters[0]
        work = <double*><long long>allocated_matrices[2]
        lapack.dgels("N", &m, &n, &nrhs, A_ptr, &lda, b_ptr, &ldb, work, &lwork, &info)
    elif alg_id == 1: # dgelsy
        lwork = estimated_parameters[0]
        work = <double*><long long>allocated_matrices[2]
        jpvt = <int*><long long>allocated_matrices[3]
        memset(jpvt, 0, n * sizeof(int))
        lapack.dgelsy(&m, &n, &nrhs, A_ptr, &lda, b_ptr, &ldb, jpvt, &rcond, &rank, work, &lwork, &info)
    elif alg_id == 2: # dgelss
        lwork = estimated_parameters[0]
        work = <double*><long long>allocated_matrices[2]
        s = <double*><long long>allocated_matrices[3]
        lapack.dgelss(&m, &n, &nrhs, A_ptr, &lda, b_ptr, &ldb, s, &rcond, &rank, work, &lwork, &info)
    elif alg_id == 3: # dgelsd
        lwork = estimated_parameters[0]
        work = <double*><long long>allocated_matrices[2]
        s = <double*><long long>allocated_matrices[3]
        liwork = estimated_parameters[1]
        iwork = <int*><long long>allocated_matrices[4]
        lapack.dgelsd(&m, &n, &nrhs, A_ptr, &lda, b_ptr, &ldb, s, &rcond, &rank, work, &lwork, iwork, &info)
    else:
        raise Exception("unknown alg_id")

    return info, <int>work[0]

def estimate_workspace(int alg_id, long long A, long long b, int m, int n, int nrhs):
    cdef unsigned long long A_size
    cdef unsigned long long b_size
    cdef double* A_ptr
    cdef double* b_ptr
    cdef int lwork, mn
    cdef double* work
    cdef int* jpvt
    cdef double* s
    cdef int* iwork
    cdef int liwork, smlsiz, nlvl
    cdef int n4, ispec

    mn = min(m,n)

    A_size = m * n * sizeof(double)
    b_size = m * nrhs * sizeof(double)

    A_ptr = <double*>malloc(A_size)
    b_ptr = <double*>malloc(b_size)

    estimated_parameters = []
    allocated_matrices = []

    if alg_id == 0: # dgels
        lwork = max(1, mn + max(mn, nrhs))
        work = <double*>malloc(lwork * sizeof(double))
        estimated_parameters = [lwork]
        allocated_matrices = [<long long>A_ptr, <long long>b_ptr, <long long>work]
    elif alg_id == 1: # dgelsy
        lwork = max(mn+3*n+1, 2*mn+nrhs)
        work = <double*>malloc(lwork * sizeof(double))
        jpvt = <int*> malloc(n * sizeof(int))
        estimated_parameters = [lwork]
        allocated_matrices = [<long long>A_ptr, <long long>b_ptr, <long long>work, <long long>jpvt]
    elif alg_id == 2: # dgelss
        lwork = 3*mn + max(2*mn, max(m,n), nrhs)
        work = <double*>malloc(lwork * sizeof(double))
        s = <double*>malloc(mn * sizeof(double))
        estimated_parameters = [lwork]
        allocated_matrices = [<long long>A_ptr, <long long>b_ptr, <long long>work, <long long>s]
    elif alg_id == 3: # dgelsd
        # ilaenv is not included in scipy, so we cannot invoke it :(
        # n4 = -1 # unused param of ilaenv
        # ispec = 9 # maximum size of the subproblems at the bottom of the computation tree in the divide-and-conquer algorithm
        # smlsiz = lapack.ilaenv(&ispec, "dgelsd", "", &m, &n, &nrhs, &n4)
        # http://www.netlib.org/lapack/explore-html/d8/d6d/_s_r_c_2ilaenv_8f_source.html, line 684 contains the hardcoded value:
        smlsiz = 25
        nlvl = max(0, <int>log2(mn/(smlsiz+1.)) + 1)
        lwork = 12*n + 2*n*smlsiz + 8*n*nlvl + n*nrhs + (smlsiz+1)**2
        work = <double*>malloc(lwork * sizeof(double))
        s = <double*>malloc(mn * sizeof(double))
        liwork = max(1, 3 * mn * nlvl + 11*mn)
        iwork = <int*>malloc(liwork * sizeof(int))
        estimated_parameters = [lwork, liwork]
        allocated_matrices = [<long long>A_ptr, <long long>b_ptr, <long long>work, <long long>s, <long long>iwork]
    else:
        raise Exception("unknown alg_id")

    return estimated_parameters, allocated_matrices

def free_workspace(workspace):
    cdef long long ptr

    estimated_parameters, allocated_matrices = workspace
    for ptr in allocated_matrices:
        free(<void*>ptr)

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