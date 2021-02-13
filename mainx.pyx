cimport scipy.linalg.cython_lapack as lapack
from cpython cimport array
import array
from libc.stdlib cimport malloc, free

def func():
    cdef array.array a = array.array('f', [
        1.44, -9.96, -7.55,  8.34,  7.08, -5.45,
        -7.84, -0.28,  3.24,  8.09,  2.52, -5.70,
        -4.39, -3.24,  6.27,  5.28,  0.74, -1.19,
        4.53,  3.83, -6.64,  2.06, -2.47,  4.70])

    cdef array.array b = array.array('f', [
        8.58,  8.26,  8.48, -5.28,  5.72,  8.93,
        9.35, -4.43, -0.70, -0.26, -7.36, -2.52])

    cdef int m = 6, n = 4, nrhs = 2
    cdef int lda = m, ldb = m, lwork = min(m, n) + max(1,m,n,nrhs), info
    cdef float* work = <float*>malloc(lwork * sizeof(float))

    if not work:
        raise MemoryError()

    try:
        lapack.sgels("N", &m, &n, &nrhs, a.data.as_floats, &lda, b.data.as_floats, &ldb, work, &lwork, &info)

        print("INFO", info)

        if info == 0:
            for i in range(n * nrhs):
                print(b[i])
            print()
            print("LWORK", work[0])
    finally:
        free(work)
