# python-related
from timeit import timeit
import numpy as np

# cython-related
import pyximport; pyximport.install()
import mainx

seed = 42 # random seed
N = 100 # repeats number

if __name__ == '__main__':

    # dimensions = [50, 100, 200, 300, 400]
    dimensions = [50, 100]
    algorithms = ['dgels', 'dgelsy', 'dgelss', 'dgelsd']
    types = ['type1', 'type2', 'type3']

    results = np.zeros((len(types), len(algorithms), len(dimensions)), dtype=np.float)

    # benchmark functions
    for dim_id, dim in enumerate(dimensions):
        m, n, nrhs, lda, ldb = dim, dim, 1, dim, dim
        for type_id, _ in enumerate(types):
            A, b = mainx.generate_matrices(seed, type_id, m, n, nrhs)
            try:
                for alg_id, _ in enumerate(algorithms):
                    results[type_id][alg_id][dim_id] = timeit("mainx.solve_lss(alg_id, A, b, m, n, nrhs, lda, ldb)", globals=globals(), number=N) / N
            finally:
                mainx.free_matrices(A, b)

    # print results
    for type_id, type_name in enumerate(types):
        print("type:", type_name)
        type_results = results[type_id]
        for alg_id, alg_name in enumerate(algorithms):
            alg_results = type_results[alg_id]
            print(alg_name, end=' ')
            for dim_id, _ in enumerate(dimensions):
                print(alg_results[dim_id], end=' ')
            print()
