# python-related
from timeit import timeit
import numpy as np
from tabulate import tabulate

# cython-related
import pyximport; pyximport.install()
import mainx

seeds = [393, 950] # random seeds
# seeds = [393, 950, 493, 923, 859, 871, 687, 996, 818, 867] # random seeds
N = 100 # repeats number

if __name__ == '__main__':

    # dimensions = np.array([50, 100, 200, 300, 400])
    dimensions = np.array([100, 200])
    algorithms = np.array(['DGELS', 'DGELSY', 'DGELSS', 'DGELSD'])
    # types = np.array([
    #     'Singular values distributed arithmetically from eps up to 1',
    #     'Singular values distributed geometrically from eps up to 1',
    #     '1 singular value at 1 and the other clustered at eps'])
    types = np.array([
        'Singular values distributed arithmetically from eps up to 1'])

    dimensions_annotated = np.insert(dimensions.astype(np.str), 0, ['Function \ n'])

    # allocate the timings array
    results = np.zeros((len(types), len(algorithms), len(dimensions)), dtype=np.float)

    for type_id, type_name in enumerate(types):
        print("Type", type_id+1, "-", type_name)

        # benchmark functions
        for seed in seeds:
            for dim_id, dim in enumerate(dimensions):
                m, n, nrhs, lda, ldb = dim, dim, 1, dim, dim
                A, b = mainx.generate_matrices(seed, type_id, m, n, nrhs)
                for alg_id, _ in enumerate(algorithms):
                    if alg_id == 0 and type_id == 2:
                        # we don't benchmark dgels on rank-deficient matrices
                        continue
                    results[type_id][alg_id][dim_id] += round(timeit("mainx.solve_lss(alg_id, A, b, m, n, nrhs, lda, ldb)", globals=globals(), number=N), 3)
                mainx.free_matrices(A, b)

        # print results
        results_annotated = np.empty((results.shape[1], results.shape[2]+1), dtype=np.object)
        results_annotated[:,1:] = results[type_id]
        results_annotated[:,0] = algorithms
        print(tabulate(results_annotated, headers=dimensions_annotated, tablefmt='latex'))
        print()
