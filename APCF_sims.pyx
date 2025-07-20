#!python
#cython: language_level=3

cimport numpy as np  # noqa
import numpy as np
from libc.math cimport sqrt

def adj_cosine(int n_x, yr, int min_support, means):
    
    # sum (r_xy * r_x'y) for common ys
    cdef double [:, ::1] prods = np.zeros((n_x, n_x), np.double)
    # number of common ys
    cdef long [:, ::1] freq = np.zeros((n_x, n_x), np.int_)
    # sum (r_xy ^ 2) for common ys
    cdef double [:, ::1] sqi = np.zeros((n_x, n_x), np.double)
    # sum (r_x'y ^ 2) for common ys
    cdef double [:, ::1] sqj = np.zeros((n_x, n_x), np.double)
    # the similarity matrix
    cdef double [:, ::1] sim = np.zeros((n_x, n_x), np.double)

    cdef int xi, xj, y
    cdef double ri, rj
    cdef int min_sprt = min_support

    for y, y_ratings in yr.items():
        mean_r = means[y]
        for xi, ri in y_ratings:
            for xj, rj in y_ratings:
                freq[xi, xj] += 1
                prods[xi, xj] += (ri - mean_r) * (rj - mean_r)
                sqi[xi, xj] += (ri - mean_r)**2
                sqj[xi, xj] += (rj - mean_r)**2

    for xi in range(n_x):
        sim[xi, xi] = 1
        for xj in range(xi + 1, n_x):
            if freq[xi, xj] < min_sprt:
                sim[xi, xj] = 0
            else:
                denum = sqrt(sqi[xi, xj] * sqj[xi, xj])
                if denum == 0:
                    sim[xi, xj] = 1
                else:
                    sim[xi, xj] = prods[xi, xj] / denum

            sim[xj, xi] = sim[xi, xj]

    return np.asarray(sim)
    
def sim_user_pat(int n_u, patterns, ir):

    n_pats = len(patterns)
     
    cdef double [:, ::1] sim = np.zeros((n_pats, n_u), np.double)

    cdef int item, u, i

    for i, p in enumerate(patterns):
          
        p_size = len(p)

        for item in p:
            for u,_ in ir[item]:
                sim[i, u] += 1.0/p_size
    
    return np.asarray(sim)
