import numpy as np
from numpy.linalg import norm

# from scipy.linalg import sqrtm as sqrtm

import ctypes


from miscelaneous import unpack_embedding


def gen_ref_kl_dist():
    # Do not use this implementation, as it is slow. It only exists to give a reference for what the C implementation does.
    def kl_dist(x, y):
        p1, cov1, inv1, inv_sqrt1 = unpack_embedding(x)
        p2, cov2, inv2, inv_sqrt2 = unpack_embedding(y)

        dist = (
            1 / 2 * norm(inv_sqrt2 @ cov1 @ inv_sqrt2 - np.eye(2), ord="fro")
            + 1 / 2 * norm(inv_sqrt1 @ cov2 @ inv_sqrt1 - np.eye(2), ord="fro")
            + 1 / np.sqrt(2) * np.sqrt((p1 - p2).transpose() @ inv1 @ (p1 - p2))
            + 1 / np.sqrt(2) * np.sqrt((p1 - p2).transpose() @ inv2 @ (p1 - p2))
        )

        return np.max([dist, 0])

    return kl_dist


def gen_c_kl_dist():
    array = ctypes.c_double * 11
    def convert(A, B):
        return array(*A.tolist()), array(*B.tolist())

    kl_dist = ctypes.CDLL("./linscan_c.so").kl_dist
    kl_dist.restype = ctypes.c_double

    def dist_func(x, y):
        return kl_dist(*convert(x, y))
    return dist_func