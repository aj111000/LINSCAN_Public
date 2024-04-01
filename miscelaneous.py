import numpy as np
import numpy.random as random


def normalize_datasets(datasets):
    def x_filt(x, x_range = [-1, 1]):
        return x_range[0] <= x <= x_range[1]
    def y_filt(y, y_range = [-1,1]):
        return y_range[0] <= y <= y_range[1]
    def filt(pt):
        return x_filt(pt[0]) and y_filt(pt[1])
    

    for i in range(len(datasets)):
        datasets[i] -= datasets[i].mean(0)

        datasets[i] /= np.max(np.abs(datasets[i]))

        datasets[i] = np.array(list(filter(filt, datasets[i].tolist())))
        datasets[i] /= np.max(np.abs(datasets[i]), axis=0)
    return datasets


def param_generator(
    datasets, labels, eps_range, min_pts_range, threshold_range, ecc_pts_range, xi_range, num_trials
):
    def gen_rand(range):
        return random.uniform(low=range[0], high=range[1])

    for i in range(num_trials):
        eps = gen_rand(eps_range)
        min_pts = int(gen_rand(min_pts_range))
        threshold = gen_rand(threshold_range)
        ecc_pts = int(gen_rand(ecc_pts_range))
        xi = gen_rand(xi_range)
        yield datasets, labels, eps, min_pts, threshold, ecc_pts, xi


def pack_mat(mat):
    return [mat[0, 0], mat[0, 1], mat[1, 1]]


def unpack_embedding(x):
    p = np.array([x[0], x[1]])
    cov = np.array([[x[2], x[3]], [x[3], x[4]]])
    inv = np.array([[x[5], x[6]], [x[6], x[7]]])
    invsqrt = np.array([[x[8], x[9]], [x[9], x[10]]])
    return p, cov, inv, invsqrt


def sqrtm(mat):
    s = np.sqrt(mat[0, 0] * mat[1, 1] - mat[0, 1] * mat[1, 0])
    t = np.sqrt(mat[0, 0] + mat[1, 1] + 2 * s)
    return 1 / t * (mat + s * np.eye(2))