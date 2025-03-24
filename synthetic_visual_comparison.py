from data_generation import gen_data
import numpy as np

import jax
import jax.numpy as jnp
import numpy.random as random




# import sklearn as skl
# from sklearn import metrics


# from adcn import adcn
from linscan import linscan


if __name__ == "__main__":

    # Number of train and test datasets
    data, labels = gen_data(
            lin_clusts=20,
            lin_num=200,
            lin_R=2,
            int_clusts=20,
            int_num=200,
            int_R=2,
            iso_clusts=9,
            iso_num=200,
            iso_R=10,
            noise_num=500,
            x_min=-50,
            x_max=50,
            y_min=-50,
            y_max=50,)

    threshold = .2
    linscan_eps = .2
    linscan_min_pts = 20
    linscan_ecc_pts = 30


    adcn_eps = linscan_eps
    adcn_min_pts = linscan_min_pts

    linscan_labels = linscan(data, min_pts=linscan_min_pts, ecc_pts=linscan_ecc_pts, eps=linscan_eps, threshold=threshold)

    optics_labels = ...

    # adcn_labels = adcn(data, adcn_eps, adcn_min_pts, threshold)
