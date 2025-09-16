from data_generation import gen_data
import numpy as np

import jax
import jax.numpy as jnp
import numpy.random as random

import matplotlib.pyplot as plt

from adcn import adcn


from sklearn.cluster import OPTICS


# from adcn import adcn
from linscan import linscan


if __name__ == "__main__":

    data, labels = gen_data(
        lin_clusts=20,
        lin_num=200,
        lin_R=2,
        int_clusts=20,
        int_num=200,
        int_R=2,
        iso_clusts=9,
        iso_num=100,
        iso_R=10,
        noise_num=500,
        x_min=-150,
        x_max=150,
        y_min=-150,
        y_max=150,
    )

    threshold = 0.5
    linscan_eps = np.inf
    linscan_min_pts = 100
    linscan_ecc_pts = 20

    optics_min_pts = linscan_min_pts

    adcn_eps = 3.0
    adcn_min_pts = linscan_min_pts

    linscan_labels = linscan(
        data,
        min_pts=linscan_min_pts,
        ecc_pts=linscan_ecc_pts,
        eps=linscan_eps,
        threshold=threshold,
    )

    optics_labels = jnp.array(
        OPTICS(
            min_samples=optics_min_pts,
            cluster_method="xi",
            max_eps=linscan_eps,
        )
        .fit(data)
        .labels_
    )

    adcn_labels = jnp.array(adcn(data, adcn_eps, adcn_min_pts, threshold))

    max_val = max(linscan_labels.max(), optics_labels.max(), adcn_labels.max())

    shuffler = random.permutation(jnp.arange(max_val + 1))

    def shuffle(input_list):
        return [shuffler[ind] if ind != -1 else ind for ind in input_list]

    plt.scatter(data[:, 0], data[:, 1], c=shuffle(linscan_labels), marker=",", s=1)
    plt.show()

    plt.scatter(data[:, 0], data[:, 1], c=shuffle(optics_labels), marker=",", s=1)
    plt.show()

    plt.scatter(data[:, 0], data[:, 1], c=shuffle(adcn_labels), marker=",", s=1)
    plt.show()

    linscan_filt = linscan_labels > -1
    optics_filt = optics_labels > -1
    adcn_filt = adcn_labels > -1

    plt.scatter(
        data[linscan_filt, 0],
        data[linscan_filt, 1],
        c=shuffle(linscan_labels[linscan_filt]),
        marker=",",
        s=1,
    )
    plt.show()

    plt.scatter(
        data[optics_filt, 0],
        data[optics_filt, 1],
        c=shuffle(optics_labels[optics_filt]),
        marker=",",
        s=1,
    )
    plt.show()

    plt.scatter(
        data[adcn_filt, 0],
        data[adcn_filt, 1],
        c=shuffle(adcn_labels[adcn_filt]),
        marker=",",
        s=1,
    )
    plt.show()
