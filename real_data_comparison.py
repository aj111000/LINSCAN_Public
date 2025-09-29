import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as random
from xlearn.cluster import OPTICS

from adcn import adcn
from data_generation import load_real_data

# from adcn import adcn
from linscan import linscan

if __name__ == "__main__":

    data = load_real_data(0, 65, 110, 120)
    print("Number of points: ", data.shape[0])
    fig1 = plt.figure()
    ax = fig1.add_subplot(111)
    ax.set_aspect("equal", adjustable="box")

    plt.scatter(data[:, 0], data[:, 1], marker=".", s=1)
    plt.show()

    threshold = 1.0
    linscan_eps = np.inf
    linscan_min_pts = 100
    linscan_ecc_pts = 30

    optics_min_pts = linscan_min_pts

    adcn_eps = 4.0
    adcn_min_pts = linscan_min_pts

    # linscan_labels = linscan(
    #     data,
    #     min_pts=linscan_min_pts,
    #     ecc_pts=linscan_ecc_pts,
    #     eps=linscan_eps,
    #     threshold=threshold,
    #     xi=0.01,
    # )

    optics_labels = jnp.array(
        OPTICS(
            min_samples=optics_min_pts,
            cluster_method="xi",
            max_eps=linscan_eps,
            xi=0.01,
        )
        .fit(data)
        .labels_
    )
    # optics_labels = linscan_labels
    linscan_labels = optics_labels

    # adcn_labels = jnp.array(adcn(data, adcn_eps, adcn_min_pts, threshold))

    adcn_labels = linscan_labels

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
