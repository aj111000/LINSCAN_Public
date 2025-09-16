# Runtime comparison
from time import time

import jax
import matplotlib.pyplot as plt
from jax import numpy as jnp
from jax import random
from scipy.spatial import KDTree

# from scipy.linalg import sqrtm as sqrtm
from linscan import embed_dataset, vmapped_jax_dist


@jax.jit
def euclidean_distance(dataset):
    @jax.vmap
    def calc_distances(point):
        return jnp.linalg.norm(dataset - point[None, :], axis=-1)

    return calc_distances(dataset)


def linscan_distance(dataset):
    kd = KDTree(dataset)

    near_neighbors = dataset[kd.query(x=dataset, k=20)[1]]

    embeddings = embed_dataset(near_neighbors)

    @jax.vmap
    def calc_distances(embedding):
        return vmapped_jax_dist(embedding, embeddings)

    dists = calc_distances(embeddings)
    return dists


num_trials = 100
ed_times = []
ld_times = []
num_list = [1000, 2000, 4000, 8000, 16000]

for num_points in num_list:
    data = random.uniform(random.key(0), shape=[num_points, 2])

    # Have to run each function at least once before timing to compile the JAX portions
    ed = euclidean_distance(data).block_until_ready()
    start = time()

    for _ in range(num_trials):
        ed = euclidean_distance(data).block_until_ready()

    end = time()

    elapsed = end - start

    ed_times.append(elapsed / num_trials)

    ld = linscan_distance(data).block_until_ready()

    start = time()

    for _ in range(num_trials):
        ld = linscan_distance(data).block_until_ready()

    end = time()

    elapsed = end - start

    ld_times.append(elapsed / num_trials)

plt.loglog(num_list, ed_times, label="Euclidean")
plt.loglog(num_list, ld_times, label="LINSCAN")
plt.legend()
plt.xlabel("Number of points")
plt.ylabel("Runtime")
plt.show()
