from scipy.spatial import KDTree

from functools import partial


import jax
from jax import numpy as jnp

from sklearn import OPTICS

from distances import jax_kl_dist


vmapped_jax_dist = jax.vmap(jax_kl_dist, in_axes=(None, 0), out_axes=0)


@jax.vmap
@jax.jit
def embed_dataset(near_neighbors):

    cov = jnp.cov(near_neighbors, rowvar=False)
    cov /= jnp.linalg.norm(cov)
    mean = near_neighbors.mean(0)
    inv = jnp.linalg.inv(cov)
    inv_sqrt = jnp.linalg.cholesky(inv)

    return mean, cov, inv, inv_sqrt


def kl_jax_scan(dataset, min_pts, ecc_pts, eps=jnp.inf, xi=0.05):
    kd = KDTree(dataset)

    near_neighbors = dataset[kd.query(x=dataset, k=ecc_pts)[1]]

    embeddings = embed_dataset(near_neighbors)

    @partial(jax.vmap, in_axes = [0, None])
    def calc_distances(embedding, eps=jnp.inf):
        too_far = jnp.norm(embedding[0][None,:] - embeddings[0][:,:]) > eps
        return jnp.where(too_far, jnp.inf, vmapped_jax_dist(embedding, embeddings))

    dists = calc_distances(embeddings, eps=eps)
    return jnp.array(
        OPTICS(
            min_samples=min_pts,
            metric="precomputed",
            cluster_method="xi",
            xi=xi,
        )
        .fit(dists)
        .labels_
    )


def linscan(dataset, eps, min_pts, ecc_pts, threshold, xi=0.05):
    typelist = kl_jax_scan(dataset, eps, min_pts, ecc_pts, xi)

    for cat in range(max(typelist)):
        cat_inds = typelist == cat
        temp = dataset[cat_inds, :]
        if temp.size == 0:
            continue
        eigenvalues, _ = jnp.linalg.eigh(jnp.cov(temp, rowvar=False))

        if min(eigenvalues) / max(eigenvalues) > threshold:
            typelist = jnp.where(cat_inds, -1, typelist)

    return typelist


if __name__ == "__main__":
    # Runtime comparison
    from time import time
    from jax import random
    import matplotlib.pyplot as plt

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

        @jax.vmap()
        def calc_distances(embedding, eps=1e-4):
            too_far = jnp.linalg.norm(embedding[0][None,:] - embeddings[0][:,:]) > eps
            return jnp.where(too_far, jnp.inf, vmapped_jax_dist(embedding, embeddings))

        dists = calc_distances(embeddings)
        return dists

    num_trials = 100
    ed_times = []
    ld_times = []
    num_list = [1000, 2000, 4000]

    for num_points in num_list:
        data = random.uniform(random.key(0), shape=[num_points, 2])

        start = time()

        for _ in range(num_trials):
            ed = euclidean_distance(data).block_until_ready()

        end = time()

        elapsed = end - start

        ed_times.append(elapsed / num_trials)

        start = time()

        for _ in range(num_trials):
            ld = linscan_distance(data).block_until_ready()

        end = time()

        elapsed = end - start

        ld_times.append(elapsed / num_trials)

    plt.loglog(num_list, ed_times, label="Euclidean")
    plt.loglog(num_list, ld_times, label="Linscan")
    plt.legend()
    plt.show()
