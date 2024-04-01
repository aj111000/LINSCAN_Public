import numpy as np
from scipy.spatial import KDTree

# from scipy.linalg import sqrtm as sqrtm

from sklearn.cluster import OPTICS


from distances import gen_c_kl_dist
from miscelaneous import pack_mat, sqrtm



def kl_embed_scan(dataset, eps, min_pts, ecc_pts, xi=0.05):
    kd = KDTree(dataset)

    embeddings = []
    for p in range(len(dataset)):
        cluster = kd.query(x=dataset[p], k=ecc_pts)[1].tolist()
        cov = np.cov(np.array([dataset[k] for k in cluster]), rowvar=False)
        cov /= max(np.linalg.eig(cov)[0])
        mean = np.mean(np.array([dataset[k] for k in cluster]), axis=0)
        inv = (
            1
            / (cov[0, 0] * cov[1, 1] - cov[0, 1] * cov[0, 1])
            * np.array([[cov[1, 1], -cov[0, 1]], [-cov[0, 1], cov[0, 0]]])
        )
        inv_sqrt = sqrtm(inv)

        embeddings.append(
            np.concatenate([mean, pack_mat(cov), pack_mat(inv), pack_mat(inv_sqrt)])
        )
    embeddings = np.array(embeddings)

    return OPTICS(
        min_samples=min_pts,
        metric=gen_c_kl_dist(),
        cluster_method="xi",
        xi=xi,
    ).fit(embeddings).labels_


def linscan(dataset, eps, min_pts, ecc_pts, threshold, xi):
    typelist = kl_embed_scan(dataset, eps, min_pts, ecc_pts, xi)

    for cat in range(max(typelist)):
        temp = np.array(
            [dataset[i, :] for i in range(len(dataset)) if typelist[i] == cat]
        )
        if temp.size == 0:
            continue
        eigenvalues, eigenvectors = np.linalg.eig(np.cov(temp, rowvar=False))

        if min(eigenvalues) / max(eigenvalues) > threshold:
            typelist = list(map(lambda x: -1 if x == cat else x, typelist))

    return typelist
