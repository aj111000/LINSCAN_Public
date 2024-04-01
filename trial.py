from linscan import linscan
from sklearn.metrics import adjusted_rand_score


def trial(args):
    datasets, true_labels, eps, min_pts, threshold, ecc_pts, xi = args
    scores = []

    for dataset, true_label in zip(datasets, true_labels):
        gen_label = linscan(dataset, eps, min_pts, ecc_pts, threshold, xi)
        scores.append(adjusted_rand_score(true_label, gen_label))

    return [[eps, min_pts, threshold, ecc_pts, xi], scores]
