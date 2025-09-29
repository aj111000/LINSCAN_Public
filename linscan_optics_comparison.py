import datetime
import time

import jax
import jax.numpy as jnp
import numpy as np
import numpy.random as random
import sklearn as skl
from sklearn import metrics

from data_generation import gen_data
from miscelaneous import normalize_datasets, param_generator
from trial import trial

if __name__ == "__main__":
    st = time.time()
    # Number of train and test datasets
    N = 100
    M = 400

    num_trials = 10000

    # Generate Samples
    temp = [gen_data(lin_clusts=10, iso_clusts=5, int_clusts=10) for i in range(N)]
    train_datasets = [jnp.array(item[0]) for item in temp]
    train_labels = [jnp.array(item[1]) for item in temp]

    temp = [gen_data(lin_clusts=10, iso_clusts=5, int_clusts=10) for i in range(M)]
    test_datasets = [jnp.array(item[0]) for item in temp]
    test_labels = [jnp.array(item[1]) for item in temp]
    del temp

    train_datasets = normalize_datasets(train_datasets)
    test_datasets = normalize_datasets(test_datasets)

    # Iterations
    eps_range = [0.7, 0.7]
    min_pts_range = [10, 80]
    threshold_range = [0, 1.0]
    ecc_pts_range = [10, 60]
    xi_range = [0.015, 0.06]

    scores = []

    scores = list(
        map(
            trial,
            param_generator(
                train_datasets,
                train_labels,
                eps_range,
                min_pts_range,
                threshold_range,
                ecc_pts_range,
                xi_range,
                num_trials,
            ),
        )
    )

    average_scores = [np.mean(samp[1]) for samp in scores]

    idx = np.array(average_scores).argmax()

    [eps, min_pts, threshold, ecc_pts, xi] = scores[idx][0]

    test_scores = trial(
        [test_datasets, test_labels, eps, min_pts, threshold, ecc_pts, xi]
    )

    test_acc = np.mean(test_scores[1])

    et = time.time()
    elapsed = et - st
    print("Execution time: ", datetime.timedelta(seconds=elapsed))
    print("LINSCAN:\n")
    print(scores[idx][0])
    # print([scores[idx][1][0], scores[idx][2][0]])
    print(average_scores[idx])
    print(test_acc)

    # OPTICS

    optics_scores = []

    def gen_rand(range):
        return random.uniform(low=range[0], high=range[1])

    for _ in range(num_trials):
        # point_scores = []
        # clust_scores = []
        opt_scores = []

        min_pts = int(np.round(gen_rand(min_pts_range)))
        threshold = gen_rand(threshold_range)
        xi = gen_rand(xi_range)

        optics_classifier = skl.cluster.OPTICS(min_samples=min_pts, xi=xi)

        for dataset, true_label in zip(train_datasets, train_labels):
            label = optics_classifier.fit_predict(dataset)

            for cat in range(max(label)):
                temp = np.array(
                    [dataset[i, :] for i in range(len(dataset)) if label[i] == cat]
                )
                if temp.size == 0:
                    continue

                eigenvalues, eigenvectors = np.linalg.eig(np.cov(temp, rowvar=False))
                if min(eigenvalues) / max(eigenvalues) > threshold:
                    label = list(map(lambda x: -1 if x == cat else x, label))

            opt_scores.append(metrics.adjusted_rand_score(label, true_label))

        optics_scores.append([[min_pts, threshold], opt_scores])

    optics_average_scores = [np.mean(samp[1]) for samp in optics_scores]

    optics_idx = np.array(optics_average_scores).argmax()

    [min_pts, threshold] = optics_scores[optics_idx][0]

    optics_classifier = skl.cluster.OPTICS(min_samples=min_pts)

    optics_test_scores = []

    opt_scores = []

    for dataset, true_label in zip(test_datasets, test_labels):
        label = optics_classifier.fit_predict(dataset)

        for cat in range(max(label)):
            temp = np.array(
                [dataset[i, :] for i in range(len(dataset)) if label[i] == cat]
            )
            if temp.size == 0:
                continue

            eigenvalues, eigenvectors = np.linalg.eig(np.cov(temp, rowvar=False))
            if min(eigenvalues) / max(eigenvalues) > threshold:
                label = list(map(lambda x: -1 if x == cat else x, label))

        opt_scores.append(metrics.adjusted_rand_score(label, true_label))

    optics_test_scores = [[min_pts, threshold], opt_scores]

    optics_test_acc = np.mean(opt_scores)
    print("\nOptics:\n")
    print(optics_scores[optics_idx][0])
    print(optics_average_scores[optics_idx])
    print(optics_test_acc)
