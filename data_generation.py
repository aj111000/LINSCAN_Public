import numpy as np
import numpy.random as rand
import random


def gen_data(lin_clusts=10,
             lin_num=100,
             lin_R=2,
             int_clusts=10,
             int_num=100,
             int_R=2,
             iso_clusts=5,
             iso_num=200,
             iso_R=10,
             noise_num=400,
             x_min=-50,
             x_max=50,
             y_min=-50,
             y_max=50):
    def gen(clusts, num_pts, R, inds, subdivs_across, len_mult=1):
        data = []
        lab = []

        def unif():
            return rand.uniform(low=-0.5, high=0.5)

        x_diff = (x_max - x_min) / subdivs_across
        y_diff = (y_max - y_min) / subdivs_across

        for i in range(clusts):
            subdiv_idx = inds.pop(random.randint(0, len(inds) - 1))
            subdiv_y = subdiv_idx // subdivs_across
            subdiv_x = subdiv_idx % subdivs_across
            x = (subdiv_x + .5) * x_diff + x_min
            y = (subdiv_y + .5) * y_diff + y_min
            s = unif() * np.pi
            length = rand.uniform(low=.5, high=1) * min(x_diff, y_diff) / 2.5 * len_mult + 1
            top_x = x + length * np.sin(s)
            bot_x = x - length * np.sin(s)
            top_y = y + length * np.cos(s)
            bot_y = y - length * np.cos(s)

            dx = (bot_x - top_x) / (num_pts - 1)
            dy = (bot_y - top_y) / (num_pts - 1)
            for j in range(num_pts):
                x1 = top_x + dx * j
                y1 = top_y + dy * j

                ddx = unif() * R * (rand.uniform() + .1)
                ddy = unif() * R * (rand.uniform() + .1)
                data.append([x1 + ddx, y1 + ddy])
                lab.append(i)

            subdiv_idx += 1

        return data, lab, inds

    def gen_int(clusts, num_pts, R, inds, subdivs_across, lab_start):
        data = []
        lab = []

        def unif():
            return rand.uniform(low=-0.5, high=0.5)

        x_diff = (x_max - x_min) / subdivs_across
        y_diff = (y_max - y_min) / subdivs_across

        for i in range(clusts):
            subdiv_idx = inds.pop(random.randint(0, len(inds) - 1))
            subdiv_y = subdiv_idx // subdivs_across
            subdiv_x = subdiv_idx % subdivs_across
            x = (subdiv_x + .5) * x_diff + x_min
            y = (subdiv_y + .5) * y_diff + y_min
            s = unif() * np.pi

            int_angle = (random.uniform(.3, .7) + random.randint(0, 1)) * np.pi

            length = rand.uniform(low=.5, high=1) * min(x_diff, y_diff) / 2.5 + 1
            top_x = x + length * np.sin(s)
            bot_x = x - length * np.sin(s)
            top_y = y + length * np.cos(s)
            bot_y = y - length * np.cos(s)

            sep_frac = random.uniform(.2, .4)

            top_x_2 = x + (1+sep_frac)*length * np.sin(s + int_angle)
            bot_x_2 = x + sep_frac * length * np.sin(s + int_angle)
            top_y_2 = y + (1+sep_frac)*length * np.cos(s + int_angle)
            bot_y_2 = y + sep_frac * length * np.cos(s + int_angle)

            dx = (bot_x - top_x) / (num_pts - 1)
            dy = (bot_y - top_y) / (num_pts - 1)

            dx_2 = (bot_x_2 - top_x_2) / (np.floor(num_pts / 2) - 1)
            dy_2 = (bot_y_2 - top_y_2) / (np.floor(num_pts / 2) - 1)
            for j in range(num_pts):
                x1 = top_x + dx * j
                y1 = top_y + dy * j

                ddx = unif() * R * (rand.uniform() + .1)
                ddy = unif() * R * (rand.uniform() + .1)
                data.append([x1 + ddx, y1 + ddy])
                lab.append(lab_start + 2 * i)

            subdiv_idx += 1

            for j in range(int(np.floor(num_pts / 2))):
                x1 = top_x_2 + dx_2 * j
                y1 = top_y_2 + dy_2 * j

                ddx = unif() * R * (rand.uniform() + .1)
                ddy = unif() * R * (rand.uniform() + .1)
                data.append([x1 + ddx, y1 + ddy])
                lab.append(lab_start + 2 * i + 1)

            subdiv_idx += 1

        return data, lab, inds

    subdivs_across = np.ceil(np.sqrt(lin_clusts + iso_clusts + int_clusts))

    inds = list(range(lin_clusts + iso_clusts + int_clusts))
    data = []
    labels = []

    lin_data, lin_labels, inds = gen(lin_clusts, lin_num, lin_R, inds, subdivs_across)

    data = [*data, *lin_data]
    labels = [*labels, *lin_labels]

    int_data, int_labels, inds = gen_int(int_clusts, int_num, int_R, inds, subdivs_across, max(labels) + 1)

    data = [*data, *int_data]
    labels = [*labels, *int_labels]

    iso_data, iso_labels, _ = gen(iso_clusts, iso_num, iso_R, inds, subdivs_across, len_mult=0)

    iso_labels = list(map(lambda x: -1, iso_labels))

    data = [*data, *iso_data]
    labels = [*labels, *iso_labels]

    noise_data = []
    noise_labels = []

    for j in range(noise_num):
        noise_data.append(
            [rand.uniform(x_min, x_max),
             rand.uniform(y_min, y_max)]
        )
        noise_labels.append(-1)

    data = [*data, *noise_data]
    labels = [*labels, *noise_labels]

    return data, labels


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt

    data, labels = gen_data()
    data = np.array(data)
    fig1 = plt.figure()
    ax = fig1.add_subplot(111)
    ax.set_aspect('equal', adjustable='box')

    plt.scatter(data[:, 0], data[:, 1], c=labels, marker='o', s=(2 * 72. / fig1.dpi) ** 2)
    plt.show()
