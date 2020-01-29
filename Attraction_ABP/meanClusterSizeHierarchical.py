import os
import argparse
import math
import glob
import numpy as np
import sklearn.cluster
import matplotlib.pyplot as plt
from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', type=str, default='.')
parser.add_argument('--steps', type=int, default=10000,
                    help="number of movement steps")
parser.add_argument('--tick', type=int, default=100,
                    help="number of steps between each save file")
args = parser.parse_args()

LENGTH = 1


def boundary_distance_matrix(coordinates):
    coordinates_rows = np.tile(coordinates, [len(coordinates), 1, 1])
    coordinates_columns = np.transpose(coordinates_rows, [1, 0, 2])

    displacement = coordinates_columns - coordinates_rows

    cross_boundary = np.abs(displacement) > (LENGTH / 2)
    displacement[cross_boundary] -= LENGTH * \
        np.sign(displacement[cross_boundary])

    return np.linalg.norm(displacement, axis=-1)


def get_mean_cluster_numbers(data_dir):
    files = [os.path.join(data_dir, f"position{i}.txt") for i in range(
        0, args.steps + 1, args.tick)]
    mean_cluster_numbers = []
    for f in files:
        particles = np.loadtxt(f)[:, :2]

        n_particles = len(particles)

        distance_matrix = boundary_distance_matrix(particles)

        clustering = sklearn.cluster.AgglomerativeClustering(
            n_clusters=None, affinity='precomputed', linkage='single', distance_threshold=0.045).fit(distance_matrix)

        labels = clustering.labels_
        num_labels = len(set(labels))

        # plt.figure(figsize=(7, 7))
        # col_value = plt.rcParams['axes.prop_cycle'].by_key()['color']
        # for i in range(len(labels)):
        #     plt.scatter(particles[i][0], particles[i][1], s=250,
        #                 color=col_value[labels[i] % len(col_value)])
        # plt.xlim(0, 1)
        # plt.ylim(0, 1)
        # plt.savefig('test.png')

        clusters = Counter(labels)
        cluster_count = Counter(clusters.values())

        mean_cluster_number = 0
        for i in cluster_count:
            mean_cluster_number += i * i * cluster_count[i] / n_particles
        mean_cluster_numbers.append(mean_cluster_number / n_particles)

    return mean_cluster_numbers


data_dirs = sorted(glob.glob(os.path.join(args.data_dir, 'data*')))
labels = [f'Angle{d}' for d in (1, 3, 5, 7, 9)]


fig, ax = plt.subplots(figsize=(10, 10))
for data_dir, label in zip(data_dirs, labels):
    mean_cluster_numbers = get_mean_cluster_numbers(data_dir)
    ax.plot(range(0, 10001, 100), mean_cluster_numbers,
            label=label)

plt.ylim(0, 1)
# plt.xlabel()
# plt.ylabel()
ax.set_title(f'Mean Cluster Size {args.data_dir}')
plt.legend(ncol=5)
plt.savefig(
    f'Mean_Cluster_Size_{"_".join(args.data_dir.split("/"))}.png')
# print('labels: ', labels)
# print('number of clusters: ', num_labels)
# print('clusters: ', clusters)
# print('cluster size count: ', cluster_count)
# print('mean cluster number:', mean_cluster_number)
