import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', type=str, default='.')
parser.add_argument('--save-dir', type=str, default='.')
parser.add_argument('--steps', type=int, default=8000,
                    help="number of movement steps")
parser.add_argument('--tick', type=int, default=2000,
                    help="number of steps between each save file")
parser.add_argument('--binsize', type=int, default=0.04)
args = parser.parse_args()


LENGTH = 1
RADIUS = 0.02


def boundary_distance_matrix(coordinates):
    coordinates_rows = np.tile(coordinates, [len(coordinates), 1, 1])
    coordinates_columns = np.transpose(coordinates_rows, [1, 0, 2])

    displacement = coordinates_columns - coordinates_rows

    cross_boundary = np.abs(displacement) > (LENGTH / 2)
    displacement[cross_boundary] -= LENGTH * \
        np.sign(displacement[cross_boundary])

    return np.linalg.norm(displacement, axis=-1)


def velocity_correlation_matrix(velocities):
    velocities_rows = np.tile(velocities, [len(velocities), 1, 1])
    velocities_columns = np.transpose(velocities_rows, [1, 0, 2])

    mat_mul = velocities_rows * velocities_columns
    correlation = np.sum(mat_mul, axis=-1)
    correlation /= np.linalg.norm(velocities_rows, axis=-1) * \
        np.linalg.norm(velocities_columns, axis=-1)
    return correlation


def main():
    files = [f"position{i}.txt" for i in range(0, args.steps + 1, args.tick)]
    fig, axes = plt.subplots()
    fig.suptitle(args.data_dir)
    for ff in files:
        file_label = ff[8:0-4]
        f = os.path.join(args.data_dir, ff)

        coordinates = np.loadtxt(f)[:, :2]
        velocities = np.loadtxt(f)[:, 2:]

        distance_matrix = boundary_distance_matrix(coordinates)
        velocity_correlation = velocity_correlation_matrix(velocities)

        bins = np.arange(0, np.sqrt(2) / 2 * LENGTH, 2 * RADIUS)
        distance_digitized = np.digitize(distance_matrix, bins[1:], right=True)

        mean_velocity_correlation_by_distance = []
        for label in range(len(bins)):
            match = distance_digitized == label
            np.fill_diagonal(match, False)
            if np.any(match):
                velocity_correlation_of_label = velocity_correlation[match]
                mean_velocity_correlation = np.mean(
                    velocity_correlation_of_label)
                std_velocity_correlation = np.std(
                    velocity_correlation_of_label)

                mean_velocity_correlation_by_distance.append(
                    [bins[label], mean_velocity_correlation, std_velocity_correlation])

        x, y, y_err = np.asarray(
            mean_velocity_correlation_by_distance).transpose()
        np.savetxt(os.path.join(args.data_dir, f'xy{ff}.txt'),
                   np.stack([x, y], axis=1))
        axes.plot(x, y, label='density 0.1')
        #axes.fill_between(x, y-y_err, y+y_err, alpha=0.4)
        axes.set_xticks(bins)
        axes.set_xticklabels(bins, rotation=45)
        axes.set_title('Velocity Correlation')
        axes.set_xlabel('Distance')
        axes.set_ylabel('Velocity Correlation')
        plt.ylim(-1, 1)

    axes.legend()

    plt.savefig(
        f'Velocity_Correlation_{"_".join(args.data_dir.split("/"))}.png')


main()
# def plotVelocityCorrelation(data, binsize, savedir):
