import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', type=str, default='.')
parser.add_argument('--steps', type=int, default=10000,
                    help="number of movement steps")
parser.add_argument('--tick', type=int, default=3000,
                    help="number of steps between each save file")
args = parser.parse_args()

LENGTH = 1
files = [f"position{i}.txt" for i in range(0, args.steps + 1, args.tick)]
for f in files:
    ff = os.path.join(args.data_dir, f)
    data = np.loadtxt(ff)
    particles = data[:, :2]
    velocities = data[:, 2:]
    n_particles = len(particles)
    fig, ax = plt.subplots(figsize=(7, 7))
    for i in range(n_particles):
        #plt.scatter(particles[i][0], particles[i][1], s=250, color='cyan')
        plt.quiver(particles[i][0], particles[i][1],
                   velocities[i][0], velocities[i][1])
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        ax.set_title(f'Velocity Plot {args.data_dir} {f[:-4]}')

    plt.savefig(
        f'Velocity_plot_{"_".join(args.data_dir.split("/"))}_{f[:-4]}.png')
