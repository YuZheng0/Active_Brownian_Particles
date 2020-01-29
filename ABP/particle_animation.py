import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', type=str, default='.')
parser.add_argument('--save-dir', type=str, default='.')
parser.add_argument('--steps', type=int,
                    help="number of movement steps")
parser.add_argument('--tick', type=int,
                    help="number of steps between each save file")
parser.add_argument('--radius', type=float, default=0.1,
                    help="radius of particles")
parser.add_argument('--length', type=float, default=1.0,
                    help="box length")
parser.add_argument('--interval', type=int, default=100,
                    help="time interval between animation frames in miliseconds")
parser.add_argument('--figsize', type=int, default=10,
                    help="figure size")
args = parser.parse_args()


def animate_data(data, region, savedir):
    plt.rcParams['animation.html'] = 'html5'

    fig, ax = plt.subplots(figsize=(args.figsize, args.figsize))

    xmin, xmax, ymin, ymax = region
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal')

    def init():
        return []

    patches = []

    def animate(i):
        if patches:
            for circle, position in zip(patches, data[i]):
                circle.center = position[0], position[1]
        else:
            for position in data[i]:
                patches.append(ax.add_patch(plt.Circle(
                    (position[0], position[1]), args.radius, edgecolor='b', facecolor='y')))

        ax.set_title(f"Step={i*args.tick}")
        return patches

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=len(data), interval=args.interval, blit=True)

    if not os.path.exists(savedir):
        os.mkdir(savedir)

    anim.save(os.path.join(savedir, 'Brownian.gif'),
              dpi=80, writer='imagemagick')


files = [f"position{i}.txt" for i in range(0, args.steps+1, args.tick)]
all_data = [np.loadtxt(os.path.join(args.data_dir, f)) for f in files]

animate_data(all_data, [0, args.length, 0, args.length], args.save_dir)
