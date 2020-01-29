import os
import glob
import argparse
import numpy as np
import progressbar

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', type=str, default='.')
parser.add_argument('--save-name', type=str, default='data')
parser.add_argument('--steps', type=int)
parser.add_argument('--tick', type=int)

args = parser.parse_args()

sub_dirs = glob.glob(os.path.join(args.data_dir, 'data*'))

all_data = []
with progressbar.ProgressBar(max_value=len(sub_dirs)) as bar:
    for p, sub_dir in enumerate(sub_dirs):
        sub_data = []
        for i in range(0, args.steps+1, args.tick):
            step = np.loadtxt(os.path.join(sub_dir, f'position{i}.txt'))
            sub_data.append(step)

        sub_data = np.stack(sub_data, axis=0)

        all_data.append(sub_data)
        bar.update(p)

    all_data = np.stack(all_data, axis=0)


np.save(os.path.join(args.data_dir, args.save_name), all_data)
