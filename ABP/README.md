# Active_Brownian_Particles

## Data generation script

`Active_Brownian_Particles.cpp` is the code for generating Brownian particles data. Global constants may be changed as needed. Of the most interest are `NUMSIMS`, `STEP` and `TICK`. `NUMSIMS` is the number of simulations, i.e. the size of dataset, `STEP` is the number of timesteps for each simulation, and `TICK` controls how often motion state is saved. The rest controls the dynamics of simulations.

Data is organized into subfolders by simulation ids. For example, folder `data243` contains the motion state of the system from simulation `243`. In each folder, each txt file contains the system state of a particular step, marked by the numerical id in the file name, e.g. `position50.txt` being the state of the system at step `50`. The four columns in the txt file represents $x, y, v_x, v_y$, while the rows corresponds to particles.

To enable proper data saving of the C++ script, a separate bash script `run.sh` is written to create the necessary folder structure as well as compiling the code and running it. It serves as a template that the user may change as needed.

## Data aggregation script

`combine_data.py` aggregates the output txt files from the data generation script to produce a single `.npy` data file, with the shape `[S, T, N, D]`, where `S` is the number of simulaitons, `T` is the number of time steps, `N` is the number of particles, and `D` is the dimension of state vector $(x, y, vx, vy)$.

Example usage:
```bash
python3 combine_data.py --steps=1000 --tick=10 --data-dir=. --save-name=data 
```
Here `--steps` and `--tick` correspend to the `STEP` and the `TICK` constants in the data generation scripts, respectively. The script tries to find all sub data folders in location specified by `--data-dir`. The output is a single `data.npy` file, as the user specified the name. `--data-dir` defaults to current directory, `--save-name` defaults to `data`.

## Visualization script

`particle_animation.py` visualizes a single instance of simulation to a gif file. Required arguments are `--steps` and `--tick` which can match the `STEP` and `TICK` in the data generation scipt. The user are free to use a smaller step and a wider tick to sample from the entire data set. `--data-dir` specifies the directory that contains the txt files, and `--save-dir` points to where the gif is to be saved.

Example usage:
```bash
python3 particle_animation.py --steps=1000 --tick=10 --data-dir=data42 --save-dir=.
```

In addition `--radius` and `--length` must match the `RADIUS` and `LENGTH` parameters respectively, but have their default values if the counter parts in the data generation script are unchanged.
