# R2GB-GA
A Reaction-Regulated Graph-Based Genetic Algorithm for Exploring Synthesizable Chemical Space


Code description
========

* genetic_torch.py

Main code of the R2GB-GA framework used for HSP90 inhibitor generation.


* gp_model.pth

Pretrained Gaussian Process regression model used to evaluate molecular properties during selection.


* initpool.dat

Initial molecular pool for the HSP90 inhibitor generation task.


* mutate_reaction.dat

List of predefined reaction rules used for mutation operations.


* crossover.py

Crossover module within the genetic algorithm.


How to Run
========

1. Install [Anaconda](https://www.anaconda.com/products/individual)


2. Create a conda environment from the `env.yml` file

```bash
$ conda env create --file env.yml
```


3. Activate the conda environment

```bash
$ conda activate R2GB
```


4. Run the main script

```bash
$ python3 genetic_torch.py > log.dat
```


Output files are stored in the 'steps/' directory. Each index file in 'steps/' corresponds to a specific step of the genetic cycle and contains the selection cutoff value, molecules in the pool, and their properties.

