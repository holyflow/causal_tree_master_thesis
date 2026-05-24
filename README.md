# Validity of Inference Methods for Causal Trees: The Honest Role of Imbalanced, Small Leaves

**Author:** Florian Husch  
**Institution:** Universität Bonn  
**Date:** May 2026

## Overview
This repository contains the code and data implementation for my master's thesis, "Validity of Inference Methods for Causal Trees: The Honest Role of Imbalanced, Small Leaves". The project investigates the effect of imbalanced splits on the validity of inference methods for causal tree estimators.

The full thesis document is available in the root directory as `thesis.pdf`.

## Repository Structure
* `src/`: Contains all modular source code for the project.
  * `causalml/`: A customized, local version of the `causalml` library (v0.16.0), modified to add parameters for the minimum number of treated and control samples per leaf.
  * `data_generation.py`: Script to generate the data.
  * `simulation.py`: Core logic for running the Monte-Carlo simulations.
  * `plotting.py` / `tables.py`: Scripts used to generate the plots and tables for the thesis.
* `main.py`: The primary orchestration script to run the pipeline and replicate the thesis results.
* `environment.yml`: Conda environment file containing all necessary dependencies for full reproducibility.

## Installation & Setup
To ensure reproducibility, this project relies on a strictly defined Conda environment. 

1. Clone the repository:
   ```bash
   git clone [https://github.com/holyflow/causal_tree_master_thesis.git](https://github.com/holyflow/causal_tree_master_thesis.git)
   cd causal_tree_master_thesis
   ```

2. Create the environment from the provided file:
   ```bash
   conda env create -f environment_locked.yml
   ```

3. Activate the environment:
   ```bash
   conda activate master_thesis_env
   ```