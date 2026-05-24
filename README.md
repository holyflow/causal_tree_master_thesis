\# Validity of Inference Methods for Causal Trees: The Honest Role of Imbalanced, Small Leaves



\*\*Author:\*\* Florian Husch 

\*\*Institution:\*\* Universität Bonn

\*\*Date:\*\* May 2026



\## Overview

This repository contains the code and data implementation for my master's thesis, "Validity of Inference Methods for Causal Trees: The Honest Role of Imbalanced, Small Leaves" The project investigates the effect of imbalanced splits on the validity of inference methods for causal tree estimators.



The full thesis document is available in the root directory as `thesis.pdf`.





\## Repository Structure

\* `src/`: Contains all modular source code for the project.

&#x20; \* `causalml/`: A customized, local version of the `causalml` library (v0.16.0), modified to add Parameters for the minimum number of treated and Control samples per leaf.

&#x20; \* `data\_generation.py`: Script to generate the data.

&#x20; \* `simulation.py`: Core logic for running the Monte-Carlo simulations.

&#x20; \* `plotting.py` / `tables.py`: Scripts used to generate the plots and tables for the thesis.

\* `main.py`: The primary orchestration script to run the pipeline and replicate the thesis results.

\* `environment.yml`: Conda environment file containing all necessary dependencies for full reproducibility.



\## Installation \& Setup

To ensure reproducibility, this project relies on a strictly defined Conda environment. 



1\. Clone the repository:

&#x20;  ```bash

&#x20;  git clone \[https://github.com/holyflow/causal\_tree\_master\_thesis.git](https://github.com/holyflow/causal\_tree\_master\_thesis.git)

&#x20;  cd causal\_tree\_master\_thesis

&#x20;  ```



2\. Create the environment from the provided file:

&#x20;  ```bash

&#x20;  conda env create -f environment\_locked.yml

&#x20;  ```



3\. Activate the environment:

&#x20;  ```bash

&#x20;  conda activate master\_thesis\_env

&#x20;  ```

