# pollcomm
Code for paper "Collapse of perturbed pollinator communities under rapid environmental change" authored by Sjoerd Terpstra, Flávia M.D. Marquitti & Vítor V. Vasconcelos.

## Installation
The Python environment is specified in ```environment.yml```.

Install the virtual environment using ```conda env create```.

Then, activate the environment using ```conda activate pollcomm```.

The environment can be deactivated using ```conda deactivate```.

## Usage
The graphs of the manuscript can be reproduced by the functions defined in ```ms.py```.

The graphs of the sensitivity analysis in the Supporting Information can be reproduced by the functions defined in ```sa.py```.

Both ```main.py``` and ```graphs.py``` contain old code for more experiments and to produce figures not used
in the manuscript.

The model is implemented in the ```pollcomm``` module. Only ```adaptive_model.py``` (adaptive foraging)
and ```base_model.py``` (no adaptive foraging) are used for the model as used in the manuscript.
