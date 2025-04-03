# Scaling Recipes

Scaling recipes is a project for understanding best practices for scaling neural networks for different tasks. 

### Scope
- Classification on MNIST. 
    - [x] Implement Standard Parametrization (SP).  
    - [x] Implement Maximal Update Parametrization (muP).  
    - [x] Evaluate performance of different parametrizations by varying different aspects like lr, width etc. 

- [ ] Flow matching on a toy dataset. 
    - [ ] Implement Standard Parametrization (SP).  
    - [ ] Implement Maximal Update Parametrization (muP).  
    - [ ] Evaluate performance of different parametrizations by varying different aspects like lr, width etc. 


## Installation

```
python -m venv venv 
source venv/bin/activate

pip install .
pip install -e .
pip install -e ".[dev]"
```

## Usage

### Config 

The config file can be found at: `slfm/cli/conf/base.yaml`

### Train and evaluate 

- Sample command to train and evaluate the model:
```
width=512
depth=5
train_and_evaluate  "++model.width=${width}" "++model.n_blocks=${depth}"
```
- Sample command to run a sweep:
```
sweep 
```

### Expected outcome 
TBD.

## Flow matching 
TBD.


## Thanks

The project starts from a very cool [notebook](https://bm371613.github.io/conditional-flow-matching/) on flow matching. A lot of the code for scaling is borrowed from this [guide](https://github.com/cloneofsimo/scaling-guide/).

