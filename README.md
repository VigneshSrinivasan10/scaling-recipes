# Scaling Laws for Flow Matching

Scaling Laws for Flow Matching (SLFM) is a project for understanding scaling laws and how they affect flow matching models.  

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

```
width=8
depth=5
train_and_evaluate "model.time_embedding_size=${width}" "model.n_blocks=${depth}"
```

## Thanks

The project starts from a very cool [notebook](https://bm371613.github.io/conditional-flow-matching/) on flow matching.  

