# ConformalAutoencoder

A research codebase implementing autoencoder variants that emphasize geometric structure preservation (conformal and isometric constraints). The project contains implementations of base, variational, isometric, scaled-isometric, and conformal autoencoders, utilities for geometric loss computation, data generators, example notebooks, and pretrained model checkpoints.

## Key features

- ConformalAutoencoder, IsometricAutoencoder and related models (see `mylib/Autoencoders.py`).
- Geometric loss functions: conformality and isometry losses, trace estimators and Jacobian utilities (see `mylib/metrics.py`).
- Synthetic data helpers (e.g. half-sphere generator) and MNIST experiment pipelines (see `mylib/data.py`).
- Example notebooks under `toyexamples/` and `mnist_optuna/` demonstrating training and evaluation.
- Model checkpoint saving/loading in `models/`.

## Quick start

Prerequisites: Python 3.8+ and git. This repository uses a `pyproject.toml`; you can install using pip or Poetry.

1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies (choose one):

# If you prefer pip and the project supports editable install
```bash
pip install -e .
```

# Or, if you use Poetry (recommended when available):
```bash
poetry install
```

If the project does not publish dependencies in pyproject, install common packages used by the notebooks (torch, torchvision, jupyter, numpy, matplotlib, pandas). Example:

```bash
pip install torch torchvision jupyterlab numpy matplotlib pandas
```

## Running experiments / examples

- Notebooks: Most experiments are implemented as Jupyter notebooks in `toyexamples/` and `mnist_optuna/`. Start Jupyter Lab and open a notebook:

```bash
jupyter lab
```

- Scripts: The core implementations are in `mylib/`. You can import the modules and run training loops from your own script, or adapt code cells from the notebooks into a script.

Example minimal usage (Python REPL or script):

```python
from mylib.Autoencoders import ConformalAutoencoder
from mylib.data import make_half_sphere

# create model, data loaders, then call the training helper defined in Autoencoders
```

## Loading pretrained models

Pretrained checkpoints are stored in `models/` with descriptive names like `lambda_{value}_model_{run}.pth` or other experiment names. Use the model loading utilities in `mylib/helper.py` or the model class' `load_model_from_checkpoint()` to restore weights for evaluation or further training.

## Project layout

- `mylib/Autoencoders.py` — model classes and training interfaces.
- `mylib/metrics.py` — conformality/isometry loss implementations and evaluation helpers.
- `mylib/data.py` — synthetic datasets and dataloaders.
- `mylib/helper.py` — persistence and optimizer/scheduler helpers.
- `toyexamples/` — many example notebooks for experiments and visualizations.
- `models/` — saved model checkpoints.

## Reproducing experiments

1. Pick a notebook in `toyexamples/` (for example `conformal_autoencoder_jacobian_comp_halfsphere.ipynb`).
2. Open it in Jupyter Lab, set the Python kernel to your project's virtual environment, and run cells in order. Notebooks follow a consistent pattern: setup -> data -> training loop -> evaluation.
3. Checkpoints will be saved to `models/` by default (see the notebook or `Autoencoders` training helpers for exact paths).

## Notes & tips

- The repository favors reproducible experiments: look for `lambda_*` hyperparameter naming and seed management inside the notebooks and training helpers.
- For large Jacobian-based losses, use a machine with a GPU. Code uses batching and trace estimators for memory efficiency.

## Citation

If you use this code in research, please cite the accompanying paper or contact the repository owner for citation details.

## Contributing

Contributions are welcome. Please open an issue to discuss changes before submitting pull requests, and follow the existing code style for minimal diffs.

## License

Check the repository root for a LICENSE file. If none exists, contact the owner to clarify licensing before reuse.
