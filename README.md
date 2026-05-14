<div align="center">

# SpaNECT

**Cell-type-informed Characterization of Spatial Niches from Spatial Multimodal and Multi-omics Data**

A Python research codebase for spatial niche identification and characterization from spatial transcriptomics, spatial multimodal, and spatial multi-omics data.

</div>

## Background

Spatial niches are shaped by local cellular composition, neighborhood structure, and multimodal molecular context. SpaNECT is designed to characterize such niches by combining spatial graph construction, feature preparation, multimodal representation learning, clustering, evaluation, and visualization within one workflow.

The current repository contains the code base used for the SpaNECT project, including:

- a pipeline-style `SpaNECT` class for end-to-end analysis
- modular APIs under `spanect.gr`, `spanect.pp`, `spanect.tl`, `spanect.pl`, `spanect.ch`, and `spanect.datasets`
- example scripts for single-omics, multimodal, and multi-omics experiments
- tests for smoke, standalone, and integration-level behaviors

## Features

- **Single-omics spatial niche analysis** for DLPFC-style spatial transcriptomics workflows.
- **Spatial multi-omics support** through multimodal inputs such as RNA plus ADT or other secondary omics matrices stored in `AnnData`.
- **Cell-type-informed characterization** through dedicated cell-type and modality-weight analysis utilities.
- **Unified pipeline interface** via `SpaNECT.load()`, `construct_graph()`, `get_model_input()`, `train()`, `evaluate()`, and `plot()`.
- **Modular internal API** for graph construction, preprocessing, training, evaluation, visualization, and characterization.
- **Raw input and** **`.h5ad`** **workflows** depending on dataset organization.

## Repository Structure

```text
SpaNECT-v1.4/
├── src/spanect/           # source package
├── scripts/pipeline/      # example pipeline scripts
├── config/                # published example configs
├── tests/                 # test suite
├── data/                  # placeholder note for local data
├── pyproject.toml         # package metadata
├── requirements.txt       # pip-oriented dependency list
├── environment.yml        # conda environment specification
├── LICENSE
└── README.md
```

## Installation

### Option 1: reproduce the conda environment

```bash
conda env create -f environment.yml
conda activate spanect
pip install -e .
```

This is the recommended route for reproducing the working environment used in development.

### Option 2: lightweight installation

```bash
pip install -r requirements.txt
pip install -e .
```

If you plan to use GPU training, install the appropriate PyTorch and CUDA build for your machine first.

## Requirements

SpaNECT was prepared in the conda environment `spanect` with Python 3.10 and PyTorch 2.2.2.

Key dependencies include:

- `python==3.10.16`
- `pytorch==2.2.2`
- `torchvision==0.17.2`
- `torchaudio==2.2.2`
- `scanpy==1.9.3`
- `anndata==0.9.1`
- `numpy==2.2.6`
- `scipy==1.11.4`
- `pandas==2.0.1`
- `scikit-learn==1.2.2`
- `matplotlib==3.7.2`
- `networkx==3.4.2`
- `python-igraph / leidenalg`
- `opencv-python==4.7.0.72`
- `PyYAML==6.0`
- `tqdm==4.65.0`
- `byol-pytorch==0.6.0`

For the full environment used in practice, see [environment.yml](./environment.yml).

### Optional R setup for `mclust`

Some evaluation utilities rely on the R package `mclust` through `rpy2`.

If R is not automatically detected on your machine, set:

```bash
export R_HOME=/path/to/your/R
export R_LIBS_USER=/path/to/your/R/library
```
The `mclust` dependency is only required when using `evaluate_mclust`.

## Quick Start

After installation:

```bash
pip install -e .
```

you can use the package API:

```python
from spanect import SpaNECT

model = SpaNECT(
    config_path="config/dlpfc1.yaml",
    dataset="DLPFC",
    data_name="151507",
    save_path="./results",
    multiomics=False,
)

model.load()
model.construct_graph()
model.get_model_input()
model.train()
model.evaluate()
model.plot()
```

## Example Pipelines

### Single-omics example

```bash
python all.py
```

Uses:

- `config/dlpfc1.yaml`

### Multi-omics example

```bash
python all_m.py
```

Uses:

- `config/tonsil24.yaml`

## Data Availability

The processed datasets used in this study are available at Zenodo:

<https://doi.org/10.5281/zenodo.20064886>

This repository does not include large local training data, generated result files, or cached intermediate artifacts.

## Tests

Run the default test suite with:

```bash
pytest
```

The current pytest configuration excludes integration tests by default.

## Paper

This repository accompanies the SpaNECT paper:

**Cell-type-informed Characterization of Spatial Niches from Spatial Multimodal and Multi-omics Data**

## License

This repository is released under the BSD 3-Clause License. See [LICENSE](./LICENSE).
