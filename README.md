# Marketing Science Bootcamp (Cohort Mar-26)

A hands-on course covering Marketing Mix Modeling (MMM), attribution, and
geo-experimentation. The repo contains Jupyter notebooks for live sessions and
self-paced offline work, plus a shared Python utilities package and synthetic
datasets.

## Prerequisites

- **Python 3.11+** (pinned to 3.11 in `.python-version` — required by TensorFlow/Meridian)
- **uv** — fast Python package and project manager
  ([install instructions](https://docs.astral.sh/uv/getting-started/installation/))
- **R** (optional) — only needed for the GeoLift notebooks that use `rpy2`
- **Graphviz system library** (optional) — for DAG visualizations
  (`brew install graphviz` on macOS)

## Quick Start with uv

### 1. Clone the repository

```bash
git clone https://github.com/gdberrio/MarketingScience-CohortMarch26.git
cd MarketingScience-CohortMarch26
```

### 2. Install dependencies

uv reads `pyproject.toml` (and the existing `uv.lock`) automatically:

```bash
uv sync
```

This creates a `.venv/` virtual environment and installs every dependency.

> **Tip:** If you only need the core data-science stack and want to skip
> optional packages like `rpy2` or `google-meridian`, you can install from the
> lightweight `requirements.txt` instead:
>
> ```bash
> uv pip install -r requirements.txt
> ```

### 3. Verify your setup

```bash
uv run jupyter nbconvert --to notebook --execute pre-course/00_smoke_test.ipynb
```

If the notebook executes without errors, your environment is ready.

### 4. Launch Jupyter

```bash
uv run jupyter lab
```

Or, if you prefer classic notebooks:

```bash
uv run jupyter notebook
```

## Repository Structure

```
cohort-Mar26/
├── pre-course/          # Setup, smoke test, introductory readings
├── week-1/              # Measurement landscape, data prep & EDA
│   ├── session-1/
│   ├── session-2/
│   └── offline/
├── week-2/              # OLS MMM, Bayesian MMM, adstock & saturation
│   ├── session-3/
│   ├── session-4/
│   └── offline/
├── week-3/              # Meridian, attribution, BYOD
│   ├── session-5/
│   ├── session-6/
│   └── offline/
├── week-4/              # Experimentation, GeoLift, capstone wrap-up
│   ├── session-7/
│   ├── session-8/
│   └── offline/
├── capstone/            # Capstone project template and guide
├── utils/               # Shared helper modules
│   ├── mmm_utils.py     #   Adstock, saturation, OLS, diagnostics
│   ├── eda_utils.py     #   Data loading, plotting, correlations
│   └── geo_utils.py     #   Geo-experiment helpers (import directly)
├── data/                # Datasets and interactive demos
│   ├── generate_synthetic_data.py
│   ├── adstock_shiny_app.py
│   └── *.csv / *.xlsx
├── pyproject.toml       # Project metadata and dependencies (uv/pip)
├── uv.lock              # Locked dependency versions
└── requirements.txt     # Lightweight alternative dependency list
```

Each notebook (`*.ipynb`) has a companion Markdown (`.md`) file with the same
content for easy diffing and review.

### Naming conventions

| Type              | Pattern                              |
|-------------------|--------------------------------------|
| Live session      | `week-N/session-M/session_XX_title.ipynb` |
| Offline exercise  | `week-N/offline/notebook_XX_title.ipynb`  |
| Reading           | `reading_XX_title.md`                |

## Using the Utils Package

Notebooks add `utils/` to `sys.path` so you can import helpers directly:

```python
import sys, os
sys.path.insert(0, os.path.join(os.pardir, "utils"))

from utils import adstock_geometric, load_workshop_data
```

`geo_utils` is **not** re-exported from the package `__init__`; import it
explicitly when needed:

```python
from utils.geo_utils import load_geo_data
```

## Common Tasks

### Regenerate synthetic datasets

```bash
uv run python data/generate_synthetic_data.py
```

### Run the interactive adstock Shiny demo

```bash
uv run python data/adstock_shiny_app.py
```

### Execute a notebook headlessly

```bash
uv run jupyter nbconvert --to notebook --execute week-1/session-1/session_01_measurement_landscape.ipynb
```

### Add a new dependency

```bash
uv add <package-name>
```

This updates `pyproject.toml` and `uv.lock` in one step.

## Key Dependencies

| Category            | Packages                                         |
|---------------------|--------------------------------------------------|
| Core data science   | pandas, numpy, matplotlib, seaborn, scipy, statsmodels, scikit-learn |
| Bayesian modeling   | pymc 5.x, arviz, pytensor                        |
| MMM frameworks      | pymc-marketing, google-meridian                   |
| Visualization       | graphviz, networkx, ipywidgets                    |
| Jupyter             | jupyter, jupyterlab                               |
| Interactive apps    | shiny, nest-asyncio                               |
| R bridge (optional) | rpy2                                              |
