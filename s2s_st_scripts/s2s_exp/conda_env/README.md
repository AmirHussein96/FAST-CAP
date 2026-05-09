# Environment Reproducibility

This directory contains all files required to **reproduce the software environment** used for training, evaluation, and streaming experiments in this repository.

The environment is managed using **Conda**, with additional packages installed via **pip** when necessary.

All experiments were developed and tested using **Python 3.10.12**.

---

## Files

### `conda_environment.yml`
Primary Conda environment specification.

- Exported using:
  ```bash
  conda env export --no-builds
  ```

  ### `conda_environment.yml`

- Platform-agnostic (build strings removed)

---

### `requirements.txt`

- List of Python packages installed via `pip`
- Exported using:
  ```bash
  pip freeze
  ```

- Must be installed after the Conda environment is created
- Ensures that all pip-only dependencies are captured

---

## Environment Setup

### 1. Create the Conda environment
```bash
conda env create -f conda_environment.yml
```
### 2. Activate the environment
```bash
conda activate <env_name>
```
Replace `<env_name>` with the environment name specified in `conda_environment.yml`.

### 3. Install pip dependencies

```bash
pip install -r requirements.txt
```

