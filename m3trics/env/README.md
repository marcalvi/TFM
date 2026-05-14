# M3TRICS Conda Environments

Setup guide for running M3TRICS on VHIO / OSIRIS GPU workers.

Use this directory from the project root or from:

```bash
/home/osiris-user/Desktop/TFM/m3trics
```

## Available Environments

| Environment | Workers | PyTorch CUDA wheels |
| --- | --- | --- |
| `m3trics_4090` | `4090`, `4090-64`, `4090-128` | CUDA 12.1 |
| `m3trics_5090` | `5090` | CUDA 13.0 |

## 1. Configure Pip Proxy

Set the VHIO pip proxy as the default package index:

```bash
pip config set global.index-url http://repo.radiomics.vhio.net:9999/simple/
pip config set global.trusted-host repo.radiomics.vhio.net
```

Equivalent one-off usage:

```bash
pip install --index-url http://repo.radiomics.vhio.net:9999/simple/ PACKAGE
```

## 2. Configure Conda Proxy Channel

Set the VHIO conda-forge proxy as a default conda channel:

```bash
conda config --add channels http://repo.radiomics.vhio.net:9999/conda/conda-forge
conda config --set channel_priority strict
```

Equivalent one-off usage:

```bash
conda install -c http://repo.radiomics.vhio.net:9999/conda/conda-forge PACKAGE
```

## 3. Create An Environment

### RTX 4090, 4090-64, 4090-128

```bash
conda env create -f env/m3trics_4090.yml
conda activate m3trics_4090
```

### RTX 5090

```bash
conda env create -f env/m3trics_5090.yml
conda activate m3trics_5090
```

## 4. Verify PyTorch And CUDA

Run this after activating the selected environment:

```bash
python - <<'PY'
import torch

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda version:", torch.version.cuda)
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
PY
```

Expected result:

- `cuda available: True`
- `gpu:` should show the assigned NVIDIA GPU.
- `cuda version:` should match the environment family, usually `12.1` for `m3trics_4090` and `13.0` for `m3trics_5090`.

## 5. Update An Existing Environment

```bash
conda env update -n m3trics_4090 -f env/m3trics_4090.yml --prune
conda env update -n m3trics_5090 -f env/m3trics_5090.yml --prune
```

## 6. Remove An Environment

```bash
conda env remove -n m3trics_4090
conda env remove -n m3trics_5090
```

## Notes

- Use `m3trics_4090` for `4090`, `4090-64`, and `4090-128` workers.
- Use `m3trics_5090` for `5090` workers.
- If CUDA is not visible after activation, first confirm that the job was launched with GPU access enabled.
