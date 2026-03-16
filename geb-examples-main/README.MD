# GEB Examples

This repository contains three minimal GEB job examples:

- `cmd/`: run a command-oriented workload (inline command or external script).
- `ssh_vscode/`: run a workload that exposes an SSH port for remote access (for example, VS Code Remote SSH).
- `nootebok/`: run a notebook-oriented workflow with SSH and Jupyter ports exposed.

## Project Structure

```text
.
├── README.MD
├── geb-activate.sh
├── cmd
│   ├── example.yaml
│   ├── example_file.yaml
│   └── file.sh
├── nootebok
│   ├── example.yaml
│   ├── example_file.yaml
│   └── file.sh
└── ssh_vscode
    ├── example.yaml
    ├── example_file.yaml
    └── file.sh
```

## Files Documentation

### Root

- `README.MD`
  - Project documentation and file reference.
- `geb-activate.sh`
  - Shell profile helper script.
  - Extends `PATH` (including 3D Slicer), initializes Conda shell integration, configures Git identity placeholders, and activates the `base` Conda environment.

### `cmd/` example

- `cmd/example.yaml`
  - Self-contained job definition.
  - Uses `/bin/bash -lc` with an inline script.
  - Prints environment variables, checks GPU availability with `nvidia-smi`, activates the environment (`/home/osiris-user/geb-activate2.sh`), and lists Conda environments.
  - Uses GPU image `harbor.vhio.net/geb-env-gpu:0.1.2` with `use_gpu: true`.

- `cmd/example_file.yaml`
  - Job definition that runs an external script file.
  - Calls `/nfs/rwork/code/geb-examples/cmd/file.sh`.
  - Uses CPU image `harbor.vhio.net/geb-env-cpu:0.1.2` with `use_gpu: false`.
  - Includes sample env variables (`MY_VAR1`, `MY_VAR2`) and runtime/resource limits.

- `cmd/file.sh`
  - Bash script executed by `cmd/example_file.yaml`.
  - Uses strict mode: `set -euo pipefail`.
  - Prints `MY_VAR1` and `MY_VAR2`, runs `nvidia-smi`, activates GEB env, and lists Conda environments.

### `ssh_vscode/` example

- `ssh_vscode/example.yaml`
  - Self-contained job definition with inline script.
  - Prints sample env variables including `PUBLIC_KEY`.
  - Runs `nvidia-smi` and starts container startup flow via `. /start.sh`.
  - Exposes SSH by mapping external port `9999` to internal port `22` through `openports`.

- `ssh_vscode/example_file.yaml`
  - Job definition that runs external script `/nfs/rwork/code/geb-examples/ssh_vscode/file.sh`.
  - Same resource profile and open port mapping as `ssh_vscode/example.yaml`.
  - Intended for the "script in file" workflow.

- `ssh_vscode/file.sh`
  - Bash script executed by `ssh_vscode/example_file.yaml`.
  - Uses strict mode: `set -euo pipefail`.
  - Prints `MY_VAR1`, `MY_VAR2`, and `PUBLIC_KEY`, runs `nvidia-smi`, then executes `. /start.sh`.

### `nootebok/` example

- `nootebok/example.yaml`
  - Self-contained job definition with inline script.
  - Prints `MY_VAR1`, `MY_VAR2`, `PUBLIC_KEY`, and `JUPYTER_PASSWORD`.
  - Starts container startup flow via `. /start.sh`.
  - Uses CPU image `harbor.vhio.net/geb-env-cpu:0.1.2` with `use_gpu: false`.
  - Exposes SSH (`9999 -> 22`) and Jupyter (`8888 -> 8888`) through `openports`.

- `nootebok/example_file.yaml`
  - Job definition that executes `/nfs/rwork/code/geb-examples/nootebok/file.sh`.
  - Same CPU resource profile and port mapping as `nootebok/example.yaml`.
  - Intended for the "script in file" workflow.

- `nootebok/file.sh`
  - Bash script executed by `nootebok/example_file.yaml`.
  - Uses strict mode: `set -euo pipefail`.
  - Prints `MY_VAR1`, `MY_VAR2`, and `JUPYTER_PASSWORD`, then runs `. /start.sh`.

## Key Differences Between Example Types

- Inline command examples:
  - `cmd/example.yaml`
  - `ssh_vscode/example.yaml`
  - `nootebok/example.yaml`
  - Script is embedded directly in YAML (`args: -lc | ...`).

- External script examples:
  - `cmd/example_file.yaml`
  - `ssh_vscode/example_file.yaml`
  - `nootebok/example_file.yaml`
  - Script logic is stored in `file.sh` and executed from YAML.

## Configuration Notes

- `worker_type`, `max_cpus`, `max_memory_mb`, and `max_runtime_minutes` define compute/runtime limits.
- `docker_image` selects the container environment.
- `env` sets runtime environment variables.
- `use_gpu` controls whether GPU resources are requested.
- `openports` (in `ssh_vscode/` and `nootebok/`) exposes container services on external ports.
- In this repository, port `22` is used for SSH and port `8888` is used for Jupyter.
