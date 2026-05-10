# paper_factor Docker image install

This project can reuse a prebuilt Docker image for factor code execution:

```bash
local_factor_exec:latest
```

## One-command install

After cloning the project, run:

```bash
bash scripts/install_factor_docker_image.sh
```

The script checks whether `local_factor_exec:latest` already exists. If it does not exist, it tries these sources in order:

1. `FACTOR_DOCKER_IMAGE_URL`, if set.
2. `release_assets/local_factor_exec_latest.tar.gz`, if the file exists locally.
3. `FACTOR_DOCKER_DEFAULT_IMAGE_URL`, if set.
4. The latest GitHub Release asset named `local_factor_exec_latest.tar.gz` from the current repository remote.

## Install from a specific file

```bash
bash scripts/install_factor_docker_image.sh release_assets/local_factor_exec_latest.tar.gz
```

## Install from a specific GitHub Release URL

```bash
bash scripts/install_factor_docker_image.sh "https://github.com/<owner>/<repo>/releases/download/<tag>/local_factor_exec_latest.tar.gz"
```

## Required paper_factor settings

Make sure `.env` contains:

```bash
DS_CODER_COSTEER_ENV_TYPE=docker
FACTOR_CoSTEER_EXECUTION_BACKEND=docker
```

If the image already exists, RD-Agent will reuse it instead of rebuilding it.
