# paper_factor

This is a standalone extraction of the RD-Agent `paper_factor` workflow. It exposes one command, `start`, for running the paper-factor pipeline.

## Install

```bash
cd paper_factor_project
python -m pip install -e .
cp .env.example .env
```

Create a `.env` with the same LLM settings used by the original project.

Docker must be installed and running. The factor-code execution image is
`local_factor_exec:latest`. Install a prebuilt image with:

```bash
bash scripts/install_factor_docker_image.sh
```

Alternatively, leave `FACTOR_CoSTEER_FORCE_DOCKER_BUILD=1` in `.env` to build
the image locally from `rdagent/components/coder/factor_coder/docker/Dockerfile`.

## Initialize Recent Data

```bash
python -m paper_factor_cli.main init --force
```

If `TUSHARE_TOKEN` is set in `.env`, initialization downloads the most recent 1 year of broad Tushare data into `git_ignore_folder/factor_implementation_source_data`, including daily OHLCV, amount, adjustment factor, turnover, valuation, share count, market value, stock industry/market metadata, selected futures, and minute bars. You can tune scope with:

- `TUSHARE_YEARS` (default `1`)
- `TUSHARE_MAX_STOCKS` (default `80`)
- `TUSHARE_MAX_FUTURES` (default `20`)
- `TUSHARE_MAX_MINUTE_SYMBOLS` (default `10`)

## Start One Paper

```bash
start --report-file /path/to/paper.pdf
```

`start` runs the full paper-factor pipeline with Docker execution and up to 10 factors per paper.

## Run A Folder

```bash
start --report-folder papers/inbox
```

Outputs are written under `git_ignore_folder/factor_outputs`, matching the original workflow.

## Publishing To GitHub

Do not commit `.env`, `git_ignore_folder/`, `log/`, `build/`, Docker image
archives, or PDF reports under `papers/`. The repository includes `.env.example`
for users to copy and fill locally.

For Docker, prefer publishing the prebuilt execution image as a GitHub Release
asset named `local_factor_exec_latest.tar.gz`:

```bash
docker build -t local_factor_exec:latest rdagent/components/coder/factor_coder/docker
docker save local_factor_exec:latest | gzip > local_factor_exec_latest.tar.gz
```

Upload that tarball to a GitHub Release. Users can then run:

```bash
bash scripts/install_factor_docker_image.sh \
  "https://github.com/<owner>/<repo>/releases/download/<tag>/local_factor_exec_latest.tar.gz"
```
