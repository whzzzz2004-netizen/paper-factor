# paper-factor

Standalone extraction of the RD-Agent `paper_factor` workflow. Reads quantitative
finance papers and automatically extracts or generates factor code from them.

## Quick Start

```bash
git clone https://github.com/whzzzz2004-netizen/paper-factor.git
cd paper-factor
bash scripts/setup.sh
```

Then edit `.env` with your LLM API key and run:

```bash
start --report-file papers/inbox/your_paper.pdf
```

## Manual Install

If you prefer to install step by step:

```bash
cd paper-factor
python -m pip install -e .
cp .env.example .env
```

Edit `.env` with your LLM settings, then install the Docker execution image:

```bash
bash scripts/install_factor_docker_image.sh
```

Initialize market data:

```bash
python -m paper_factor_cli.main init --force
```

## Docker Image

Factor code executes inside a Docker container (`local_factor_exec:latest`).
The install script fetches the image from:

1. Alibaba Cloud ACR (if `FACTOR_DOCKER_ACR_IMAGE` is set — best for users in China)
2. GitHub Container Registry (`ghcr.io/<owner>/<repo>/local_factor_exec:latest`)
3. GitHub Releases (tar.gz download)
4. Local build from Dockerfile (set `FACTOR_CoSTEER_FORCE_DOCKER_BUILD=1` in `.env`)

Force a specific source:

```bash
bash scripts/install_factor_docker_image.sh --source ghcr
bash scripts/install_factor_docker_image.sh --source build
bash scripts/install_factor_docker_image.sh --source acr
bash scripts/install_factor_docker_image.sh --show-sources
```

## Initialize Market Data

```bash
python -m paper_factor_cli.main init --force
```

If `TUSHARE_TOKEN` is set in `.env`, initialization downloads the most recent
year of broad Tushare data (daily OHLCV, amount, adjustment factor, turnover,
valuation, shares, market value, industry metadata, futures, and minute bars).

Tune scope with:

- `TUSHARE_YEARS` (default `1`)
- `TUSHARE_MAX_STOCKS` (default `80`)
- `TUSHARE_MAX_FUTURES` (default `20`)
- `TUSHARE_MAX_MINUTE_SYMBOLS` (default `10`)

## Start One Paper

```bash
start --report-file papers/inbox/paper.pdf
```

Runs the full paper-factor pipeline with Docker execution.

## Run a Folder

```bash
start --report-folder papers/inbox
```

Outputs are written under `git_ignore_folder/factor_outputs`.

## CLI Commands

| Command | Description |
|---------|-------------|
| `start` | Run the full pipeline (Docker execution) |
| `run`  | Run the pipeline (conda/local execution) |
| `sync` | Download arXiv papers matching a query |
| `init` | Initialize workspace and market data |
| `dashboard` | Launch factor dashboard web UI |

## Publishing Docker Image

The `.github/workflows/docker-publish.yml` workflow auto-builds the image and
pushes to ghcr.io and GitHub Releases when a `v*` tag is pushed.

### Optional: Alibaba Cloud ACR

For faster pulls in China, add these secrets to your GitHub repo:

- `ALIYUN_ACR_USERNAME` — Alibaba Cloud ACR username
- `ALIYUN_ACR_PASSWORD` — Alibaba Cloud ACR password
- `ALIYUN_ACR_NAMESPACE` — ACR namespace (optional, defaults to GitHub username)

Users then set `FACTOR_DOCKER_ACR_IMAGE` in their `.env` to use ACR instead of ghcr.io.

## Repo Hygiene

Do not commit `.env`, `git_ignore_folder/`, `log/`, `build/`, Docker image
archives, or PDF reports under `papers/`. The repository includes `.env.example`
for users to copy and fill locally.
