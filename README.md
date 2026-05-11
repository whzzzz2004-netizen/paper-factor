# paper-factor

Extracts and generates quantitative trading factors from academic papers using LLMs.

## Quick Start

```bash
git clone https://github.com/whzzzz2004-netizen/paper-factor.git
cd paper-factor
bash scripts/setup.sh
```

Setup handles everything: dependency install, config, Docker image build, data download.

Then edit `.env` with your API keys, drop a paper in `papers/inbox/`, and run:

```bash
start --report-file papers/inbox/paper.pdf
```

Outputs go to `git_ignore_folder/factor_outputs/`.

## Requirements

- Python 3.10+
- Docker

## CLI Commands

| Command     | Description |
|-------------|-------------|
| `start`     | Full pipeline — extract + generate factors (Docker) |
| `run`       | Same pipeline, no forced backend |
| `sync`      | Download arXiv papers by search query |
| `init`      | Initialize workspace and download market data |
| `dashboard` | Launch factor results dashboard |

## Repo Hygiene

Do not commit `.env`, `git_ignore_folder/`, `log/`, `build/`, Docker image
archives, or PDF reports under `papers/`.
