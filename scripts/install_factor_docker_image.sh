#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${FACTOR_DOCKER_IMAGE:-local_factor_exec:latest}"
ASSET_NAME="${FACTOR_DOCKER_ASSET_NAME:-local_factor_exec_latest.tar.gz}"
DEFAULT_IMAGE_URL="${FACTOR_DOCKER_DEFAULT_IMAGE_URL:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CACHE_DIR="${FACTOR_DOCKER_CACHE_DIR:-${PROJECT_ROOT}/release_assets}"
FORCE="${FACTOR_DOCKER_FORCE_INSTALL:-0}"
SOURCE=""

usage() {
  cat <<EOF
Usage:
  bash scripts/install_factor_docker_image.sh [--force] [image_tar_or_url]

Examples:
  bash scripts/install_factor_docker_image.sh
  bash scripts/install_factor_docker_image.sh release_assets/local_factor_exec_latest.tar.gz
  bash scripts/install_factor_docker_image.sh https://github.com/<owner>/<repo>/releases/download/<tag>/local_factor_exec_latest.tar.gz

Environment variables:
  FACTOR_DOCKER_IMAGE              Docker image tag to verify. Default: local_factor_exec:latest
  FACTOR_DOCKER_IMAGE_URL          Download URL used when no argument is provided.
  FACTOR_DOCKER_ASSET_NAME         Release asset filename. Default: local_factor_exec_latest.tar.gz
  FACTOR_DOCKER_CACHE_DIR          Download/cache directory. Default: release_assets
  FACTOR_DOCKER_FORCE_INSTALL=1    Reload even when the image already exists.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --force)
      FORCE=1
      shift
      ;;
    *)
      if [[ -n "${SOURCE}" ]]; then
        echo "ERROR: only one image tar path or URL is allowed." >&2
        usage >&2
        exit 2
      fi
      SOURCE="$1"
      shift
      ;;
  esac
done

command_exists() {
  command -v "$1" >/dev/null 2>&1
}

origin_release_url() {
  local remote repo
  remote="$(git -C "${PROJECT_ROOT}" config --get remote.origin.url 2>/dev/null || true)"
  if [[ -z "${remote}" ]]; then
    return 1
  fi

  case "${remote}" in
    git@github.com:*)
      repo="${remote#git@github.com:}"
      ;;
    https://github.com/*)
      repo="${remote#https://github.com/}"
      ;;
    http://github.com/*)
      repo="${remote#http://github.com/}"
      ;;
    *)
      return 1
      ;;
  esac

  repo="${repo%.git}"
  if [[ -z "${repo}" || "${repo}" != */* ]]; then
    return 1
  fi
  printf 'https://github.com/%s/releases/latest/download/%s\n' "${repo}" "${ASSET_NAME}"
}

download_file() {
  local url="$1"
  local output="$2"

  mkdir -p "$(dirname "${output}")"
  if command_exists curl; then
    curl -fL --retry 3 --connect-timeout 30 -o "${output}.tmp" "${url}"
  elif command_exists wget; then
    wget -O "${output}.tmp" "${url}"
  else
    echo "ERROR: curl or wget is required to download ${url}" >&2
    exit 1
  fi
  mv "${output}.tmp" "${output}"
}

if ! command_exists docker; then
  echo "ERROR: docker command not found. Install Docker Desktop or Docker Engine first." >&2
  exit 1
fi

if ! docker info >/dev/null 2>&1; then
  echo "ERROR: Docker daemon is not running. Start Docker Desktop first." >&2
  exit 1
fi

if [[ "${FORCE}" != "1" ]] && docker image inspect "${IMAGE_NAME}" >/dev/null 2>&1; then
  echo "Docker image already installed: ${IMAGE_NAME}"
  echo "paper_factor will reuse it when FACTOR_CoSTEER_EXECUTION_BACKEND=docker."
  exit 0
fi

if [[ -z "${SOURCE}" ]]; then
  if [[ -n "${FACTOR_DOCKER_IMAGE_URL:-}" ]]; then
    SOURCE="${FACTOR_DOCKER_IMAGE_URL}"
  elif [[ -f "${CACHE_DIR}/${ASSET_NAME}" ]]; then
    SOURCE="${CACHE_DIR}/${ASSET_NAME}"
  elif [[ -n "${DEFAULT_IMAGE_URL}" ]]; then
    SOURCE="${DEFAULT_IMAGE_URL}"
  else
    SOURCE="$(origin_release_url || true)"
  fi
fi

if [[ -z "${SOURCE}" ]]; then
  echo "ERROR: no Docker image archive or release URL was provided." >&2
  echo "Pass a local tar file or a GitHub Release asset URL:" >&2
  echo "  bash scripts/install_factor_docker_image.sh release_assets/${ASSET_NAME}" >&2
  echo "  bash scripts/install_factor_docker_image.sh https://github.com/<owner>/<repo>/releases/download/<tag>/${ASSET_NAME}" >&2
  exit 2
fi

ARCHIVE="${SOURCE}"
case "${SOURCE}" in
  http://*|https://*)
    ARCHIVE="${CACHE_DIR}/${ASSET_NAME}"
    if [[ "${FORCE}" != "1" && -f "${ARCHIVE}" ]]; then
      echo "Using cached Docker image archive: ${ARCHIVE}"
    else
      echo "Downloading Docker image archive:"
      echo "  ${SOURCE}"
      download_file "${SOURCE}" "${ARCHIVE}"
    fi
    ;;
  *)
    if [[ ! -f "${ARCHIVE}" ]]; then
      echo "ERROR: Docker image archive not found: ${ARCHIVE}" >&2
      exit 1
    fi
    ;;
esac

echo "Loading Docker image from: ${ARCHIVE}"
docker load -i "${ARCHIVE}"

if docker image inspect "${IMAGE_NAME}" >/dev/null 2>&1; then
  echo "Docker image is ready: ${IMAGE_NAME}"
  echo "Use these settings for paper_factor:"
  echo "  DS_CODER_COSTEER_ENV_TYPE=docker"
  echo "  FACTOR_CoSTEER_EXECUTION_BACKEND=docker"
else
  echo "WARNING: docker load finished, but ${IMAGE_NAME} was not found." >&2
  echo "Check the loaded image tag with: docker images" >&2
  exit 1
fi
