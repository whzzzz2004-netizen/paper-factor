#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${FACTOR_DOCKER_IMAGE:-local_factor_exec:latest}"
ASSET_NAME="${FACTOR_DOCKER_ASSET_NAME:-local_factor_exec_latest.tar.gz}"
DOCKERFILE_DIR="${FACTOR_DOCKERFILE_DIR:-rdagent/components/coder/factor_coder/docker}"
GHCR_IMAGE="${FACTOR_DOCKER_GHCR_IMAGE:-}"
ACR_IMAGE="${FACTOR_DOCKER_ACR_IMAGE:-}"
DEFAULT_IMAGE_URL="${FACTOR_DOCKER_DEFAULT_IMAGE_URL:-}"
IMAGE_URL="${FACTOR_DOCKER_IMAGE_URL:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CACHE_DIR="${FACTOR_DOCKER_CACHE_DIR:-${PROJECT_ROOT}/release_assets}"
FORCE="${FACTOR_DOCKER_FORCE_INSTALL:-0}"
SOURCE=""

usage() {
  cat <<EOF
Usage:
  bash scripts/install_factor_docker_image.sh [OPTIONS]

Install the factor execution Docker image from a registry, release asset,
or local build.

Options:
  --source ghcr       Pull from GitHub Container Registry (ghcr.io)
  --source acr        Pull from Alibaba Cloud ACR (registry.cn-hangzhou.aliyuncs.com)
  --source release    Download tar.gz from GitHub Releases
  --source build      Build locally from Dockerfile
  --source file PATH  Load from local tar.gz file
  --source url URL    Download from a direct URL
  --force             Reinstall even if the image already exists
  --show-sources      Print resolved registry paths and exit (no action)
  -h, --help          Show this help

If no --source is given, sources are tried in this order:
  1. Alibaba Cloud ACR (if FACTOR_DOCKER_ACR_IMAGE is set)
  2. GitHub Container Registry (ghcr.io)
  3. GitHub Releases tar.gz download
  4. Local cache in release_assets/

Environment variables:
  FACTOR_DOCKER_IMAGE           Image tag (default: ${IMAGE_NAME})
  FACTOR_DOCKER_GHCR_IMAGE      ghcr.io image path (auto-detected from git remote)
  FACTOR_DOCKER_ACR_IMAGE       Alibaba Cloud ACR image path
  FACTOR_DOCKER_IMAGE_URL       Direct download URL for tar.gz
  FACTOR_DOCKER_FORCE_INSTALL=1 Reinstall even if image exists
EOF
}

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

ghcr_image_name() {
  local remote repo repo_lower
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
  repo_lower="$(echo "${repo}" | tr '[:upper:]' '[:lower:]')"
  printf 'ghcr.io/%s/%s\n' "${repo_lower}" "${IMAGE_NAME%%:*}"
}

download_file() {
  local url="$1"
  local output="$2"

  mkdir -p "$(dirname "${output}")"
  echo "Downloading ${url} ..."
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

pull_from_registry() {
  local registry_image="$1"
  echo "Trying docker pull ${registry_image} ..."
  if docker pull "${registry_image}" 2>&1; then
    if [[ "${registry_image}" != "${IMAGE_NAME}" ]]; then
      echo "Tagging ${registry_image} as ${IMAGE_NAME}"
      docker tag "${registry_image}" "${IMAGE_NAME}"
    fi
    return 0
  fi
  return 1
}

# --- Parse arguments ---
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
    --source)
      shift
      case "${1:-}" in
        ghcr|release|build|acr)
          SOURCE="$1"
          shift
          ;;
        file)
          shift
          SOURCE="file:$1"
          shift
          ;;
        url)
          shift
          SOURCE="url:$1"
          shift
          ;;
        *)
          echo "ERROR: unknown --source value: $1" >&2
          echo "Valid values: ghcr, acr, release, build, file PATH, url URL" >&2
          exit 2
          ;;
      esac
      ;;
    --show-sources)
      SHOW_SOURCES=true
      shift
      ;;
    *)
      if [[ -n "${SOURCE}" ]]; then
        echo "ERROR: only one source is allowed." >&2
        exit 2
      fi
      SOURCE="$1"
      shift
      ;;
  esac
done

# --- Prerequisites ---
if ! command_exists docker; then
  echo "ERROR: docker command not found. Install Docker Desktop or Docker Engine first." >&2
  exit 1
fi

if ! docker info >/dev/null 2>&1; then
  echo "ERROR: Docker daemon is not running. Start Docker Desktop first." >&2
  exit 1
fi

# --- Show sources mode ---
if [[ "${SHOW_SOURCES:-false}" == "true" ]]; then
  echo "Resolved image sources (in priority order):"
  echo ""
  if [[ -n "${ACR_IMAGE}" ]]; then
    echo "  [acr]     ${ACR_IMAGE}"
  else
    echo "  [acr]     (not configured)"
  fi
  echo "  [ghcr]    $(ghcr_image_name || echo 'unable to determine')"
  echo "  [release] $(origin_release_url || echo 'unable to determine')"
  if [[ -f "${CACHE_DIR}/${ASSET_NAME}" ]]; then
    echo "  [cache]   ${CACHE_DIR}/${ASSET_NAME} (exists)"
  else
    echo "  [cache]   ${CACHE_DIR}/${ASSET_NAME} (not found)"
  fi
  echo "  [build]   ${DOCKERFILE_DIR}"
  exit 0
fi

# --- Already installed? ---
if [[ "${FORCE}" != "1" ]] && docker image inspect "${IMAGE_NAME}" >/dev/null 2>&1; then
  echo "Docker image already installed: ${IMAGE_NAME}"
  exit 0
fi

# --- Resolve source ---
if [[ -z "${SOURCE}" ]]; then
  # Auto mode: try each source in priority order

  # 1) Alibaba Cloud ACR (if configured, best for China)
  if [[ -n "${ACR_IMAGE}" ]]; then
    if pull_from_registry "${ACR_IMAGE}"; then
      echo "Installed from Alibaba Cloud ACR."
      exit 0
    fi
    echo "ACR pull failed, trying next source ..." >&2
  fi

  # 2) GitHub Container Registry
  GHCR="$(ghcr_image_name || true)"
  if [[ -n "${GHCR}" ]]; then
    if pull_from_registry "${GHCR}:latest"; then
      echo "Installed from GitHub Container Registry."
      exit 0
    fi
    echo "ghcr.io pull failed, trying next source ..." >&2
  fi

  # 3) Local cached tar.gz
  if [[ -f "${CACHE_DIR}/${ASSET_NAME}" ]]; then
    SOURCE="file:${CACHE_DIR}/${ASSET_NAME}"
  # 4) GitHub Releases (via derived URL or IMAGE_URL)
  elif [[ -n "${IMAGE_URL}" ]]; then
    SOURCE="url:${IMAGE_URL}"
  else
    RELEASE_URL="$(origin_release_url || true)"
    if [[ -n "${RELEASE_URL}" ]]; then
      SOURCE="url:${RELEASE_URL}"
    fi
  fi
fi

# --- Execute the chosen source ---
case "${SOURCE}" in
  file:*)
    ARCHIVE="${SOURCE#file:}"
    if [[ ! -f "${ARCHIVE}" ]]; then
      echo "ERROR: archive not found: ${ARCHIVE}" >&2
      exit 1
    fi
    echo "Loading Docker image from ${ARCHIVE} ..."
    docker load -i "${ARCHIVE}"
    ;;
  url:*)
    URL="${SOURCE#url:}"
    ARCHIVE="${CACHE_DIR}/${ASSET_NAME}"
    if [[ "${FORCE}" != "1" && -f "${ARCHIVE}" ]]; then
      echo "Using cached archive: ${ARCHIVE}"
    else
      download_file "${URL}" "${ARCHIVE}"
    fi
    echo "Loading Docker image from ${ARCHIVE} ..."
    docker load -i "${ARCHIVE}"
    ;;
  ghcr)
    GHCR="$(ghcr_image_name || true)"
    if [[ -z "${GHCR}" ]]; then
      echo "ERROR: could not determine ghcr.io image path." >&2
      exit 1
    fi
    if ! pull_from_registry "${GHCR}:latest"; then
      echo "ERROR: failed to pull from ghcr.io." >&2
      exit 1
    fi
    ;;
  acr)
    if [[ -z "${ACR_IMAGE}" ]]; then
      echo "ERROR: FACTOR_DOCKER_ACR_IMAGE is not set." >&2
      exit 1
    fi
    if ! pull_from_registry "${ACR_IMAGE}"; then
      echo "ERROR: failed to pull from ACR." >&2
      exit 1
    fi
    ;;
  build)
    echo "Building Docker image from ${DOCKERFILE_DIR} ..."
    docker build -t "${IMAGE_NAME}" "${PROJECT_ROOT}/${DOCKERFILE_DIR}"
    ;;
  "")
    echo "ERROR: no Docker image source available." >&2
    echo "Set FACTOR_COSTEER_FORCE_DOCKER_BUILD=1 in .env to build locally," >&2
    echo "or provide --source ghcr, --source file, or --source url." >&2
    exit 2
    ;;
  *)
    echo "ERROR: unknown source: ${SOURCE}" >&2
    exit 2
    ;;
esac

# --- Verify ---
if docker image inspect "${IMAGE_NAME}" >/dev/null 2>&1; then
  echo "Docker image is ready: ${IMAGE_NAME}"
else
  echo "WARNING: installation finished but ${IMAGE_NAME} was not found." >&2
  echo "Check with: docker images" >&2
  exit 1
fi
