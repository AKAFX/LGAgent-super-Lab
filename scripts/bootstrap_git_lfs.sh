#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION="3.7.1"
INSTALL_DIR="${ROOT_DIR}/.tools/git-lfs/v${VERSION}/bin"
GIT_LFS="${INSTALL_DIR}/git-lfs"

case "$(uname -s)-$(uname -m)" in
  Darwin-arm64)
    ARCHIVE="git-lfs-darwin-arm64-v${VERSION}.zip"
    SHA256="76260fb34f4ee622ff0a66b857e5954aa49c7e343a92e57a1ec4a760618c94b2"
    ;;
  Darwin-x86_64)
    ARCHIVE="git-lfs-darwin-amd64-v${VERSION}.zip"
    SHA256="b5b1b641c0648c83661fa9eda991cd3eff945264dabc2cdf411a80dfe7ec0970"
    ;;
  Linux-x86_64)
    ARCHIVE="git-lfs-linux-amd64-v${VERSION}.tar.gz"
    SHA256="1c0b6ee5200ca708c5cebebb18fdeb0e1c98f1af5c1a9cba205a4c0ab5a5ec08"
    ;;
  Linux-aarch64|Linux-arm64)
    ARCHIVE="git-lfs-linux-arm64-v${VERSION}.tar.gz"
    SHA256="73a9c90eeb4312133a63c3eaee0c38c019ea7bfa0953d174809d25b18588dd8d"
    ;;
  *)
    echo "Unsupported platform: $(uname -s)-$(uname -m)" >&2
    exit 1
    ;;
esac

if [[ ! -x "${GIT_LFS}" ]]; then
  TMP_DIR="$(mktemp -d)"
  trap 'rm -rf "${TMP_DIR}"' EXIT
  URL="https://github.com/git-lfs/git-lfs/releases/download/v${VERSION}/${ARCHIVE}"

  echo "Downloading Git LFS v${VERSION}..."
  curl --fail --location --silent --show-error "${URL}" --output "${TMP_DIR}/${ARCHIVE}"
  printf '%s  %s\n' "${SHA256}" "${TMP_DIR}/${ARCHIVE}" | shasum -a 256 --check

  mkdir -p "${INSTALL_DIR}"
  if [[ "${ARCHIVE}" == *.zip ]]; then
    unzip -q "${TMP_DIR}/${ARCHIVE}" -d "${TMP_DIR}/unpacked"
  else
    mkdir -p "${TMP_DIR}/unpacked"
    tar -xzf "${TMP_DIR}/${ARCHIVE}" -C "${TMP_DIR}/unpacked"
  fi
  EXTRACTED_GIT_LFS="$(find "${TMP_DIR}/unpacked" -type f -name git-lfs -print -quit)"
  if [[ -z "${EXTRACTED_GIT_LFS}" ]]; then
    echo "The downloaded archive does not contain git-lfs." >&2
    exit 1
  fi
  cp "${EXTRACTED_GIT_LFS}" "${GIT_LFS}"
  chmod +x "${GIT_LFS}"
fi

export PATH="${INSTALL_DIR}:${PATH}"
cd "${ROOT_DIR}"
git lfs install --local --force
git config --local filter.lfs.clean "${GIT_LFS} clean -- %f"
git config --local filter.lfs.smudge "${GIT_LFS} smudge -- %f"
git config --local filter.lfs.process "${GIT_LFS} filter-process"
git lfs pull "$@"
git lfs version
