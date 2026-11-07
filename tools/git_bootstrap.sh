#!/bin/sh
# POSIX counterpart of git_bootstrap.ps1: verified uv, private CPython, then the shared Git installer.
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
setup="$root/.setup"
resources="$setup/bootstrap-git"
lock="$root/installer/runtime-lock-linux.json"
mkdir -p "$resources"
field() { sed -n 's/^ *"'"$1"'": *"\([^"]*\)".*/\1/p' "$lock" | head -n 1; }
uv_version=$(field uv_version)
uv_sha256=$(field uv_sha256)
[ -n "$uv_version" ] && [ -n "$uv_sha256" ] || { echo "The Linux runtime lock has no uv pin." >&2; exit 1; }
uv="$resources/uv"
digest() { sha256sum "$1" | cut -d' ' -f1; }
if [ ! -f "$uv" ] || [ "$(digest "$uv")" != "$uv_sha256" ]; then
    archive="$resources/uv-download.tar.gz"
    extract="$resources/uv-extracted"
    url="https://releases.astral.sh/github/uv/releases/download/$uv_version/uv-x86_64-unknown-linux-gnu.tar.gz"
    if [ ! -f "$archive" ]; then
        echo 'Downloading the verified uv bootstrap...'
        curl -fL --proto '=https' --tlsv1.2 -o "$archive.partial" "$url"
        mv "$archive.partial" "$archive"
    fi
    rm -rf "$extract" && mkdir -p "$extract"
    tar -xzf "$archive" -C "$extract"
    candidate="$extract/uv-x86_64-unknown-linux-gnu/uv"
    if [ ! -f "$candidate" ] || [ "$(digest "$candidate")" != "$uv_sha256" ]; then
        echo 'The uv bootstrap hash does not match the pinned version. Remove the downloaded bootstrap archive and retry.' >&2
        exit 1
    fi
    cp "$candidate" "$uv" && chmod 700 "$uv"
fi
python_dir="$setup/python"
export UV_PYTHON_INSTALL_DIR="$python_dir" UV_CACHE_DIR="$setup/uv-cache" UV_NO_CONFIG=1 PYTHONUTF8=1
"$uv" python install 3.10.22 --install-dir "$python_dir" --no-bin --no-config --cache-dir "$UV_CACHE_DIR"
exec "$python_dir/cpython-3.10.22-linux-x86_64-gnu/bin/python3.10" -I -X utf8 -B "$root/tools/git_managed_install.py" "$@"
