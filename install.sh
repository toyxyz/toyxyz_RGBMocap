#!/usr/bin/env bash
# Linux counterpart of install.cmd (unofficial port). Safe to rerun; verified runtimes are kept.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
exec python3 -I -X utf8 -B tools/install_linux.py "$@"
