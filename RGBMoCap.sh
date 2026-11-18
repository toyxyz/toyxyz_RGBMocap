#!/bin/sh
# Launch RGB MoCap on Linux. Prepare the environments first: python3 tools/setup_linux.py
set -eu
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
    echo "RGB MoCap is not set up. Run: python3 tools/setup_linux.py" >&2
    exit 1
fi
exec .venv/bin/python -I -X utf8 -u -B launch_linux.py "$@"
