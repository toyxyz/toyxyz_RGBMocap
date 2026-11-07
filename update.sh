#!/bin/sh
# Linux counterpart of update.cmd. The braces make sh read the whole body before
# Git can replace this running script; git_update.py reports the outcome.
{
root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
python="$root/runtime/bin/python3.10"
if [ ! -x "$python" ]; then
    echo "Run ./install.sh first."
    exit 1
fi
cd -- "$root" || exit 1
"$python" -I -X utf8 -B "$root/tools/git_update.py" "$@"
exit $?
}
