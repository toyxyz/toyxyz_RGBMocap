#!/bin/sh
# Linux counterpart of install.cmd.
cd -- "$(dirname -- "$0")" || exit 1
status=0
sh tools/git_bootstrap.sh "$@" || status=$?
[ "$status" -eq 0 ] || echo "Installation failed. See the message above; rerun ./install.sh to retry."
exit "$status"
