#!/usr/bin/env bash
# Linux counterpart of update.cmd (unofficial port): replay the local Linux
# changes on top of upstream main, then reconcile the runtimes with the new pins.
# Close the app first. If upstream touched a patched file, git reports the conflict.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
git pull --rebase --autostash origin main
exec ./install.sh "$@"
