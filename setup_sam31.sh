#!/bin/sh
# Optional SAM 3.1 overlay for the managed Linux ExMR runtime.
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
. "$root/tools/posix_install_guard.sh"
rgbmocap_require_posix_install "$root"
if [ "$(uname -s)" != Linux ] || [ "$(uname -m)" != x86_64 ]; then
    echo 'SAM 3.1 setup requires x86-64 Linux.' >&2
    exit 1
fi
python="$root/runtime_exmr/bin/python3.12"
if [ ! -x "$python" ]; then
    echo 'The ExMR runtime is missing. Run ./install.sh first.' >&2
    exit 1
fi
echo 'Optional SAM 3.1 setup. Close RGB MoCap before continuing.'
echo 'This setup reuses the existing ExMR Python and CUDA runtime.'
echo 'Review the SAM terms at https://github.com/facebookresearch/sam3/blob/f1647e8d0c1d280991f4d1b7d974c2704537fe66/LICENSE'
printf 'I reviewed the upstream terms. Continue? [y/N] '
IFS= read -r answer || exit 1
case "$answer" in
    [yY]) ;;
    *) exit 1 ;;
esac
exec "$python" -I -X utf8 -B "$root/tools/setup_sam31.py"
