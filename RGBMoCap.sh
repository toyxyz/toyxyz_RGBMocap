#!/usr/bin/env bash
# Linux counterpart of RGBMoCap.cmd (unofficial port).
cd "$(dirname "$(readlink -f "$0")")"
if [ ! -x runtime/python.exe ]; then
  echo "RGB MoCap is not installed. Run ./install.sh first."
  exit 1
fi
# Qt's xcb platform plugin is the tested path; the 3D viewport needs desktop OpenGL 4.3.
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"
exec runtime/python.exe -I -X utf8 -u -B launch_source.py "$@"
