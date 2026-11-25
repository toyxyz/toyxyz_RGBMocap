#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
    printf 'Usage: %s LINUX_CHECKOUT APPIMAGETOOL TYPE2_RUNTIME OUTPUT_DIR\n' "$0" >&2
    exit 2
fi

packaging=$(cd -- "$(dirname -- "$0")" && pwd -P)
source=$(realpath -- "$1")
tool=$(realpath -- "$2")
runtime=$(realpath -- "$3")
mkdir -p -- "$4"
output=$(realpath -- "$4")
revision=$(git -C "$source" rev-parse --verify HEAD)
image="RGBMoCap-Linux-${revision:0:8}-x86_64.AppImage"
tmp=$(mktemp -d -- "$output/.build.XXXXXXXX")
trap 'rm -rf -- "$tmp"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# Match the upstream SHA-256 values for these versioned GitHub release assets.
printf '%s  %s\n' ed4ce84f0d9caff66f50bcca6ff6f35aae54ce8135408b3fa33abfc3cb384eb0 "$tool" \
    2fca8b443c92510f1483a883f60061ad09b46b978b2631c807cd873a47ec260d "$runtime" | sha256sum -c -

appdir="$tmp/RGBMoCap.AppDir"
mkdir -- "$appdir"
install -m 755 -- "$packaging/AppRun" "$appdir/AppRun"
install -m 644 -- "$packaging/rgbmocap.desktop" "$appdir/rgbmocap.desktop"
printf '%s\n' "$revision" > "$appdir/source-revision"

# Read committed bytes only. README demonstration videos are not app assets.
git -C "$source" archive --format=tar "$revision" -- \
    . ':(exclude,glob)docs/samples/*.mp4' > "$appdir/rgbmocap-source.tar"
git -C "$source" show "$revision:assets/rgbmocap-icon.png" > "$appdir/rgbmocap.png"
ln -s -- rgbmocap.png "$appdir/.DirIcon"

ARCH=x86_64 VERSION="${revision:0:8}" \
    "$tool" --appimage-extract-and-run --no-appstream \
    --runtime-file "$runtime" "$appdir" "$tmp/$image"
mv -- "$tmp/$image" "$output/$image"
cd -- "$output"
sha256sum -- "$image" > "$image.sha256"
du -h -- "$image"
