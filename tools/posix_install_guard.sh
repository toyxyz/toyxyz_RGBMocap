# Sourced by the Linux installer and updater before they modify setup state.
rgbmocap_require_posix_install() {
    for marker in runtime/python.exe runtime_exmr/python.exe .setup/bootstrap-git/uv.exe; do
        if [ -e "$1/$marker" ] || [ -L "$1/$marker" ]; then
            echo "This folder contains a Windows RGB MoCap installation. Use a separate Linux checkout with its own runtimes." >&2
            return 1
        fi
    done
    return 0
}
