"""macOS presentation of the unbundled app; standard library only."""
import ctypes
import ctypes.util


def set_app_name(name):
    """Show `name` in the menu bar instead of the interpreter's file name.

    Must run before QApplication exists. Failures are ignored: the name is cosmetic.
    """
    try:
        objc = ctypes.CDLL(ctypes.util.find_library('objc'))
        ctypes.CDLL(ctypes.util.find_library('Foundation'))
        pointer = ctypes.c_void_p
        objc.objc_getClass.restype = objc.sel_registerName.restype = pointer
        objc.objc_getClass.argtypes = objc.sel_registerName.argtypes = [ctypes.c_char_p]

        def send(target, selector, *arguments, types=None):
            # objc_msgSend is not variadic on arm64: every call needs its exact signature.
            call = ctypes.CFUNCTYPE(pointer, pointer, pointer, *(types or [pointer] * len(arguments)))(('objc_msgSend', objc))
            return call(target, objc.sel_registerName(selector), *arguments)

        def string(text):
            return send(objc.objc_getClass(b'NSString'), b'stringWithUTF8String:', text.encode('utf-8'), types=[ctypes.c_char_p])

        bundle = send(objc.objc_getClass(b'NSBundle'), b'mainBundle')
        info = send(bundle, b'localizedInfoDictionary') or send(bundle, b'infoDictionary')
        if info:
            send(info, b'setObject:forKey:', string(name), string('CFBundleName'))
    except (OSError, AttributeError, TypeError, ValueError):
        pass
