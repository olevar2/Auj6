import platform
import os

def get_os_info():
    return platform.system()

def is_windows():
    return platform.system() == 'Windows'

def get_platform_metadata():
    return {
        'os': platform.system(),
        'release': platform.release(),
        'version': platform.version(),
        'machine': platform.machine(),
        'processor': platform.processor()
    }
