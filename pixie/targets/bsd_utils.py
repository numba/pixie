from ctypes import CDLL, c_char_p, c_void_p, create_string_buffer, byref, c_uint
from ctypes.util import find_library

libc = CDLL(find_library("c"))

# Define the sysctlbyname function signature
libc.sysctlbyname.argtypes = [
    c_char_p,  # name
    c_void_p,  # oldp
    c_void_p,  # oldlenp
    c_void_p,  # newp
    c_void_p,  # newlenp
]


def sysctlbyname(name: bytes) -> bytes:
    """
    Wrapper around sysctlbyname on BSD-like, including Darwin.

    >>> result = sysctlbyname(b"machdep.cpu.brand_string\00")
    >>> print(result)
    """

    # Determine the size of the output buffer
    size = c_uint(0)
    libc.sysctlbyname(name, None, byref(size), None, byref(c_uint(0)))
    # Create a buffer of the determined size
    buf = create_string_buffer(size.value)
    # Call sysctlbyname again, this time passing the buffer
    libc.sysctlbyname(name, buf, byref(size), None, None)
    # Return the buffer content
    return buf.raw
