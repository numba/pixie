from ctypes import (
    CDLL,
    c_char_p,
    c_void_p,
    c_size_t,
    POINTER,
    create_string_buffer,
    byref,
)
from ctypes.util import find_library

libc = CDLL(find_library("c"))

# Define the sysctlbyname function signature
# int sysctlbyname(const char *, void *, size_t *, void *, size_t);
libc.sysctlbyname.argtypes = [
    c_char_p,           # name
    c_void_p,           # oldp
    POINTER(c_size_t),  # oldlenp
    c_void_p,           # newp
    c_size_t,           # newlen
]


def sysctlbyname(name: bytes) -> bytes:
    """
    Wrapper around sysctlbyname on BSD-like, including Darwin.

    >>> result = sysctlbyname(b"machdep.cpu.brand_string\00")
    >>> print(result)
    """

    # Determine the size of the output buffer
    size = c_size_t(0)
    libc.sysctlbyname(name, None, byref(size), None, 0)
    # Create a buffer of the determined size
    buf = create_string_buffer(size.value)
    # Call sysctlbyname again, this time passing the buffer
    libc.sysctlbyname(name, buf, byref(size), None, 0)
    # Return the buffer content
    return buf.raw
