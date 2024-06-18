"""
This file uses constant values from darwin.

Portions Copyright (c) 1999-2007 Apple Inc.  All Rights Reserved.

This file contains Original Code and/or Modifications of Original Code as
defined in and that are subject to the Apple Public Source License Version 2.0
(the 'License').  You may not use this file except in compliance with the
License.  Please obtain a copy of the License at
http://www.opensource.apple.com/apsl/ and read it before using this file.

The Original Code and all software distributed under the License are distributed
on an 'AS IS' basis, WITHOUT WARRANTY OF ANY KIND, EITHER EXPRESS OR IMPLIED,
AND APPLE HEREBY DISCLAIMS ALL SUCH WARRANTIES, INCLUDING WITHOUT LIMITATION,
ANY WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, QUIET
ENJOYMENT OR NON-INFRINGEMENT.  Please see the License for the specific language
governing rights and limitations under the License.
"""

# commpage address
# https://github.com/apple-oss-distributions/xnu/blob/94d3b452840153a99b38a3a9659680b2a006908e/osfmk/arm/cpu_capabilities.h#L162  # noqa 501
commpage_addr = 0x0000000FFFFFC000


# commpage offset for the cpu_family
# https://github.com/apple-oss-distributions/xnu/blob/94d3b452840153a99b38a3a9659680b2a006908e/osfmk/arm/cpu_capabilities.h#L332  # noqa 501
commpage_cpu_family_offset = 0x80


cpu_families = dict(
    # Reference: https://github.com/apple-oss-distributions/xnu/blob/94d3b452840153a99b38a3a9659680b2a006908e/osfmk/mach/machine.h#L428-L444  # noqa 501
    APPLE_M1=0x1b588bb3,  # M1 is FIRESTORM_ICESTORM
    # From running sysctl hw.cpufamily on a M2
    # hw.cpufamily: -634136515
    APPLE_M2=0xda33d83d,  # M2 is AVALANCHE_BLIZZARD
)
