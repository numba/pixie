# Example 1 explores the `__PIXIE__` module level dictionary.
from pprint import pprint

import objective_function

pixied = objective_function.__PIXIE__
print(f"__PIXIE__ dictionary keys: {pixied.keys()}")
print(f"Available ISAs: {pixied['available_isas']}")

pixied['bitcode']  # this is the LLVM bitcode

print(f"Selected ISA: {pixied['selected_isa']}")
print(f"Symbols: {pixied['symbols'].keys()}")
print("Symbol 'f':")
pprint(pixied['symbols']['f'])
