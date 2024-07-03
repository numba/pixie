// This file contains code for the derivative of the objective function
// `result = cos(x)` as defined in objective_function.pyx.
#include <math.h>

void __attribute__((always_inline)) _Z4dfdxPdS_(double* x, double* result){
    *result = -sin(*x);
}
