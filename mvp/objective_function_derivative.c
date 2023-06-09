#include<math.h>

void __attribute__((always_inline)) dfdx(double* x, double* result){
    *result = -sin(*x);
}
