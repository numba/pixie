#include<math.h>

void __attribute__((always_inline)) f(double* x, double* result){
    *result = cos(*x) + 1;
}
