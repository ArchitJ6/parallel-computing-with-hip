#include <cstdio>
#include <hip/hip_runtime.h>

__global__ void hello() {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Thread ID: %d\n", id);
}

int main() {
    hello<<<2, 4>>>(); // 8 threads
    hipDeviceSynchronize();
    return 0;
}