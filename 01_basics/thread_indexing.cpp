#include <cstdio>
#include <hip/hip_runtime.h>

__global__ void indexing() {
    printf("blockIdx.x=%d threadIdx.x=%d\n", blockIdx.x, threadIdx.x);
}

int main() {
    indexing<<<2, 3>>>();
    hipDeviceSynchronize();
    return 0;
}