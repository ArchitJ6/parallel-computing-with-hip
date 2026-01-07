#include <cstdio>

__global__ void indexing() {
    printf("blockIdx.x=%d threadIdx.x=%d\n", blockIdx.x, threadIdx.x);
}

int main() {
    indexing<<<2, 3>>>();
    cudaDeviceSynchronize();
    return 0;
}