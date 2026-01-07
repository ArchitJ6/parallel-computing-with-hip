#include <cstdio>

__global__ void hello() {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Thread ID: %d\n", id);
}

int main() {
    hello<<<2, 4>>>();
    cudaDeviceSynchronize();
    return 0;
}