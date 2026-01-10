#include <cstdio>
#include <cuda_runtime.h>

#define N 10

__global__ void reverse(float *a, float *b, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < n) {
        b[n - id - 1] = a[id];
    }
}

int main() {
    float h_a[N], h_b[N];

    for (int i = 0; i < N; i++) h_a[i] = i;

    float *d_a, *d_b;
    cudaMalloc(&d_a, sizeof(float)*N);
    cudaMalloc(&d_b, sizeof(float)*N);

    cudaMemcpy(d_a, h_a, sizeof(float)*N, cudaMemcpyHostToDevice);

    reverse<<<1, N>>>(d_a, d_b, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_b, d_b, sizeof(float)*N, cudaMemcpyDeviceToHost);

    printf("Reversed:\n");
    for (int i = 0; i < N; i++) printf("%.1f ", h_b[i]);

    cudaFree(d_a); cudaFree(d_b);
    return 0;
}