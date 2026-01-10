#include <cstdio>
#include <cuda_runtime.h>

#define N 16

__global__ void stride(float *a, int stride) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    int idx = id * stride;
    if (idx < N) {
        a[idx] *= 2;
    }
}

int main() {
    float h_a[N];
    for (int i = 0; i < N; i++) h_a[i] = i;

    float *d_a;
    cudaMalloc(&d_a, sizeof(float)*N);

    cudaMemcpy(d_a, h_a, sizeof(float)*N, cudaMemcpyHostToDevice);

    stride<<<1, N>>>(d_a, 2);
    cudaDeviceSynchronize();

    cudaMemcpy(h_a, d_a, sizeof(float)*N, cudaMemcpyDeviceToHost);

    printf("Strided Access Result:\n");
    for (int i = 0; i < N; i++) printf("%.1f ", h_a[i]);

    cudaFree(d_a);
    return 0;
}