#include <cstdio>

#define N 16

__global__ void square(float *a, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) a[id] *= a[id];
}

int main() {
    float h_a[N];
    for (int i = 0; i < N; i++) h_a[i] = i;

    float *d_a;
    cudaMalloc(&d_a, sizeof(float)*N);

    cudaMemcpy(d_a, h_a, sizeof(float)*N, cudaMemcpyHostToDevice);

    square<<<1, N>>>(d_a, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_a, d_a, sizeof(float)*N, cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) printf("%.1f ", h_a[i]);

    cudaFree(d_a);
    return 0;
}