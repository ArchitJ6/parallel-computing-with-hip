#include <cstdio>

#define N 1024

__global__ void dot(float *a, float *b, float *res, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < n) {
        atomicAdd(res, a[id] * b[id]);
    }
}

int main() {
    float h_a[N], h_b[N], h_res = 0;

    for (int i = 0; i < N; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    float *d_a, *d_b, *d_res;
    cudaMalloc(&d_a, sizeof(float)*N);
    cudaMalloc(&d_b, sizeof(float)*N);
    cudaMalloc(&d_res, sizeof(float));

    cudaMemcpy(d_a, h_a, sizeof(float)*N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeof(float)*N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_res, &h_res, sizeof(float), cudaMemcpyHostToDevice);

    dot<<<(N+255)/256, 256>>>(d_a, d_b, d_res, N);
    cudaDeviceSynchronize();

    cudaMemcpy(&h_res, d_res, sizeof(float), cudaMemcpyDeviceToHost);

    printf("Dot Product = %f\n", h_res);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_res);
    return 0;
}