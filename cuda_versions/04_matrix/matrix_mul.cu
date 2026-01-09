#include <cstdio>
#include <cuda_runtime.h>

#define N 4

__global__ void mat_mul(float *A, float *B, float *C, int w) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    if (r < w && c < w) {
        float sum = 0;
        for (int k = 0; k < w; k++) {
            sum += A[r*w + k] * B[k*w + c];
        }
        C[r*w + c] = sum;
    }
}

int main() {
    float h_A[N*N], h_B[N*N], h_C[N*N];

    for (int i = 0; i < N*N; i++) {
        h_A[i] = 1;
        h_B[i] = 1;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeof(float)*N*N);
    cudaMalloc(&d_B, sizeof(float)*N*N);
    cudaMalloc(&d_C, sizeof(float)*N*N);

    cudaMemcpy(d_A, h_A, sizeof(float)*N*N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeof(float)*N*N, cudaMemcpyHostToDevice);

    dim3 threads(2,2);
    dim3 blocks((N+1)/2, (N+1)/2);

    mat_mul<<<blocks, threads>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, sizeof(float)*N*N, cudaMemcpyDeviceToHost);

    printf("Matrix Mul Result:\n");
    for (int i = 0; i < N*N; i++) {
        printf("%.1f ", h_C[i]);
    }

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}