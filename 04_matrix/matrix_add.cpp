#include <cstdio>
#include <hip/hip_runtime.h>

#define N 4

__global__ void mat_add(float *A, float *B, float *C, int w) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    if (r < w && c < w) {
        int idx = r * w + c;
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    float h_A[N*N], h_B[N*N], h_C[N*N];

    for (int i = 0; i < N*N; i++) {
        h_A[i] = i;
        h_B[i] = i;
    }

    float *d_A, *d_B, *d_C;
    hipMalloc(&d_A, sizeof(float)*N*N);
    hipMalloc(&d_B, sizeof(float)*N*N);
    hipMalloc(&d_C, sizeof(float)*N*N);

    hipMemcpy(d_A, h_A, sizeof(float)*N*N, hipMemcpyHostToDevice);
    hipMemcpy(d_B, h_B, sizeof(float)*N*N, hipMemcpyHostToDevice);

    dim3 threads(2,2);
    dim3 blocks((N+1)/2, (N+1)/2);

    mat_add<<<blocks, threads>>>(d_A, d_B, d_C, N);
    hipDeviceSynchronize();

    hipMemcpy(h_C, d_C, sizeof(float)*N*N, hipMemcpyDeviceToHost);

    printf("Matrix Add:\n");
    for (int i = 0; i < N*N; i++) printf("%.1f ", h_C[i]);

    hipFree(d_A); hipFree(d_B); hipFree(d_C);
    return 0;
}