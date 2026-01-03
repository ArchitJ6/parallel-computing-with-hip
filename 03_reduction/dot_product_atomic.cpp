#include <cstdio>
#include <hip/hip_runtime.h>

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
    hipMalloc(&d_a, sizeof(float)*N);
    hipMalloc(&d_b, sizeof(float)*N);
    hipMalloc(&d_res, sizeof(float));

    hipMemcpy(d_a, h_a, sizeof(float)*N, hipMemcpyHostToDevice);
    hipMemcpy(d_b, h_b, sizeof(float)*N, hipMemcpyHostToDevice);
    hipMemcpy(d_res, &h_res, sizeof(float), hipMemcpyHostToDevice);

    dot<<<(N+255)/256, 256>>>(d_a, d_b, d_res, N);
    hipDeviceSynchronize();

    hipMemcpy(&h_res, d_res, sizeof(float), hipMemcpyDeviceToHost);

    printf("Dot Product = %f\n", h_res);

    hipFree(d_a); hipFree(d_b); hipFree(d_res);
    return 0;
}