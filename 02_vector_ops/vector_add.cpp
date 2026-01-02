#include <cstdio>
#include <hip/hip_runtime.h>

#define N 16

__global__ void add(float *a, float *b, float *c, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) c[id] = a[id] + b[id];
}

int main() {
    float h_a[N], h_b[N], h_c[N];

    for (int i = 0; i < N; i++) {
        h_a[i] = i;
        h_b[i] = 2*i;
    }

    float *d_a, *d_b, *d_c;
    hipMalloc(&d_a, sizeof(float)*N);
    hipMalloc(&d_b, sizeof(float)*N);
    hipMalloc(&d_c, sizeof(float)*N);

    hipMemcpy(d_a, h_a, sizeof(float)*N, hipMemcpyHostToDevice);
    hipMemcpy(d_b, h_b, sizeof(float)*N, hipMemcpyHostToDevice);

    add<<<1, N>>>(d_a, d_b, d_c, N);
    hipDeviceSynchronize();

    hipMemcpy(h_c, d_c, sizeof(float)*N, hipMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) printf("%.1f ", h_c[i]);

    hipFree(d_a); hipFree(d_b); hipFree(d_c);
    return 0;
}