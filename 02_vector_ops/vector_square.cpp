#include <cstdio>
#include <hip/hip_runtime.h>

#define N 16

__global__ void square(float *a, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) a[id] *= a[id];
}

int main() {
    float h_a[N];
    for (int i = 0; i < N; i++) h_a[i] = i;

    float *d_a;
    hipMalloc(&d_a, sizeof(float)*N);

    hipMemcpy(d_a, h_a, sizeof(float)*N, hipMemcpyHostToDevice);

    square<<<1, N>>>(d_a, N);
    hipDeviceSynchronize();

    hipMemcpy(h_a, d_a, sizeof(float)*N, hipMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) printf("%.1f ", h_a[i]);

    hipFree(d_a);
    return 0;
}