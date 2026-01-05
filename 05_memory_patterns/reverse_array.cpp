#include <cstdio>
#include <hip/hip_runtime.h>

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
    hipMalloc(&d_a, sizeof(float)*N);
    hipMalloc(&d_b, sizeof(float)*N);

    hipMemcpy(d_a, h_a, sizeof(float)*N, hipMemcpyHostToDevice);

    reverse<<<1, N>>>(d_a, d_b, N);
    hipDeviceSynchronize();

    hipMemcpy(h_b, d_b, sizeof(float)*N, hipMemcpyDeviceToHost);

    printf("Reversed:\n");
    for (int i = 0; i < N; i++) printf("%.1f ", h_b[i]);

    hipFree(d_a); hipFree(d_b);
    return 0;
}