#include <cstdio>
#include <hip/hip_runtime.h>

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
    hipMalloc(&d_a, sizeof(float)*N);

    hipMemcpy(d_a, h_a, sizeof(float)*N, hipMemcpyHostToDevice);

    stride<<<1, N>>>(d_a, 2);
    hipDeviceSynchronize();

    hipMemcpy(h_a, d_a, sizeof(float)*N, hipMemcpyDeviceToHost);

    printf("Strided Access:\n");
    for (int i = 0; i < N; i++) printf("%.1f ", h_a[i]);

    hipFree(d_a);
    return 0;
}