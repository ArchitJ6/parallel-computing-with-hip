#include <cstdio>
#include <hip/hip_runtime.h>

#define PIXELS 8

__global__ void grayscale(unsigned char *input, unsigned char *output, int pixels) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < pixels) {
        int i = id * 3;
        unsigned char r = input[i];
        unsigned char g = input[i+1];
        unsigned char b = input[i+2];

        output[id] = 0.299f*r + 0.587f*g + 0.114f*b;
    }
}

int main() {
    unsigned char h_input[PIXELS * 3] = {
        255,0,0, 0,255,0, 0,0,255, 255,255,0,
        255,0,255, 0,255,255, 128,128,128, 50,100,150
    };

    unsigned char h_output[PIXELS];

    unsigned char *d_input, *d_output;

    hipMalloc(&d_input, sizeof(unsigned char)*PIXELS*3);
    hipMalloc(&d_output, sizeof(unsigned char)*PIXELS);

    hipMemcpy(d_input, h_input, sizeof(unsigned char)*PIXELS*3, hipMemcpyHostToDevice);

    grayscale<<<1, PIXELS>>>(d_input, d_output, PIXELS);
    hipDeviceSynchronize();

    hipMemcpy(h_output, d_output, sizeof(unsigned char)*PIXELS, hipMemcpyDeviceToHost);

    printf("Grayscale:\n");
    for (int i = 0; i < PIXELS; i++) printf("%d ", h_output[i]);

    hipFree(d_input);
    hipFree(d_output);

    return 0;
}