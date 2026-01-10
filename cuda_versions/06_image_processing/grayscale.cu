#include <cstdio>
#include <cuda_runtime.h>

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

    cudaMalloc(&d_input, sizeof(unsigned char)*PIXELS*3);
    cudaMalloc(&d_output, sizeof(unsigned char)*PIXELS);

    cudaMemcpy(d_input, h_input, sizeof(unsigned char)*PIXELS*3, cudaMemcpyHostToDevice);

    grayscale<<<1, PIXELS>>>(d_input, d_output, PIXELS);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, sizeof(unsigned char)*PIXELS, cudaMemcpyDeviceToHost);

    printf("Grayscale:\n");
    for (int i = 0; i < PIXELS; i++) printf("%d ", h_output[i]);

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}