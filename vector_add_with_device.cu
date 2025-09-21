#include <stdio.h>
#include <stdlib.h>
#include <time.h>  // Add this header

// CUDA kernel function (runs on GPU)
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    // Calculate global thread ID
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    
    // Check bounds
    if (i< n) {
        // printf("Triggering %d\n", i);
        c[i] = a[i] + b[i];
    }
}

int main() {
    clock_t start, end;
    double cpu_time_used;
    
    start = clock();  
    
    const int N = 100000000;
    const int size = N * sizeof(float);

    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;

    printf("Initializing memory ...\n");
    h_a = (float*)malloc(size);
    h_b = (float*)malloc(size);
    h_c = (float*)malloc(size);

    printf("Assigning values into arrays...\n");
    for (int i = 0; i < N; i++) {
        h_a[i] = i * 1.0f;
        h_b[i] = i * 2.0f;
    }

    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    float gridSize = ceil(N/1024.0);
    int blockSize = 1024;
    vectorAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);

    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    end = clock();  
    
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Total execution time: %f seconds\n", cpu_time_used);
    
    free(h_a);
    free(h_b);
    free(h_c);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    return 0;
}