#include <stdio.h>
#include <cuda_runtime.h>

// CUDA kernel function (runs on GPU)
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    // Calculate global thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check bounds
    if (tid < n) {
        c[tid] = a[tid] + b[tid];
    }
}

// Helper function to check CUDA errors
void checkCudaError(cudaError_t error, const char* msg) {
    if (error != cudaSuccess) {
        printf("CUDA Error: %s - %s\n", msg, cudaGetErrorString(error));
        exit(1);
    }
}

int main() {
    const int N = 1000;  // Array size
    const int size = N * sizeof(float);
    
    // Host (CPU) arrays
    float *h_a, *h_b, *h_c;
    
    // Device (GPU) arrays
    float *d_a, *d_b, *d_c;
    
    // Allocate memory on host
    h_a = (float*)malloc(size);
    h_b = (float*)malloc(size);
    h_c = (float*)malloc(size);
    
    // Initialize host arrays
    printf("Initializing arrays...\n");
    for (int i = 0; i < N; i++) {
        h_a[i] = i * 1.0f;
        h_b[i] = i * 2.0f;
    }
    
    // Allocate memory on device (GPU)
    checkCudaError(cudaMalloc(&d_a, size), "Failed to allocate device memory for a");
    checkCudaError(cudaMalloc(&d_b, size), "Failed to allocate device memory for b");
    checkCudaError(cudaMalloc(&d_c, size), "Failed to allocate device memory for c");
    
    // Copy data from host to device
    printf("Copying data to GPU...\n");
    checkCudaError(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice), "Failed to copy a to device");
    checkCudaError(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice), "Failed to copy b to device");
    
    // Define grid and block dimensions
    int blockSize = 256;  // Number of threads per block
    int gridSize = (N + blockSize - 1) / blockSize;  // Number of blocks
    
    printf("Launching kernel with %d blocks of %d threads each...\n", gridSize, blockSize);
    
    // Launch kernel
    vectorAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
    
    // Check for kernel launch errors
    checkCudaError(cudaGetLastError(), "Kernel launch failed");
    
    // Wait for GPU to finish
    checkCudaError(cudaDeviceSynchronize(), "Device synchronization failed");
    
    // Copy result from device to host
    printf("Copying result back to CPU...\n");
    checkCudaError(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost), "Failed to copy result to host");
    
    // Verify result (check first 10 elements)
    printf("\nVerifying results (first 10 elements):\n");
    bool success = true;
    for (int i = 0; i < 10; i++) {
        float expected = h_a[i] + h_b[i];
        printf("a[%d] + b[%d] = %.1f + %.1f = %.1f (expected: %.1f)\n", 
               i, i, h_a[i], h_b[i], h_c[i], expected);
        if (h_c[i] != expected) {
            success = false;
        }
    }
    
    if (success) {
        printf("\n✓ All results are correct!\n");
    } else {
        printf("\n✗ Some results are incorrect!\n");
    }
    
    // Clean up memory
    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    printf("\nMemory cleaned up. Program finished.\n");
    
    return 0;
}