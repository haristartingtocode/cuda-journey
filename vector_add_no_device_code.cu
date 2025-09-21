#include <stdio.h>
#include <stdlib.h>
#include <time.h>  // Add this header



int main() {
    clock_t start, end;
    double cpu_time_used;
    
    start = clock();  // Start timing
    
    const int N = 100000000;
    const int size = N * sizeof(float);

    float *h_a, *h_b, *h_c;

    printf("Initializing memory ...\n");
    h_a = (float*)malloc(size);
    h_b = (float*)malloc(size);
    h_c = (float*)malloc(size);

    printf("Assigning values into arrays...\n");
    for (int i = 0; i < N; i++) {
        h_a[i] = i * 1.0f;
        h_b[i] = i * 2.0f;
    }

    for (int i = 0; i < N; i++) {
        h_c[i] = h_a[i] + h_b[i];
    }

    end = clock();  // End timing
    
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Total execution time: %f seconds\n", cpu_time_used);
    
    // Don't forget to free memory
    free(h_a);
    free(h_b);
    free(h_c);
    
    return 0;
}