#include <stdio.h>
#include <stdlib.h>
#include <openacc.h>
#include <time.h>

#define N 512
#define TILE_WIDTH 16

// Tiled matrix multiplication using OpenACC
void matrixMulOpenACC_Tiled(int *a, int *b, int *c) {
    #pragma acc data copyin(a[0:N*N], b[0:N*N]) copyout(c[0:N*N])
    {
        #pragma acc parallel loop tile(TILE_WIDTH, TILE_WIDTH)
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                int sum = 0;
                for (int k = 0; k < N; k++) {
                    sum += a[i * N + k] * b[k * N + j];
                }
                c[i * N + j] = sum;
            }
        }
    }
}

int main() {
    int *a, *b, *c;
    size_t size = N * N * sizeof(int);

    a = (int*)malloc(size);
    b = (int*)malloc(size);
    c = (int*)malloc(size);

    // Initialize matrices
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            a[i * N + j] = i;  
            b[i * N + j] = j + 2;  
            c[i * N + j] = 0;  
        }
    }

    // Timing
    clock_t start_time = clock();
    matrixMulOpenACC_Tiled(a, b, c);
    clock_t end_time = clock();
    double time_spent = (double)(end_time - start_time) / CLOCKS_PER_SEC;

    printf("OpenACC Tiled Matrix Multiplication time: %f seconds\n", time_spent);

    // Verify the output
    int valid = 1;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (c[i * N + j] != i * (j + 2) * N) {
                printf("Error: Incorrect computation at element [%d][%d]\n", i, j);
                valid = 0;
                break;
            }
        }
        if (!valid) break;
    }
    if (valid) {
        printf("Success: All values are computed correctly.\n");
    }

    free(a);
    free(b);
    free(c);

    return 0;
}
