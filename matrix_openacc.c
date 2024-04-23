#include <stdio.h>
#include <stdlib.h>
#include <openacc.h>
#include <time.h>

#define N 512

// Basic matrix multiplication using OpenACC
void matrixMulOpenACC(int *a, int *b, int *c) {
    #pragma acc data copyin(a[0:N*N], b[0:N*N]) copyout(c[0:N*N])
    {
        #pragma acc kernels
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

// CPU matrix multiplication
void matrixMulCPU(int *a, int *b, int *c) {
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

int main() {
    int *a, *b, *c, *c_cpu;
    size_t size = N * N * sizeof(int);
    a = (int*)malloc(size);
    b = (int*)malloc(size);
    c = (int*)malloc(size);
    c_cpu = (int*)malloc(size);

    // Initialize matrices
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            a[i * N + j] = i;
            b[i * N + j] = j + 2;
            c[i * N + j] = 0;
            c_cpu[i * N + j] = 0;
        }
    }

    clock_t start_cpu = clock();
    matrixMulCPU(a, b, c_cpu);
    clock_t end_cpu = clock();
    double cpu_time_used = ((double) (end_cpu - start_cpu)) / CLOCKS_PER_SEC;
    printf("CPU time: %f seconds\n", cpu_time_used);

    clock_t start_acc = clock();
    matrixMulOpenACC(a, b, c);
    clock_t end_acc = clock();
    double acc_time_used = ((double) (end_acc - start_acc)) / CLOCKS_PER_SEC;
    printf("OpenACC Basic time: %f seconds\n", acc_time_used);

    free(a);
    free(b);
    free(c);
    free(c_cpu);

    return 0;
}
