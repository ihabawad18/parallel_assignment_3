#include <stdio.h>
#include <cuda_runtime.h>
#include <ctime>

#define N  512  
#define TILE_WIDTH 16  

__global__ void matrixMulGPU_Basic(int *a, int *b, int *c) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;

    if (row < N && col < N) {
        for (int k = 0; k < N; ++k) {
            sum += a[row * N + k] * b[k * N + col];
        }
        c[row * N + col] = sum;
    }
}

__global__ void matrixMulGPU_Tiled(int *a, int *b, int *c) {
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    int temp = 0;

    __shared__ int tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ int tileB[TILE_WIDTH][TILE_WIDTH];

    for (int p = 0; p < (N / TILE_WIDTH); ++p) {
        tileA[threadIdx.y][threadIdx.x] = a[row * N + (p * TILE_WIDTH + threadIdx.x)];
        tileB[threadIdx.y][threadIdx.x] = b[(p * TILE_WIDTH + threadIdx.y) * N + col];
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            temp += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }
        __syncthreads();
    }
    c[row * N + col] = temp;
}

void matrixMulCPU(int *a, int *b, int *c) {
    for (int row = 0; row < N; ++row) {
        for (int col = 0; col < N; ++col) {
            int sum = 0;
            for (int k = 0; k < N; ++k) {
                sum += a[row * N + k] * b[k * N + col];
            }
            c[row * N + col] = sum;
        }
    }
}

int main() {
    int *a, *b, *c_cpu, *c_gpu_basic, *c_gpu_tiled;
    int size = N * N * sizeof(int);

    cudaMallocManaged(&a, size);
    cudaMallocManaged(&b, size);
    cudaMallocManaged(&c_cpu, size);
    cudaMallocManaged(&c_gpu_basic, size);
    cudaMallocManaged(&c_gpu_tiled, size);

    for (int row = 0; row < N; ++row) {
        for (int col = 0; col < N; ++col) {
            a[row * N + col] = row;
            b[row * N + col] = col + 2;
            c_cpu[row * N + col] = 0;
            c_gpu_basic[row * N + col] = 0;
            c_gpu_tiled[row * N + col] = 0;
        }
    }

    dim3 threads_per_block(TILE_WIDTH, TILE_WIDTH);
    dim3 num_blocks((N + threads_per_block.x - 1) / threads_per_block.x, 
                    (N + threads_per_block.y - 1) / threads_per_block.y);

    // Time GPU (Basic)
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    matrixMulGPU_Basic<<<num_blocks, threads_per_block>>>(a, b, c_gpu_basic);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("GPU Basic time: %f ms\n", milliseconds);

    // Time GPU (Tiled)
    cudaEventRecord(start);
    matrixMulGPU_Tiled<<<num_blocks, threads_per_block>>>(a, b, c_gpu_tiled);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("GPU Tiled time: %f ms\n", milliseconds);

    // Time CPU
    clock_t start_cpu = clock();
    matrixMulCPU(a, b, c_cpu);
    clock_t end_cpu = clock();
    double cpu_time_used = ((double) (end_cpu - start_cpu)) / CLOCKS_PER_SEC * 1000.0; // Convert to milliseconds
    printf("CPU time: %f ms\n", cpu_time_used);

    // Compare results
    for (int i = 0; i < N * N; i++) {
        if (c_cpu[i] != c_gpu_basic[i]) {
            printf("Mismatch found in Basic GPU implementation at index %d\n", i);
            break;
        }
        if (c_cpu[i] != c_gpu_tiled[i]) {
            printf("Mismatch found in Tiled GPU implementation at index %d\n", i);
            break;
        }
    }

    // Free memory
    cudaFree(a); cudaFree(b);
    cudaFree(c_cpu); cudaFree(c_gpu_basic); cudaFree(c_gpu_tiled);

    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
