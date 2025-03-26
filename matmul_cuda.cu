#include <stdio.h>
#include <cuda_runtime.h>

#define N 1024  // Tamaño de la matriz (N x N)
#define TILE_SIZE 32  // Tamaño del bloque de hilos

#define cudaCheckError() {                                                        \
    cudaError_t e = cudaGetLastError();                                            \
    if (e != cudaSuccess) {                                                         \
        printf("CUDA error: %s\n", cudaGetErrorString(e));                          \
        exit(1);                                                                    \
    }                                                                              \
}

// Kernel de multiplicación de matrices usando memoria compartida y optimización por bloques
__global__ void matrixMultiplyShared(float *A, float *B, float *C, int n) {
    __shared__ float sA[TILE_SIZE][TILE_SIZE];  // Memoria compartida para submatriz A
    __shared__ float sB[TILE_SIZE][TILE_SIZE];  // Memoria compartida para submatriz B

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0;

    // Realizar la multiplicación en tiles
    for (int i = 0; i < (n / TILE_SIZE); ++i) {
        // Cargar submatrices A y B en memoria compartida
        sA[threadIdx.y][threadIdx.x] = A[row * n + (i * TILE_SIZE + threadIdx.x)];
        sB[threadIdx.y][threadIdx.x] = B[(i * TILE_SIZE + threadIdx.y) * n + col];

        __syncthreads();  // Sincronización para asegurarse de que los datos están cargados en la memoria compartida

        // Multiplicación de las sub-matrices
        for (int j = 0; j < TILE_SIZE; ++j) {
            sum += sA[threadIdx.y][j] * sB[j][threadIdx.x];
        }

        __syncthreads();  // Sincronización para el siguiente ciclo de carga de submatrices
    }

    // Escribir el resultado en la matriz C
    if (row < n && col < n) {
        C[row * n + col] = sum;
    }
}

int main() {
    int size = N * N * sizeof(float);
    float *h_A, *h_B, *h_C;  // Memoria en CPU
    float *d_A, *d_B, *d_C;  // Memoria en GPU

    // Reservar memoria en la CPU
    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);

    // Inicializar matrices A y B
    for (int i = 0; i < N * N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // Reservar memoria en la GPU
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copiar datos de CPU a GPU
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Configurar el tamaño del bloque e hilos
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    // Ejecutar kernel en la GPU
    matrixMultiplyShared<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaCheckError();

    // Copiar resultado de GPU a CPU
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Imprimir una parte del resultado
    printf("C[0][0] = %f\n", h_C[0]);

    // Liberar memoria
    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaCheckError();

    return 0;
}


