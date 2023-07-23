#include <stdio.h>
#include <assert.h>
#include <cuda.h>

__global__ void matMultKernel(float* A, float* B, float* C, int N) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  float sum = 0;
  if(row < N && col < N) {
    for(int i = 0; i < N; i++) {
      sum += A[row * N + i] * B[i * N + col];
    }
    C[row * N + col] = sum;
  }
}

void matMult(float* A, float* B, float* C, int N) {
  int size = N * N * sizeof(float);
  float *d_A, *d_B, *d_C;

  cudaMalloc((void**) &d_A, size);
  cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
  cudaMalloc((void**) &d_B, size);
  cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

  cudaMalloc((void**) &d_C, size);

  dim3 threadsPerBlock(N, N);
  dim3 blocksPerGrid(1, 1);
  if (N*N > 1024){
    threadsPerBlock.x = 1024;
    threadsPerBlock.y = 1024;
    blocksPerGrid.x = ceil(double(N)/double(threadsPerBlock.x));
    blocksPerGrid.y = ceil(double(N)/double(threadsPerBlock.y));
  }

  matMultKernel<<<blocksPerGrid,threadsPerBlock>>>(d_A, d_B, d_C, N);

  cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

  cudaFree(d_A); 
  cudaFree(d_B); 
  cudaFree(d_C);
}

int main() {
  int N = 2;
  float A[N*N], B[N*N], C[N*N];
  for(int i = 0; i < N*N; i++) {
    A[i] = i;
    B[i] = i;
  }
  matMult(A, B, C, N);
  for(int i = 0; i < N*N; i++) {
    printf("%f ", C[i]);
  }
  return 0;
}
