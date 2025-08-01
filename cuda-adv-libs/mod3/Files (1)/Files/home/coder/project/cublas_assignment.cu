#include <stdlib.h>
#include <stdio.h>
#include <cublas.h>
#define HA 2
#define WA 9
#define WB 2
#define HB WA 
#define WC WB   
#define HC HA  
#define index(i,j,ld) (((j)*(ld))+(i))

void printMat(float*P,int uWP,int uHP){
  int i,j;
  for(i=0;i<uHP;i++){

      printf("\n");

      for(j=0;j<uWP;j++)
          printf("%f ",P[index(i,j,uHP)]);
  }
}

__host__ float* initializeHostMemory(int height, int width, bool random, float nonRandomValue) {
  // TODO allocate host memory of type float of size height * width called hostMatrix

  // TODO fill hostMatrix with either random data (if random is true) else set each value to nonRandomValue

  return hostMatrix;
}

__host__ float *initializeDeviceMemoryFromHostMemory(int height, int width, float *hostMatrix) {
  // TODO allocate device memory of type float of size height * width called deviceMatrix

  // TODO set deviceMatrix to values from hostMatrix

  return deviceMatrix;
}

__host__ float *retrieveDeviceMemory(int height, int width, float *deviceMatrix, float *hostMemory) {
  // TODO get matrix values from deviceMatrix and place results in hostMemory

  return hostMemory;
}

__host__ void printMatrices(float *A, float *B, float *C){
  printf("\nMatrix A:\n");
  printMat(A,WA,HA);
  printf("\n");
  printf("\nMatrix B:\n");
  printMat(B,WB,HB);
  printf("\n");
  printf("\nMatrix C:\n");
  printMat(C,WC,HC);
  printf("\n");
}

__host__ int freeMatrices(float *A, float *B, float *C, float *AA, float *BB, float *CC){
  free( A );  free( B );  free ( C );
  cublasStatus status = cublasFree(AA);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! memory free error (A)\n");
    return EXIT_FAILURE;
  }
  status = cublasFree(BB);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! memory free error (B)\n");
    return EXIT_FAILURE;
  }
  status = cublasFree(CC);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! memory free error (C)\n");
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

int  main (int argc, char** argv) {
  cublasStatus status;
  cublasInit();

  // TODO initialize matrices A and B (2d arrays) of floats of size based on the HA/WA and HB/WB to be filled with random data

  if( A == 0 || B == 0){
    return EXIT_FAILURE;
  } else {
    // TODO create arrays of floats C filled with random value
    // TODO create arrays of floats alpha filled with 1's
    // TODO create arrays of floats beta filled with 0's

    // TODO use initializeDeviceMemoryFromHostMemory to create AA from matrix A
    // TODO use initializeDeviceMemoryFromHostMemory to create BB from matrix B
    // TODO use initializeDeviceMemoryFromHostMemory to create CC from matrix C

    cublasHandle_t handle;
    cublasCreate_v2(&handle);

    // TODO perform Single-Precision Matrix to Matrix Multiplication, GEMM, on AA and BB and place results in CC

    C = retrieveDeviceMemory(HC, WC, CC, C);

    printMatrices(A, B, C);

    freeMatrices(A, B, C, AA, BB, CC);
    
    /* Shutdown */
    status = cublasShutdown();
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! shutdown error (A)\n");
      return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
  }

}
