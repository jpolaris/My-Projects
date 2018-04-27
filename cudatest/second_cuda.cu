#include <iostream>
#include <cuda_runtime.h>
#include <cstdio>

#define NUM_THREADS 256 

bool initCUDA()
{
    int count;
    cudaGetDeviceCount(&count);
    if(count == 0){
        std::cout<<"There is no device."<<std::endl;
        return false;
    }

    int i;
    for(i=0;i<count;i++){
        cudaDeviceProp prop;
        if(cudaGetDeviceProperties(&prop,i)==cudaSuccess){
            if(prop.major>=1) break;
        }
    }

    if(i==count){
        std::cout<<"There is no device supporting CUDA 1.x/n"<<std::endl;
        return false;
    }

    cudaSetDevice(i);
    return true;
}

void matgen(float* mat, int dimx, int dimy)
{
    
    for(int i=0;i<dimx;i++){
        for(int j=0;j<dimy;j++){
            mat[i*dimx+j] = (float)rand()/RAND_MAX;// + (float)rand()/(float)(RAND_MAX*RAND_MAX);
        }
    }
}

void matmult(const float* mata, const float* matb, float* result, int ax, int aybx, int by)
{
    for(int i=0;i<ax;i++){
        for(int j=0;j<by;j++){
            double t = 0.0;
            for(int k=0;k<aybx;k++){
                t += mata[i*aybx+k]*matb[k*aybx+j];
            }

            result[i*ax+j] = t; 
        }
    }
}

/*void matmult(const float* a, int lda, const float* b, int ldb, float* c, int ldc, int n)
{
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            double t = 0;
            for(int k=0;k<n;k++){
                t += a[i*lda+k]*b[k*ldb+j];
            }

            c[i*ldc+j] = t;
        }
    }
}*/

void compare_mat(const float* a, int lda, const float* b, int ldb, int n)
{
    float max_err = 0;
    float average_err = 0;
    int i, j;

    for(i = 0; i < n; i++) {
        for(j = 0; j < n; j++) {
            if(b[i * ldb + j] != 0) {
                float err = fabs((a[i * lda + j] - b[i * ldb + j]) / b[i * ldb + j]);
                if(max_err < err) max_err = err;
                average_err += err;
            }
        }
    }

    std::cout<<"Max error: "<<max_err<<" Average error: "<<average_err<<std::endl;
}

__global__
static void matMultCUDA(const float* a, size_t lda, const float* b, size_t ldb, float* c, size_t ldc, int n)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int idx = bid*blockDim.x + tid;
    const int row = idx/n;
    const int column = idx % n;
    int i;

    /*if(row < n && column < n) {
        float t = 0;
        for(i = 0; i < n; i++) {
            t += a[row * lda + i] * b[i * ldb + column];
        }
        c[row * ldc + column] = t;
    }*/

    if(row < n && column < n) {
        float t = 0;
        float y = 0;
        for(i = 0; i < n; i++) {
            float r;
            y -= a[row * lda + i] * b[i * ldb + column];
            r = t - y;
            y = (r - t) + y;
            t = r;
        }
    }
}

clock_t matmultCUDA(const float* a, int lda, const float* b, int ldb, float* c, int ldc, int n)
{
    float *ac, *bc, *cc;
    clock_t start, end;

    start = clock();
    cudaMalloc((void**)&ac, sizeof(float)*n*n);
    cudaMalloc((void**)&bc, sizeof(float)*n*n);
    cudaMalloc((void**)&cc, sizeof(float)*n*n);

    cudaMemcpy2D(ac, sizeof(float)*n, a, sizeof(float)*lda, sizeof(float)*n, n, cudaMemcpyHostToDevice);
    cudaMemcpy2D(bc, sizeof(float)*n, b, sizeof(float)*ldb, sizeof(float)*n, n, cudaMemcpyHostToDevice);

    int blocks = (n+NUM_THREADS-1)/NUM_THREADS;
    matMultCUDA<<<blocks * n, NUM_THREADS>>>
            (ac, n, bc, n, cc, n, n);

    cudaMemcpy2D(c, sizeof(float) * ldc, cc, sizeof(float) * n,
        sizeof(float) * n, n, cudaMemcpyDeviceToHost);

    cudaFree(ac);
    cudaFree(bc);
    cudaFree(cc);

    end = clock();

    return end - start;

}

int main()
{
    if(!initCUDA()) return 0;
    std::cout<<"CUDA initialized"<<std::endl;

    float *a, *b, *c, *d;
    int n=1000;

    a = (float*)malloc(sizeof(float)*n*n);
    b = (float*)malloc(sizeof(float)*n*n);
    c = (float*)malloc(sizeof(float)*n*n);
    d = (float*)malloc(sizeof(float)*n*n);

    /*int temp = 1;
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            a[i*n+j] = temp;
            b[i*n+j] = temp;
            temp++;
        }
    }*/
    srand(0);
    matgen(a,n,n);
    matgen(b,n,n);

    //matmult(a,n,b,n,c,n,n);
    
    clock_t time = matmultCUDA(a, n, b, n, c, n, n);
    matmult(a,b,d,n,n,n);
    compare_mat(c, n, d, n, n);
    /*for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            std::cout<<"["<<i<<","<<j<<"]="<<a[i*n+j]<<std::endl;
        }
    }

    std::cout<<"printing b..."<<std::endl;
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            std::cout<<"["<<i<<","<<j<<"]="<<b[i*n+j]<<std::endl;
        }
    }

    std::cout<<"printing c..."<<std::endl;
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            std::cout<<"["<<i<<","<<j<<"]="<<c[i*n+j]<<std::endl;
        }
    }*/

    double sec = (double) time / CLOCKS_PER_SEC;
    char rst[100];
    //sprintf(rst,“Time used: %.2f %.2lf GFLOPS”, sec, 2.0 * n * n * n / (sec * 1E9));
    sprintf(rst,"Time used: %0.2f (%.2lf GFLOPS)", sec, 2.0*n*n*n/(sec*1E9));
    std::cout<<rst<<std::endl;

    return 0;
}