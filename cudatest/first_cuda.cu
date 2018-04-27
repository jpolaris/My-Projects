#include <iostream>
#include <cuda_runtime.h>

#define DATA_SIZE 1048576
#define THREAD_NUM 256
#define BLOCK_NUM 32

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

void generateNumbers(int* number, int size)
{
    for(int i=0;i<size;i++){
        number[i] = rand() % 10;
    }
}


__global__
static void sumOfSquares(int* num, int* result, clock_t* time)
{
    extern __shared__ int shared[];
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    int offset = 1, mask = 1;
    //const int size = DATA_SIZE/THREAD_NUM;

    if(tid == 0) time[bid] = clock();
    shared[tid] = 0;

    for(int i=bid*THREAD_NUM+tid;i<DATA_SIZE;i+=THREAD_NUM*BLOCK_NUM){
        shared[tid] += num[i]*num[i];
    }

    __syncthreads();
    while(offset<THREAD_NUM){
        if((tid & mask) == 0){
            shared[tid] += shared[tid+offset];
        }

        offset += offset;
        mask = offset+mask;
        __syncthreads();
    }

    if(tid == 0){
        time[bid+BLOCK_NUM] = clock();
        result[bid] = shared[0];
    }
}

int sumOfSquared(int* data){
    int sum = 0;
    for(int i=0;i<DATA_SIZE;i++){
        sum += data[i]*data[i];
    }

    return sum;
}

int main()
{
    if(!initCUDA()) return 0;
    std::cout<<"CUDA initialized"<<std::endl;

    int data[DATA_SIZE];
    generateNumbers(data, DATA_SIZE);

    int* gpudata, *result;
    clock_t* time;

    cudaMalloc((void**)&gpudata, sizeof(int)*DATA_SIZE);
    cudaMalloc((void**)&result, sizeof(int)*BLOCK_NUM);
    cudaMalloc((void**)&time, sizeof(clock_t)*BLOCK_NUM*2);
    cudaMemcpy(gpudata, data, sizeof(int)*DATA_SIZE, cudaMemcpyHostToDevice);

    sumOfSquares<<<BLOCK_NUM,THREAD_NUM,THREAD_NUM*sizeof(int)>>>(gpudata, result, time);

    int sum[THREAD_NUM*BLOCK_NUM];
    clock_t time_used[BLOCK_NUM*2];

    cudaMemcpy(sum, result, sizeof(int)*BLOCK_NUM, cudaMemcpyDeviceToHost);
    cudaMemcpy(&time_used, time, sizeof(clock_t)*BLOCK_NUM*2, cudaMemcpyDeviceToHost);
    cudaFree(gpudata);
    cudaFree(result);
    cudaFree(time);

    int final_sum = 0;
    for(int i=0;i<BLOCK_NUM;i++){
        final_sum += sum[i];
    }

    clock_t timeUsed;
    for(int i=0;i<BLOCK_NUM;i++){
        timeUsed += (time_used[i+BLOCK_NUM]-time_used[i]);
    }

    std::cout<<"sum (GPU): "<<final_sum<<"; time:"<<timeUsed<<std::endl;

    clock_t start = clock();
    int sumCpu = sumOfSquared(data);
    clock_t usedTime = clock() - start;
    std::cout<<"sum (CPU): "<<sumCpu<<"; time:"<<usedTime<<std::endl;
    return 0;
}