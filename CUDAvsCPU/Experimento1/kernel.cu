#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>
#include <math.h>

/**********************		CONSTANTS		**********************/
#define NUM_PARAMS 2		// #elements and option ( 0 = CPU , !=0 = CUDA )
#define WORK_ITERATIONS 1000

#define CPU_THREADS 8
#define MAX_CUDA_THREADS_PER_BLOCK 1024	//Divided in warps of 32 threads 
#define MAX_CUDA_BLOCKS 1024*1024
#define MAX_CHARGE_PER_THREAD 1024

/**********************		VARIABLES		**********************/
int numElements;
int option;
int programResult;

float* a;
float* b;
float* resultArray;

float* c_a;
float* c_b;
int CUDA_THREADS = MAX_CUDA_THREADS_PER_BLOCK;
int CUDA_BLOCKS;
int chargePerThread;

/**********************		METHODS		**********************/
bool ParamsIniAndControl(int na, char* arg[]);

void InicialitzeData();

void CpuExecution();

bool CudaExecution();

void ResultControl();

void FreeData();

/**********************		MAIN		**********************/
int main(int na, char* arg[])
{
	programResult = 0;

	if(ParamsIniAndControl(na,arg))
	{
		InicialitzeData();

		clock_t tMed = clock();
			if (option == 0) 
				CpuExecution();
			else 
				CudaExecution();	

		printf("Execution time = %f seconds\n", ((double)(clock() - tMed)) / CLOCKS_PER_SEC);

		ResultControl();

		FreeData();
	}  

    return programResult;
}


/**********************	DATA CONTROL	**********************/
bool ParamsIniAndControl(int na, char* arg[])
{
	if (na != NUM_PARAMS + 1)
	{
		printf("PARAM ERROR : The program need only %i parameters ( #ELEMENTS , Option ( 0 -> CPU , !=0 -> CUDA ) ) \n", NUM_PARAMS);
		programResult = 1;
	}
	else
	{
		numElements = atoi(arg[1]);
		option = atoi(arg[2]);

		if (numElements <= 0)
		{
			printf("PARAM ERROR : The first parameter (#ELEMENTS) should be 1 or more\n");
			programResult = 2;
		}
	}

	return ( programResult == 0 );
}

void InicialitzeData()
{
	a = (float *)malloc(numElements * sizeof(float));
	b = (float *)malloc(numElements * sizeof(float));
	resultArray = (float *)malloc(numElements * sizeof(float));

	int i;
	for (i = 0; i < numElements; i++)
	{
		a[i] = 1.5;
		b[i] = 0.127;
	}
}

void FreeData()
{
	free(a);
	free(b);
	free(resultArray);
}

float Calc(float a, float b)
{
	float result = a;

	int i;
	for (i = 0; i < WORK_ITERATIONS; i++)
	{
		result += b;
	}

	return result;
}

int NumOfFails()
{
	int notCorrect = 0;

	bool perfect = true;
	int i;
	for (i = 0; i < numElements; i++)
	{
		if (resultArray[i] != (Calc(a[i], b[i]))) {
			notCorrect++;
			if (perfect) printf("Not perfect results\n");
			perfect = false;
		}
			
	}

	return notCorrect;
}

void ResultControl()
{
	int notCorrects = NumOfFails();
	if (notCorrects == 0)
		printf("Results : All correct!\n");
	else
		printf("Results : %i not corrects\n", notCorrects);
}

/**********************	CPU		**********************/
void CpuExecution()
{
	int i;
	for (i = 0; i < numElements; i++)
	{
		resultArray[i] = Calc(a[i], b[i]);
	}
}


/**********************	CUDA	**********************/
__device__ float CalcCUDA(float a, float b)
{
	float result = a;

	int i;
	for (i = 0; i < WORK_ITERATIONS; i++)
	{
		result += b;
	}

	return result;
}

__global__ void DoCudaWork(int chargePerThread, float* c_a, float* c_b, int numElements) {
	
	int i = (blockIdx.x + (blockIdx.y * gridDim.x)) * blockDim.x * blockDim.y * chargePerThread +
		(threadIdx.x + (threadIdx.y * blockDim.x)) * chargePerThread ;
	int limit = i + chargePerThread - 1;

	//printf("Th (%i,%i)(%i,%i) : %i to %i\n", blockIdx.x, blockIdx.y, threadIdx.x,threadIdx.y, i, limit);

	while (i <= limit && i < numElements)
	{
		c_a[i] = CalcCUDA(c_a[i], c_b[i]);
		i++;
	}

	//printf("Thread %i finished\n", blockIdx.x * blockDim.x + threadIdx.x);

}

void MyCudaMemInicialization()
{
	/***********	MEMORY ALLOC AND CPY	***********/
	cudaMalloc(&c_a, numElements * sizeof(float));
	cudaMalloc(&c_b, numElements * sizeof(float));
	cudaMemcpy(c_a, a, numElements*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(c_b, b, numElements*sizeof(float), cudaMemcpyHostToDevice);
}

void CudaChargeControl()
{
	/***********	SET BLOCKS AND THREADS		***********/
	// Control too many threads
	while (numElements < CUDA_THREADS)
	{
		if (CUDA_THREADS <= 32) 
		{
			CUDA_THREADS--;
		}
		else 
		{
			CUDA_THREADS -= 32;
		}
	}

	//Adjust #BLOCKS
	CUDA_BLOCKS = 1;
	while (CUDA_THREADS * CUDA_BLOCKS < numElements && CUDA_BLOCKS < MAX_CUDA_BLOCKS)
	{
		CUDA_BLOCKS++;
	}

	// Adjust charge by thread
	chargePerThread = numElements / (CUDA_THREADS * CUDA_BLOCKS);
	if (chargePerThread == 0) chargePerThread = 1;

	//Control manage all the elements incrementing charge per thread
	while (numElements > (CUDA_THREADS * CUDA_BLOCKS * chargePerThread)) 
	{
		chargePerThread++;
	}

}

void CudaReturnAndFree(cudaError returnCode) 
{
	/***********	RETURN INFO AND FREE		***********/
	if (returnCode != cudaSuccess)
	{
		printf("CUDA ERROR! Error type: %s\n", cudaGetErrorString(returnCode));
	}

	cudaMemcpy(resultArray, c_a, numElements*sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(a);
	cudaFree(b);
}

bool CudaExecution()
{
	bool result = true;

	MyCudaMemInicialization();

	CudaChargeControl();		

	/***********	WORKING		***********/
	printf("CUDA launch with %i block/s and %i threads per block and a charge of %i per thread\n",
		 CUDA_BLOCKS, CUDA_THREADS , chargePerThread);

	printf("%i elements not managed in CUDA execution\n",
		numElements - ( CUDA_THREADS * CUDA_BLOCKS * chargePerThread ) );
	
	clock_t tExe = clock();
	
	DoCudaWork <<< CUDA_BLOCKS, CUDA_THREADS >>>(chargePerThread, c_a, c_b, numElements);
	cudaError returnCode = cudaDeviceSynchronize();

	printf("CUDA time (without memory alloc and copy)= %f seconds\n", ((double)(clock() - tExe)) / CLOCKS_PER_SEC);

	CudaReturnAndFree(returnCode);

	return result;
}