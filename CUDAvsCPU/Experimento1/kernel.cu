/* 
 * Author : Benet Manzanares Salor
 * 
 * Date : 20 / 1 / 2019
 * 
 * Description: 
 * 	A program to compare the execution time between CUDA execution and CPU execution.
 * 	Specifically, after check the parameters, inicialization and choose between CUDA or CPU,
 *  the program do a work (loop/s, operations ...) for every element of an array and 
 * 	indicate the execution time.
 * 
 * 	At the CUDA call, the configuration priority is : 
 * 		CUDA_THREADS (multiple of 32 if is possible) > CUDA_BLOCKS (at least 1 ) > ChargePerThread
 * 
 * 	This code only test one thread of CPU, to use multiple threads you need to use the
 *  CPUVersion ( locate at the CPUVerison folder ).
 * 
 * Parameters:
 * 	Number of elements : Length of the array to do the work
 * 	Option : Choose between CUDA or CPU
 * 		0 -> CPU || !=0 -> CUDA
 *  	
 */

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>
#include <math.h>

/**********************		CONSTANTS		**********************/
#define NUM_PARAMS 2
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
bool CorrectParameters(int na, char* arg[]);

void InicialitzeData();

void CpuExecution();

bool CudaExecution();

void ResultControl();

void FreeData();

/**********		MAIN		**********/
int main(int na, char* arg[])
{
	programResult = 0;

	if(CorrectParameters(na,arg))
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


/* CorrectParameters
 *	Description:
 * 		Check if the enter parameters of the programs are correct
 *	Parameters:
 * 		na : Number of arguments introduced by user
 * 		arg : Reference of the table with arguments
 * 	Return:
 * 		True if all is correct 
 */
bool CorrectParameters(int na, char* arg[])
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

/* InicialitzeData 
 *	Description:
 *		Inicialize all the elements of the arrays used in the program with constant values.
 * 		This arrays are directly use at the CPU execution and copyed 
 * 		for the CUDA execution at the MyCudaMemInicialization method. * 
 */
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

/* FreeData
 *	Description: 
 * 		Free the arrays used
 */
void FreeData()
{
	free(a);
	free(b);
	free(resultArray);
}

/* Calc
 *	Description:
 * 		Do the calculation for a element a and b.
 * 	Parameters:
 * 		a : first value
 * 		b : second value
 * 	Return:
 * 		The final value of the operation to put it in the resultArray 
 */
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

/**********************		CPU 1 THREAD 	**********************/
void CpuExecution()
{
	int i;
	for (i = 0; i < numElements; i++)
	{
		resultArray[i] = Calc(a[i], b[i]);
	}
}


/**********************		CUDA	**********************/
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
	//	Memory allocation and copy
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

	// Control manage all the elements incrementing charge per thread
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
