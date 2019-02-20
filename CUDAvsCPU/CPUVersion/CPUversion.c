#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>
#include <pthread.h>

#define NUM_PARAMS 1		// #elements and option ( 0 = CPU , !=0 = CUDA )
#define WORK_ITERATIONS 1000
#define CPU_THREADS 8

int numElements;
float* a;
float* b;
float* result;
pthread_t threadList[CPU_THREADS];
int threadCharge;


void InicialitzeData()
{
	a = (float *)malloc(numElements * sizeof(float));
	b = (float *)malloc(numElements * sizeof(float));
	result = (float *)malloc(numElements * sizeof(float));

	int i;
	for (i = 0; i < numElements; i++)
	{
		a[i] = 1.5;
		b[i] = 0.127;
	}
}

float Calc (float a, float b)
{
	float result = a;

	int i;
	for (i = 0; i < WORK_ITERATIONS; i++)
	{
		result += b;
	}

	return result;
}

int CheckCorrect()
{
	int notCorrect = 0;

	int i;
	for (i = 0; i < numElements; i++)
	{
		if (result[i] != (Calc(a[i],b[i])))
			notCorrect++;
	}

	return notCorrect;
}

void FreeData()
{
	free(a);
	free(b);
	free(result);
}

/****************	CPU		*************/
/**   PRINCIPAL FUNCTIONS  **/
void * DoWork (void * param)
{
    int thID = (intptr_t) param;

	int ini = thID * threadCharge;
	int fin = ini + threadCharge;

	int i;
	for (i = ini; i < fin; i++)
	{
		result[i] = Calc(a[i], b[i]);
	}

    return thID;
}

void CreateThreads()
{

    int i;
    for( i = 0; i < CPU_THREADS; i++ )
    {
        pthread_create(&threadList[i], NULL, DoWork, (void *)(intptr_t) i);
    }
}

void WaitThreads()
{

    int i,result;
    for( i = 0; i < CPU_THREADS; i++ )
    {
        pthread_join (threadList[i], (void *)(intptr_t)&result);
    }
}

void CpuExecution()
{
    threadCharge = numElements/CPU_THREADS;

	CreateThreads();

	WaitThreads();
}

int main(int na, char* arg[])
{
	int result = 0;

	if (na != NUM_PARAMS + 1)
	{
		printf("The program need only %i parameters (#ELEMENTS)\n", NUM_PARAMS);
		result = 1;
	}
	else
	{
		numElements = atoi(arg[1]);

		if (numElements <= 0)
		{
			printf("The first parameter (#ELEMENTS) shoud be 1 or more\n");
			result = 1;
		}
		else
		{
			InicialitzeData();

            printf("Starting execution with %i threads\n",CPU_THREADS);
			clock_t tMed = clock();

            CpuExecution();

			double exeTime = ((double)(clock() - tMed)) / CLOCKS_PER_SEC;
			printf("Execution time = %f seconds\n", exeTime);

			int notCorrects = CheckCorrect();
			if (notCorrects == 0)
				printf("All correct!\n");
			else
				printf("%i not corrects\n",notCorrects);

			FreeData();
		}
	}

    return result;
}
