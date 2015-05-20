#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <vector>

/**
 *  This file is about the cuda code for the strig match:sunday algorithm.
 *  This main idea to use GPU(cuda) to accelerate the speed of the sunday algorithm.
 *  Random data experiment shows that we can achieve almost ten times speedup.
 *
 * 	In the GPU method we use one thread to do sunday algorithm on small substring(I divide the string into substring with length of 64)
 * 
 * 	@author gaozhefeng XIDIAN UNIVERSITY
 *   
 */

// the block size of the one-dimensional block.
#define BLOCKSIZE 256
// the length of the substring.
#define STRLEN 64
// the default size of the text.
#define DEFAULT_SIZE 16777216


using namespace std;

// generate a string randomly
void initial(char * text, const int n);
// generate the next array from the pattern string.
void create_next(int * next, const char * pattern);
// to padding some redundant data so that it is convenient for the gpu processing.
void pad_text(int * host_text, int size, char * text, int len_t, int boundry);
// the sunday algorithm code for CPU.
void sunday(vector<int> &location, const char* text, int len_t, const char* pattern, int len_p, const int * next);
// the sunday algorithm code for GPU.
__global__ void sunday_kernel(int * dev_text, int len_t, int * dev_pattern, int len_p, int * dev_location, int * dev_next, int size);


int main(int argc, char* argv[])
{
	
	// timing for CPU.
	clock_t cpu_start, cpu_end;
	// the time that the CPU code cost.
	float cpu_time_elapsed = 0.;
	// timing for GPU.
	cudaEvent_t gpu_start, gpu_end;
	// the time that the GPU code cost.
	float gpu_time_elapsed = 0.;

	// the size of the text.It can obtain from command line or initial by default.
	int N;
	if (argc > 1)
	{
		N = atoi(argv[1]);
	}
	else
	{
		N = DEFAULT_SIZE;
	}

	// @@ allocate memory for the raw text.All the char in the text is A-Z. 
	char * text = (char*) calloc(N+1, sizeof(char));
	// the random pattern 
	char pattern[] = "MOWZ";
	// the host_text is used to holds text which has been padded
	int * host_text = NULL;
	// convert char to int
	int * host_pattern = NULL;
	// host_next is the next array calculate from the pattern.
	int * host_next = NULL;
	// if match in the location i than host_location[i] = 1 else host_location[i] = 0.
	int * host_location = NULL;
	
	// @@ generate the raw text randomly
	initial(text, N);
	// get the length of the raw text and pattern.
	int len_t = N;
	int len_p = strlen(pattern);
	
	// the length of the redundant data
	int boundry = STRLEN-len_p+1;
	// the size is length of the padding text
	int size = (len_t/boundry);
	// obtain the number substrings
	if (len_t%boundry > len_p-1)
	{
		size += 1;
	}

	size *= STRLEN;
	
	// @@ allocate memory on CPU
	host_text = (int *)calloc(size, sizeof(int));
	host_location = (int *)calloc(size, sizeof(int));
	host_pattern = (int *)calloc(len_p, sizeof(int));
	host_next = (int *)calloc(26, sizeof(int));
	
	if (host_text == NULL || host_pattern == NULL || host_next == NULL || host_location == NULL)
	{
		printf("Allocating memroy on cpu failed!\n");
		return -1;
	}
	// obtain the next array
	create_next(host_next, pattern);
	
	// padding the raw text and obtain the padded text
	pad_text(host_text, size, text, len_t, boundry);

	// @@ run sunday algorithm on the CPU.
	vector<int> location;
	cpu_start = clock();
	sunday(location, text, len_t, pattern, len_p, host_next);
	cpu_end = clock();
	cpu_time_elapsed = (float)(cpu_end-cpu_start)/CLOCKS_PER_SEC;

	printf("CPU sunday done.\n");
	
	// convert char to int
	for (int i = 0; i < len_p; i++)
	{
		host_pattern[i] = (int)pattern[i];
	}

	// data on GPU.
	int * dev_text = NULL;
	int * dev_pattern = NULL;
	int * dev_next = NULL;
	int * dev_location = NULL;

	cudaError_t err;

	// @@ allocate memory on GPU.
	err = cudaMalloc((void **)&dev_text, size*sizeof(int));
	if (err != cudaSuccess)
	{
		printf("Allocating memroy on gpu failed!\n");
		return -1;
	}
	err = cudaMalloc((void **)&dev_pattern, len_p*sizeof(int));
	if (err != cudaSuccess)
	{
		printf("Allocating memroy on gpu failed!\n");
		return -1;
	}
	err = cudaMalloc((void **)&dev_next, 26*sizeof(int));
	if (err != cudaSuccess)
	{
		printf("Allocating memroy on gpu failed!\n");
		return -1;
	}
	err = cudaMalloc((void **)&dev_location, size*sizeof(int));
	if (err != cudaSuccess)
	{
		printf("Allocating memroy on gpu failed!\n");
		return -1;
	}

	// @@ copy data from CPU to GPU.
	cudaMemcpy(dev_text, host_text, size*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_pattern, host_pattern, len_p*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_next, host_next, 26*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_location, host_location, size*sizeof(int), cudaMemcpyHostToDevice);

	cudaEventCreate(&gpu_start);   
	cudaEventCreate(&gpu_end);
	// the dim of block and grid.
	dim3 dimBlock(BLOCKSIZE,1,1);
	dim3 dimGrid((size/STRLEN-1)/BLOCKSIZE+1,1,1);

	cudaEventRecord(gpu_start, 0);
	// @@ run sunday kernel on the GPU.
	sunday_kernel<<<dimGrid, dimBlock>>>(dev_text, len_t, dev_pattern, len_p, dev_location, dev_next, size);
	cudaEventRecord(gpu_end, 0);
	cudaEventSynchronize(gpu_start);  
	cudaEventSynchronize(gpu_end);    
	cudaEventElapsedTime(&gpu_time_elapsed,gpu_start,gpu_end);

	printf("GPU sunday done.\n");

	// copy data from GPU to CPU.
	cudaMemcpy(host_location, dev_location, size*sizeof(int), cudaMemcpyDeviceToHost);

	// show the result that sunday algorithm run on CPU and GPU.
	printf("\n\nresults...\nthe length of the text is %d\n", N);
	printf("sunday algorithm on cpu cost time=%f(s)\n", cpu_time_elapsed);
	printf("sunday algorithm on gpu cost time=%f(s)\n", gpu_time_elapsed/1000.);

	printf("the matches that cpu code has found:\n");
	if (0 == location.size())
	{
		printf("cpu do not find the matches.\n");
	}
	else
	{
		for (int i = 0; i < location.size(); i++)
		{
			printf("cpu location:%d\n", location[i]);
		}
	}
	printf("the matches that gpu code has found:\n");
	int flag = 1;
	for (int i = 0; i < len_t; i++)
	{
		if (host_location[i])
		{
			printf("gpu location:%d\n", i);
			flag = 0;
		}
			
	}
	if (flag)
	{
		printf("gpu do not find the matches.\n");
	}
	 //destory the event
	cudaEventDestroy(gpu_start);   
	cudaEventDestroy(gpu_end);
	

	// free memory on CPU.
	free(host_text);
	free(host_pattern);
	free(host_next);
	free(host_location);
	
	// free memory on GPU.
	cudaFree(dev_text);
	cudaFree(dev_pattern);
	cudaFree(dev_next);
	cudaFree(dev_location);

	system("pause");
	return 0;
}



/**
 * generate raw text randomly
 * @param text raw text
 * @param n    the length of the text
 */
void initial(char * text, const int n)
{
	// obtain the seed 
	srand(unsigned(time(0)));
	for (int i = 0; i < n; i++)
	{
		text[i] = rand()%26+'A';
	}
	// add an '\0' at the end.
	text[n] = '\0';
	printf("Initial text done.\n");
}

/**
 * generate next array
 * @param next    next array
 * @param pattern the pattern text
 */
void create_next(int * next, const char * pattern)
{
	int len_p = strlen(pattern);
	// generate delta shift table
	for (int i = 0; i < 26; i++)
	{
		next[i] = len_p + 1;
	}
	for (int i = 0; i < len_p; i++)
	{
		next[pattern[i]-'A'] = len_p - i;
	}

}


/**
 * sunday code on CPU
 * @param location record the location where matched
 * @param text     raw text
 * @param len_t    the length of the text
 * @param pattern  pattern text
 * @param len_p    the length of the pattern
 * @param next     next array
 */
void sunday(vector<int> &location, const char* text, int len_t, const char* pattern, int len_p, const int * next)
{
	
	// the current postion of the raw text
	int pos = 0;

	while(pos < (len_t - len_p+1)) 
	{
		
		int i = pos;
		// j is used to trace in the pattern
		int j;

		for (j = 0; j < len_p; j++, i++)
		{
			if( text[i] != pattern[j])//doesn't match
			{
				if (pos + len_p >= len_t)// all done. return 
					return;
				// jump to the next position
				pos += next[text[pos + len_p] - 'A'];
				break;  
			}
		
		}
		// find a match
		if ( j == len_p)
		{
			// record the match location
			location.push_back(pos);
			// jump to the next position
			pos += 1;
			

		}
	}

}

/**
 * padding the raw text
 * @param host_text padded text
 * @param size     	the length of the padded text
 * @param text      raw text
 * @param len_t     the length of the raw text
 * @param boundry   the length of the redundant string data
 */
void pad_text(int * host_text, int size, char * text, int len_t, int boundry)
{
	
	//boundry = STRLEN - strlen(pattern) + 1
	int offset = 0;
	for (int i = 0; i < size; i++)
	{
		if (i && i%STRLEN == 0)
		{
			offset += boundry;
		}
		
		if (offset + i%STRLEN < len_t)
		{
			host_text[i] = (int)text[offset + i%STRLEN];
		}
		else
		{
			break;
		}

	}
}

/**
 * sunday code on GPU
 * @param dev_text     padded text on GPU
 * @param len_t        the length of text
 * @param dev_pattern  pattern on GPU
 * @param len_p        the length of pattern
 * @param dev_location record the location of match
 * @param dev_next     next array on GPU
 * @param size         the length of padded text
 */
__global__ void sunday_kernel(int * dev_text, int len_t, int * dev_pattern, int len_p, int * dev_location, int * dev_next, int size)
{
	
	// the number of substring
	int idx =  threadIdx.x + blockIdx.x * blockDim.x;
	// the start position in the dev_text
	int offset = idx * STRLEN;
	int pos = 0;

	
	if (idx >= size/STRLEN)
		return ;
	
	while ( pos < (STRLEN-len_p+1))
	{
		// the current position in the dev text
		int i = pos+offset;
		// the position in the pattern
		int j;
		
		for (j = 0; j < len_p; ++j, ++i)
		{
			if( dev_text[i] != dev_pattern[j])//doesn't match
			{
				if (0 == dev_text[pos + offset + len_p])// the substring is done.
				{
					return ;
				}
				// jump to the next position
				pos += dev_next[dev_text[pos + offset + len_p] - 65];
				break;  
			}
		
		}
		// find a match
		if ( j == len_p)
		{
			// record the location
			dev_location[idx * (STRLEN-len_p+1) + pos] = 1;
			pos += 1;
		}

	}


}
