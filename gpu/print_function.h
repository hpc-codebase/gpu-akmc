#include "gpu_simulate.h"
#include "global_var.h"


__device__ void printHashTableKernel(GPUHashSet* table, int max_entries, bool verbose);
__device__ void printDevMetaKernel(const dev_meta* meta);


__global__ void print_pair_atoms(int arr_size);

__device__ void printDevMetaKernel(const dev_meta* meta);

__device__ void printTopN(GPUHashSet* table, int max_entries);

__global__ void print_g_Pair_Atoms(int arr_size);

void printDevMeta(const dev_meta& meta);

__global__ void print_surface_and_ghost();

__global__ void print_Hash_and_Pair(int arr_size,int flag);

__global__ void print_momo_and_more();
