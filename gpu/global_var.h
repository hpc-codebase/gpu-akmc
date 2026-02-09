#define debug_gpu true
__device__ static GPUHashSet *g_MoRe_Hash;
__device__ static GPUHashSet *g_MoMo_Hash;
__device__ static GPUHashSet *g_Re_Hash;
__device__ static GPUHashSet *g_V_Hash;
__device__ static dev_meta *g_meta;
__device__ static GPUHashSet *g_Busy_Set;
__device__ static GPUHashSet *g_surface_Hash;
__device__ static GPUHashSet *g_ghost_Hash;
__device__ static long int  *g_Pair_Atoms;
static long int arr_size;
static long int pair_atoms_size;
