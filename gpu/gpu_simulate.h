#ifndef GPU_SIMULATE_H
#define GPU_SIMULATE_H

#include "hip/hip_runtime.h"
#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>

#include "gpuError.h"
#include "DeviceVacRatesSolver.h"
#include <cassert>
#include <iostream>
#include <hiprand/hiprand_kernel.h>

#include <logs/logs.h>
#include "../models/abvi/kmc.h"
#include "../models/abvi/box.h"
#include "../models/abvi/rate/vacancy_rates_solver.h"
#include "../models/abvi/defect/vacancy_list.h"
#include "../models/abvi/rate/bonds/pair_bond.hpp"
#include "../models/abvi/rate/rates_types.h"
#include "../models/abvi/event.h"

#include "../src/type_define.h"
#include "../src/lattice/lattice.h"
#include "../src/lattice/lattices_list.h"
#include "../src/lattice/lattice_types.h"
#include "../src/lattice/lattice_list_meta.h"

void recb_solver_GPU(
                    const unsigned int& sector_id,long int arr_size,
                    long int *&h_ghost_Hash,long int *&h_surface_Hash,
                    long int *&h_ghost_type, long int *&h_surface_type,
                    int *sizes
                  );
double calculate_GPU(_type_lattice_id *vac_idArray,  const _type_lattice_count& vac_count);

void selectAndPerformEventGPU(double excepted_rand, int rank, _type_lattice_count step, int sect,
                              std::array<std::unordered_set<_type_lattice_id>, 7>& exchange_ghost, const unsigned int sector_id,
                              std::array<std::unordered_set<_type_lattice_id>, 8>& exchange_surface_x, 
                              std::array<std::unordered_set<_type_lattice_id>, 8>& exchange_surface_y,
                              std::array<std::unordered_set<_type_lattice_id>, 8>& exchange_surface_z,
                              std::array<comm::Region<comm::_type_lattice_size>, 56>& ghost_region);

void initialize_gpu(int process_rank);

void gpu_final();

void gpu_prepare(double& v, double& T, LatticesList *p_list, comm::ColoredDomain *_p_domain,
                std::vector<long int> pair_atoms
                );

void exchange_lattices(dev_event* h_event);

void add_surface(_type_lattice_id& surface_id,
                 std::array<std::unordered_set<_type_lattice_id>, 8>& exchange_surface_x, 
                 std::array<std::unordered_set<_type_lattice_id>, 8>& exchange_surface_y,
                 std::array<std::unordered_set<_type_lattice_id>, 8>& exchange_surface_z);
                 
void output_time();

struct GpuCommMaintenanceProfiler {
  double h2d_time = 0.0;
  double hash_update_time = 0.0;
  double rebuild_cleanup_time = 0.0;
};

void resetGpuCommMaintenanceProfiler();
GpuCommMaintenanceProfiler getGpuCommMaintenanceProfiler();

__device__ LatticeTypes::lat_type getType(const long int& id);

void transferBufferToGPU(ChangeLattice *buffer, int receive_len, const int dimension,
  const _type_lattice_count *sub_box_lattice_size, const _type_lattice_count *neighbour_local_sub_box, unsigned int id);
void transferBufferToGPU2(ChangeLattice *buffer, int receive_len, const int dimension,
  const _type_lattice_count *sub_box_lattice_size, const _type_lattice_count *neighbour_local_sub_box, unsigned int next_id);
// 线程事务状态机结构体
struct AtomTx {
    _type_lattice_id src_id;   // 原原子ID
    _type_lattice_id dst_id;   // 目标位置ID
    bool active;               // 是否仍在活跃（活没干完）
    bool success;              // 本轮物理交换是否成功
    int op_type;               // 反应类型：1-移动，2-身份转换...
};
// __global__ void calcExchangePairs(dev_meta meta,long int *Pair_Atoms,
//                             HIPHashSet *MoRe_Hash,HIPHashSet *MoMo_Hash,
//                             HIPHashSet *Re_Hash,HIPHashSet *V_Hash,
//                             HIPHashSet *Busy_Set,
//                             HIPHashSet *ghost_Hash,HIPHashSet *surface_Hash
// );

// __device__ int isSurfaceLat(_type_lattice_id lid);

// __device__ int isGhostLat(_type_lattice_id lid);

// __device__ void getCoordByLId(_type_lattice_id lid, _type_lattice_coord *x, _type_lattice_coord *y,
//                             _type_lattice_coord *z);

// __device__ float randomFloat(unsigned long long seed);

// __device__ void getRandomLattice(const _type_lattice_coord& x, const _type_lattice_coord& y, const _type_lattice_coord& z,
//                                 _type_lattice_coord& temp_x, _type_lattice_coord& temp_y, _type_lattice_coord& temp_z, const int& randomValue){

// __device__ int randomInt(unsigned long long seed,int min,int max);

// __device__ void get2nn(_type_lattice_coord x, _type_lattice_coord y, _type_lattice_coord z,
//                               dev_nnLattice *_2nn_list);

// __device__ void get1nn(_type_lattice_coord x, _type_lattice_coord y, _type_lattice_coord z,
//                   dev_nnLattice *_1nn_list);

// __device__ void getnnlattice(_type_lattice_id latti_id, dev_nnLattice *_1nn_list_lattice);

// __device__ _type_lattice_id getId(_type_lattice_size x,_type_lattice_size y,_type_lattice_size z);

// __device__ LatticeTypes::lat_type getType(const _type_lattice_id& id);

// __global__ void setGlobalPointer(dev_meta meta,
//                             HIPHashSet *MoRe_Hash,HIPHashSet *MoMo_Hash,
//                             HIPHashSet *Re_Hash,HIPHashSet *V_Hash,HIPHashSet *Busy_Set);

// __device__ int hash_set_contains(const int* table, _type_lattice_id key);

// __device__ int hash_set_insert(int* table, _type_lattice_id key);

// __device__ int hash_set_remove(int* table, _type_lattice_id key);
#endif /*GPU_SIMULATE_H*/
