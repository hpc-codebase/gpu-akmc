#pragma once

#include "gpu_simulate.h"
#include "gpu_hash_op.hip"

__device__ LatticeTypes::lat_type getType(const long int& id){
    // printf("START GET TYPE");
    if(hash_set_contains(g_MoRe_Hash,id))       return LatticeTypes::MoRe;
    else if(hash_set_contains(g_MoMo_Hash,id))  return LatticeTypes::MoMo;
    else if(hash_set_contains(g_Re_Hash,id))    return LatticeTypes::Re;
    else if(hash_set_contains(g_V_Hash,id))     return LatticeTypes::V;
    else return LatticeTypes::Mo;
}

__device__ _type_lattice_id getId(_type_lattice_size x,_type_lattice_size y,_type_lattice_size z){
  return x + g_meta->size_x * (y + z * g_meta->size_y);
}

__device__ void getnnlattice(_type_lattice_id latti_id, dev_Lattice *_1nn_list_lattice){
    _1nn_list_lattice->id = latti_id;
    if(hash_set_contains(g_MoRe_Hash,latti_id))        _1nn_list_lattice->type = LatticeTypes::MoRe;
    else if(hash_set_contains(g_MoMo_Hash,latti_id))   _1nn_list_lattice->type = LatticeTypes::MoMo;
    else if(hash_set_contains(g_Re_Hash,latti_id))     _1nn_list_lattice->type = LatticeTypes::Re;
    else if(hash_set_contains(g_V_Hash,latti_id))      _1nn_list_lattice->type = LatticeTypes::V;
    else _1nn_list_lattice->type = LatticeTypes::Mo;
}
__device__ void get1nn(_type_lattice_coord x, _type_lattice_coord y, _type_lattice_coord z,
                  dev_Lattice *_1nn_list){
    _type_lattice_id latti_id;
    if(x % 2 == 0){
        latti_id = getId(x - 1, y - 1, z - 1);
        getnnlattice(latti_id, &_1nn_list[0]);
        latti_id = getId(x - 1, y - 1, z);
        getnnlattice(latti_id, &_1nn_list[1]);
        latti_id = getId(x - 1, y, z - 1);
        getnnlattice(latti_id, &_1nn_list[2]);
        latti_id = getId(x - 1, y, z);
        getnnlattice(latti_id, &_1nn_list[3]);
        latti_id = getId(x + 1, y - 1, z - 1);
        getnnlattice(latti_id, &_1nn_list[4]);
        latti_id = getId(x + 1, y - 1, z);
        getnnlattice(latti_id, &_1nn_list[5]);
        latti_id = getId(x + 1, y, z - 1);
        getnnlattice(latti_id, &_1nn_list[6]);
        latti_id = getId(x + 1, y, z);
        getnnlattice(latti_id, &_1nn_list[7]);
    }else {
        latti_id = getId(x - 1, y, z);
        getnnlattice(latti_id, &_1nn_list[0]);
        latti_id = getId(x - 1, y, z + 1);
        getnnlattice(latti_id, &_1nn_list[1]);
        latti_id = getId(x - 1, y + 1, z);
        getnnlattice(latti_id, &_1nn_list[2]);
        latti_id = getId(x - 1, y + 1, z + 1);
        getnnlattice(latti_id, &_1nn_list[3]);
        latti_id = getId(x + 1, y, z);
        getnnlattice(latti_id, &_1nn_list[4]);
        latti_id = getId(x + 1, y, z + 1);
        getnnlattice(latti_id, &_1nn_list[5]);
        latti_id = getId(x + 1, y + 1, z);
        getnnlattice(latti_id, &_1nn_list[6]);
        latti_id = getId(x + 1, y + 1, z + 1);
        getnnlattice(latti_id, &_1nn_list[7]);
    }
}

__device__ void get2nn(_type_lattice_coord x, _type_lattice_coord y, _type_lattice_coord z,
                              dev_Lattice *_2nn_list){
    _type_lattice_id latti_id;
    latti_id = getId(x - 2, y, z);
    getnnlattice(latti_id, &_2nn_list[0]);
    // _2nn_list[0] = &_lattices[z + 0][y + 0][x - 2];
    latti_id = getId(x, y - 1, z);
    getnnlattice(latti_id, &_2nn_list[1]);
    // _2nn_list[1] = &_lattices[z + 0][y - 1][x + 0];
    latti_id = getId(x, y, z - 1);
    getnnlattice(latti_id, &_2nn_list[2]);
    // _2nn_list[2] = &_lattices[z - 1][y + 0][x + 0];
    latti_id = getId(x, y, z + 1);
    getnnlattice(latti_id, &_2nn_list[3]);
    // _2nn_list[3] = &_lattices[z + 1][y + 0][x + 0];
    latti_id = getId(x, y + 1, z);
    getnnlattice(latti_id, &_2nn_list[4]);
    // _2nn_list[4] = &_lattices[z + 0][y + 1][x + 0];
    latti_id = getId(x + 2, y, z);
    getnnlattice(latti_id, &_2nn_list[5]);
    // _2nn_list[5] = &_lattices[z + 0][y + 0][x + 2];
}

__device__ int randomInt(unsigned long long seed, int min, int max) {
    // TODO:实现一个简单的随机数生成器
    // // 获取线程的唯一标识符
    // int threadSeed = threadIdx.x + blockIdx.x * blockDim.x; 

    // // 初始化每个线程的hiprandState
    // hiprandState state;
    // hiprand_init(seed + threadSeed, threadIdx.x, 0, &state); 

    // // 生成一个随机数，范围在[min, max]之间
    // int randNum = hiprand(&state) % (max - min + 1) + min;

    // return randNum;
    return 0;
}

__device__ void getRandomLattice(const _type_lattice_coord& x, const _type_lattice_coord& y, const _type_lattice_coord& z,
                                _type_lattice_coord& temp_x, _type_lattice_coord& temp_y, _type_lattice_coord& temp_z, const int& randomValue){
  // if(randomValue < 0 || randomValue >= 14){
  //   printf("wrong random value:%d\n",randomValue);
  // }
    switch (randomValue)
    {
    case 0:
        if(x % 2 == 0) {
            temp_x = x - 1;
            temp_y = y - 1;
            temp_z = z - 1;
        } else {
            temp_x = x - 1;
            temp_y = y;
            temp_z = z;
        }
        break;
    case 1:
        if(x % 2 == 0) {
            temp_x = x - 1;
            temp_y = y - 1;
            temp_z = z;
        } else {
            temp_x = x - 1;
            temp_y = y;
            temp_z = z + 1;
        }
        break;
    case 2:
        if(x % 2 == 0) {
            temp_x = x - 1;
            temp_y = y;
            temp_z = z - 1;
        } else {
            temp_x = x - 1;
            temp_y = y + 1;
            temp_z = z;
        }
        break;
    case 3:
        if(x % 2 == 0) {
            temp_x = x - 1;
            temp_y = y;
            temp_z = z;
        } else {
            temp_x = x - 1;
            temp_y = y + 1;
            temp_z = z + 1;
        }
        break;
    case 4:
        if(x % 2 == 0) {
            temp_x = x + 1;
            temp_y = y - 1;
            temp_z = z - 1;
        } else {
            temp_x = x + 1;
            temp_y = y;
            temp_z = z;
        }
        break;
    case 5:
        if(x % 2 == 0) {
            temp_x = x + 1;
            temp_y = y - 1;
            temp_z = z;
        } else {
            temp_x = x + 1;
            temp_y = y;
            temp_z = z + 1;
        }
        break;
    case 6:
        if(x % 2 == 0) {
            temp_x = x + 1;
            temp_y = y;
            temp_z = z - 1;
        } else {
            temp_x = x + 1;
            temp_y = y + 1;
            temp_z = z;
        }
        break;
    case 7:
        if(x % 2 == 0) {
            temp_x = x + 1;
            temp_y = y;
            temp_z = z;
        } else {
            temp_x = x + 1;
            temp_y = y + 1;
            temp_z = z + 1;
        }
        break;
    case 8:
        temp_x = x - 2;
        temp_y = y;
        temp_z = z;
        break;
    case 9:
        temp_x = x;
        temp_y = y - 1;
        temp_z = z;
        break;
    case 10:
        temp_x = x;
        temp_y = y;
        temp_z = z - 1;
        break;
    case 11:
        temp_x = x;
        temp_y = y;
        temp_z = z + 1;
        break;
    case 12:
        temp_x = x;
        temp_y = y + 1;
        temp_z = z;
        break;
    case 13:
        temp_x = x + 2;
        temp_y = y;
        temp_z = z;
        break;
    default:
      // printf("wrong random period,%d\n",randomValue);
      // abort();
        break;
    }
}
__device__ float randomFloat(unsigned long long seed) {
    // 获取线程的唯一标识符
    unsigned long long seed1 = clock();
    
    // 定义并初始化hiprandState
    hiprandState state;
    hiprand_init(seed + seed1, threadIdx.x, 0, &state);  // 初始化状态
    
    // 生成 0 到 1 之间的浮点数
    float randNum = hiprand_uniform(&state);  // 使用 hiprand_uniform 生成浮点数
    return randNum;
    // return 0;
}

__device__ void getCoordByLId(_type_lattice_id lid, _type_lattice_coord *x, _type_lattice_coord *y,
                            _type_lattice_coord *z){
    _type_lattice_id lid2 = lid;
    *x = lid % g_meta->size_x;
    lid = lid /  g_meta->size_x;
    *y = lid %  g_meta->size_y;
    *z = lid /  g_meta->size_y;
}

__device__ int isGhostLat(_type_lattice_id lid){
    _type_lattice_coord x = 0, y = 0, z = 0;
    _type_lattice_coord zero = 0;
    getCoordByLId(lid, &x, &y, &z);
    x -= g_meta->ghost_x;
    y -= g_meta->ghost_y;
    z -= g_meta->ghost_z;
    // printf("%ld - (%d,%d,%d) is in ghost\n",lid,x,y,z);
    // printf("ghost region:x:%ld,y:%ld,z:%ld\n",
    //         g_meta->box_x,
    //         g_meta->box_y,
    //         g_meta->box_z
    //         );
    // printf("%d,%d,%d,%d,%d,%d\n", x < zero,y < zero,z < zero,
    //                             x >= g_meta->box_x, y >= g_meta->box_y, z >= g_meta->box_z
    //     );
    int res = x < zero || y < zero || z < zero || x >= g_meta->box_x || y >= g_meta->box_y || z >= g_meta->box_z;
    // if(res == 0)
    //     printf("%ld - (%d,%d,%d) is not in ghost\n",lid,x,y,z);
    // else
    //     printf("%ld - (%d,%d,%d) is in ghost\n",lid,x,y,z);
    return res;
}

__device__ int isSurfaceLat(_type_lattice_id lid){
    _type_lattice_coord x = 0, y = 0, z = 0;
    _type_lattice_coord zero = 0;
    getCoordByLId(lid, &x, &y, &z);
    x -= g_meta->ghost_x;
    y -= g_meta->ghost_y;
    z -= g_meta->ghost_z;
    
    // printf("surface region:x[0,%ld)-[%ld,%ld),y[0,%ld)-[%ld,%ld),z[0,%ld)-[%ld,%ld)\n",
    //         g_meta->ghost_x,g_meta->box_x - g_meta->ghost_x,g_meta->box_x,
    //         g_meta->ghost_y,g_meta->box_y - g_meta->ghost_y,g_meta->box_y,
    //         g_meta->ghost_z,g_meta->box_z - g_meta->ghost_z,g_meta->box_z
    //         );
    // printf("%d,%d,%d,%d,%d,%d\n",   (zero <= x && x < g_meta->ghost_x),
    //                                 (zero <= y && y < g_meta->ghost_y),
    //                                 (zero <= z && z < g_meta->ghost_z),
    //                                 (x >= (g_meta->box_x - g_meta->ghost_x) && x < g_meta->box_x),
    //                                 (y >= (g_meta->box_y - g_meta->ghost_y) && y < g_meta->box_y),
    //                                 (z >= (g_meta->box_z - g_meta->ghost_z) && z < g_meta->box_z)
    //     );
    // printf("%d,%d\n",zero <= z,z < g_meta->ghost_z);
    int res = (zero <= x && x < g_meta->ghost_x) ||(zero <= y && y < g_meta->ghost_y) || (zero <= z && z < g_meta->ghost_z) || (x >= (g_meta->box_x - g_meta->ghost_x) && x < g_meta->box_x) || (y >= (g_meta->box_y - g_meta->ghost_y) && y < g_meta->box_y) || (z >= (g_meta->box_z - g_meta->ghost_z) && z < g_meta->box_z);
    // if(res!=0)
    //     printf("%ld - (%d,%d,%d) is in surface\n",lid,x,y,z);
    // else
    //     printf("%ld - (%d,%d,%d) is not in surface\n",lid,x,y,z);
    return res;
}

__global__ void update_Pair_Atoms(int arr_size){
    //这个函数用于更新g_pair_atoms
    /*
        一共有两种情况需要更新pair_Atoms
        1,每次calcExchangePairs计算结束后如果发生了交换，则交换的位置要被更新到此函数中
        2,通信过程中需要更新,
        这个函数应用于第二种更新，第一种更新直接放到函数中执行就行
    */
    //todo 更新函数有问题
    for(int i=0;i<arr_size;i++)
        g_Pair_Atoms[i] = -1; 
    int index = 0;
    long int momo=0,more=0;
    #ifdef DEBUG_MODE_GPU
        printf("-------------MOMO_DATA-------------\n");
    #endif
    for(int i=0;i<g_MoMo_Hash->capacity;i++){
        if(g_MoMo_Hash->keys[i]!=g_MoMo_Hash->empty_flag && g_MoMo_Hash->keys[i]!=g_MoMo_Hash->tombstone_flag){
            g_Pair_Atoms[index++]=g_MoMo_Hash->keys[i];
            printf("pair atom is MoMo,(%ld,%ld)",g_MoMo_Hash->keys[i],getType(g_MoMo_Hash->keys[i]));
            if(!isGhostLat(g_MoMo_Hash->keys[i])){
                #ifdef DEBUG_MODE_GPU
                    printf("%ld\t",g_MoMo_Hash->keys[i]);
                #endif
                momo++;
            }
        }
    }
    #ifdef DEBUG_MODE_GPU
        printf("\n");
        printf("-------------MORe_DATA-------------\n");
    #endif
    for(int i=0;i<g_MoRe_Hash->capacity;i++){
        if(g_MoRe_Hash->keys[i]!=g_MoRe_Hash->empty_flag && g_MoRe_Hash->keys[i]!=g_MoRe_Hash->tombstone_flag){
            g_Pair_Atoms[index++]=g_MoRe_Hash->keys[i];
            if(!isGhostLat(g_MoRe_Hash->keys[i])){
                printf("pair atom is MoRe,(%ld,%ld)\n",g_MoRe_Hash->keys[i],getType(g_MoRe_Hash->keys[i]));
                #ifdef DEBUG_MODE_GPU
                    printf("%ld\t",g_MoRe_Hash->keys[i]);
                #endif
                more++;
            }
        }
    }
    int V=0;
    #ifdef DEBUG_MODE_GPU
        printf("-------------VAC_DATA-------------\n");
    #endif
    for(int i=0;i<g_V_Hash->capacity;i++){
        if(g_V_Hash->keys[i]!=g_V_Hash->empty_flag && g_V_Hash->keys[i]!=g_V_Hash->tombstone_flag){
            g_Pair_Atoms[index++]=g_V_Hash->keys[i];
            if(!isGhostLat(g_V_Hash->keys[i])){
                #ifdef DEBUG_MODE_GPU
                    printf("%ld\t",g_V_Hash->keys[i]);
                #endif
                V++;
            }
        }
    }
    #ifdef DEBUG_MODE_GPU
        printf("\n");
        printf("momo num is :%ld\t",momo);
        printf("more num is:%ld\n",more);
        printf("vac num is:%ld\n",V);
        printf("AFTER UPDATE atom_num is:%ld/%ld\n",index,arr_size);
    #endif

}
