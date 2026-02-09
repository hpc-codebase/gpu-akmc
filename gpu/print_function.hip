#pragma once
#include "gpu_simulate.h"

__device__ void printHashTableKernel(GPUHashSet* table, int max_entries, bool verbose) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {  // 仅由第一个线程执行
        printf("=== GPU Hash Table Info ===\n");
        printf("Capacity: %d\n", table->capacity);
        printf("Empty Flag: %ld\n", table->empty_flag);
        printf("Tombstone Flag: %ld\n", table->tombstone_flag);
        printf("---------------------------\n");
        
        const int display_count = min(max_entries, table->capacity);
        int valid_count = 0;
        
        for (int i = 0; i < display_count; i++) {
            long int key = table->keys[i];
            
            if (key == table->empty_flag) {
                if (verbose) printf("[%3d]: EMPTY\n", i);
            } else if (key == table->tombstone_flag) {
                if (verbose) printf("[%3d]: TOMBSTONE\n", i);
            } else {
                printf("[%3d]: %ld\n", i, key);
                valid_count++;
            }
        }
        
        if (table->capacity > display_count) {
            printf("... and %d more entries.\n", table->capacity - display_count);
        }
        
        float load_factor = (float)valid_count / table->capacity;
        printf("Total valid entries: %d\n", valid_count);
        printf("Load factor: %.3f\n", load_factor);
        printf("===========================\n");
    }
}

__global__ void print_pair_atoms(int arr_size){
    printf("===========start print pair==========");
    for(int i=0;i<arr_size * 2;i++){
        printf("%lld\t",g_Pair_Atoms[i]);
        if(i%10 == 0)
            printf("\n");
    }
    printf("===========end print pair============");
}

__device__ void printDevMetaKernel(const dev_meta* meta) {
    //用于打印设备信息，晶格尺寸等初始化信息
    // 仅由第一个线程执行打印，避免输出重复
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("=== dev_meta Structure Information ===\n");
        
        // 打印晶格尺寸信息
        printf("Lattice Sizes:\n");
        printf("  size_x: %lld (simulation lattice size)\n", meta->size_x);
        printf("  size_y: %lld\n", meta->size_y);
        printf("  size_z: %lld\n", meta->size_z);
        
        // 打印当前进程的模拟盒尺寸
        printf("\nCurrent Process Box Sizes:\n");
        printf("  box_x: %lld (2x real size for BCC structure)\n", meta->box_x);
        printf("  box_y: %lld (same as real size)\n", meta->box_y);
        printf("  box_z: %lld (same as real size)\n", meta->box_z);
        
        // 打印全局模拟盒尺寸
        printf("\nGlobal Simulation Box Sizes:\n");
        printf("  g_box_x: %lld\n", meta->g_box_x);
        printf("  g_box_y: %lld\n", meta->g_box_y);
        printf("  g_box_z: %lld\n", meta->g_box_z);
        
        // 打印全局坐标基准
        printf("\nGlobal Coordinate Baselines:\n");
        printf("  g_base_x: %lld (for global lattice ID calculation)\n", meta->g_base_x);
        printf("  g_base_y: %lld\n", meta->g_base_y);
        printf("  g_base_z: %lld\n", meta->g_base_z);
        
        // 打印幽灵晶格尺寸
        printf("\nGhost Lattice Sizes:\n");
        printf("  ghost_x: %lld (2x real size)\n", meta->ghost_x);
        printf("  ghost_y: %lld\n", meta->ghost_y);
        printf("  ghost_z: %lld\n", meta->ghost_z);
        
        // 打印ID相关信息
        printf("\nID Information:\n");
        printf("  _max_id: %d (max local lattice ID)\n", meta->_max_id);
        printf("  local_base_id: %d (global ID = local ID + this value)\n", meta->local_base_id);
        
        // 打印regions数组（分维度显示）
        printf("\nRegions Array (48 elements):\n");
        const int regions_per_dim = 8;
    for (int dim = 0; dim < 3; ++dim) {
            printf("  Dimension %d Regions:\n", dim);
            for (int region = 0; region < regions_per_dim; ++region) {
                int index = dim * regions_per_dim + region;
                printf("    Region %d: %lld", region, meta->regions[index]);
                if (region % 2 == 1) {
                    printf(" (boundary: %lld to %lld)", meta->regions[index - 1], meta->regions[index]);
                }
                printf("\n");
            }
        }
        
        printf("=====================================\n");
    }
}


__device__ void printTopN(GPUHashSet* table, int max_entries){
    //用于展示一个哈希表topN的信息
   if (threadIdx.x == 0 && blockIdx.x == 0) {  // 仅由第一个线程执行
        printf("=== TOP %d info===\n",max_entries);
        printf("---------------------------\n");
        
        const int display_count = min(max_entries, table->capacity);
        int valid_count = 0;
        
        for (int i = 0; i < table->capacity; i++) {
            long int key = table->keys[i];
            if(valid_count >= max_entries)
            break;
            if (key == table->empty_flag) {
                continue;
            } else if (key == table->tombstone_flag) {
                continue;
            } else {
                printf("[%3d]: %ld\t", i, key);
                valid_count++;
            }
        }
        printf("\n");
        printf("===========================\n");
    }
}


__global__ void print_g_Pair_Atoms(int arr_size){
    printf("==========pair_atoms============\n");
    for(int i = 0;i<arr_size;i++){
        printf("%ld\t",g_Pair_Atoms[i]);
        if(i%5==0)
            printf("\n");
    }
    printf("\n");
    printf("================================\n");
}

void printDevMeta(const dev_meta& meta) {
    std::cout << "=== dev_meta Structure Information ===" << std::endl;
    
    // 打印晶格尺寸信息
    std::cout << "Lattice Sizes:" << std::endl;
    std::cout << "  size_x: " << meta.size_x << " (simulation lattice size)" << std::endl;
    std::cout << "  size_y: " << meta.size_y << std::endl;
    std::cout << "  size_z: " << meta.size_z << std::endl;
    
    // 打印当前进程的模拟盒尺寸（注意box_x的特殊说明）
    std::cout << "\nCurrent Process Box Sizes:" << std::endl;
    std::cout << "  box_x: " << meta.box_x << " (2x real size for BCC structure)" << std::endl;
    std::cout << "  box_y: " << meta.box_y << " (same as real size)" << std::endl;
    std::cout << "  box_z: " << meta.box_z << " (same as real size)" << std::endl;
    
    // 打印全局模拟盒尺寸
    std::cout << "\nGlobal Simulation Box Sizes:" << std::endl;
    std::cout << "  g_box_x: " << meta.g_box_x << std::endl;
    std::cout << "  g_box_y: " << meta.g_box_y << std::endl;
    std::cout << "  g_box_z: " << meta.g_box_z << std::endl;
    
    // 打印全局坐标基准
    std::cout << "\nGlobal Coordinate Baselines:" << std::endl;
    std::cout << "  g_base_x: " << meta.g_base_x << " (for global lattice ID calculation)" << std::endl;
    std::cout << "  g_base_y: " << meta.g_base_y << std::endl;
    std::cout << "  g_base_z: " << meta.g_base_z << std::endl;
    
    // 打印幽灵晶格尺寸
    std::cout << "\nGhost Lattice Sizes:" << std::endl;
    std::cout << "  ghost_x: " << meta.ghost_x << " (2x real size)" << std::endl;
    std::cout << "  ghost_y: " << meta.ghost_y << std::endl;
    std::cout << "  ghost_z: " << meta.ghost_z << std::endl;
    
    // 打印ID相关信息
    std::cout << "\nID Information:" << std::endl;
    std::cout << "  _max_id: " << meta._max_id << " (max local lattice ID)" << std::endl;
    std::cout << "  local_base_id: " << meta.local_base_id << " (global ID = local ID + this value)" << std::endl;
    
    // 打印regions数组（分维度显示）
    std::cout << "\nRegions Array (48 elements):" << std::endl;
    const int regions_per_dim = 8;  // 假设每个维度有8个区域
    for (int dim = 0; dim < 3; ++dim) {
        std::cout << "  Dimension " << dim << " Regions:" << std::endl;
        for (int region = 0; region < regions_per_dim; ++region) {
            int index = dim * regions_per_dim + region;
            std::cout << "    Region " << region << ": " << meta.regions[index];
            if (region % 2 == 1) {
                std::cout << " (boundary: " << meta.regions[index - 1] << " to " << meta.regions[index] << ")";
            }
            std::cout << std::endl;
        }
    }
    
    std::cout << "=====================================" << std::endl;
}

__global__ void print_surface_and_ghost(){
    int g=0,s=0;
    long int x=0,y=0,z=0;
    printf("++++++++++++s+++++++++++++++++++++\n");
    for(int i=0;i<g_surface_Hash->capacity;i++)
        if(g_surface_Hash->keys[i]!=-1 && g_surface_Hash->keys[i]!=-1){
            long int id = g_surface_Hash->keys[i];
            x = id%g_meta->size_x;
            y = (id / g_meta->size_x) % g_meta->size_y;
            z = id / (g_meta->size_x * g_meta->size_y);
            printf("s(%ld,%ld)-(%ld,%ld,%ld)\t",g_surface_Hash->keys[i],getType(g_surface_Hash->keys[i]),x,y,z);
            s++;
            if(s%5==0)
            printf("\n");
        }
    printf("\n");
    printf("sssssssssssssss\n");
    for(int i=0;i<g_ghost_Hash->capacity;i++)
        if(g_ghost_Hash->keys[i]!=-1 && g_ghost_Hash->keys[i]!=-1){
            long int id = g_ghost_Hash->keys[i];
            x = id%g_meta->size_x;
            y = (id / g_meta->size_x) % g_meta->size_y;
            z = id / (g_meta->size_x * g_meta->size_y);
            printf("g(%ld,%ld)-(%ld,%ld,%ld)\t",g_ghost_Hash->keys[i],getType(g_ghost_Hash->keys[i]),x,y,z);
            g++;
            if(g%5==0)
            printf("\n");
        }
    printf("\n");
}

__global__ void print_Hash_and_Pair(int arr_size,int flag){
    long int pair_cnt=0,momo=0,more=0;
    for(int i=0;i<arr_size * 2;i++)
        if(g_Pair_Atoms[i]!=-1)
            pair_cnt++;
    for(int i=0;i<g_MoMo_Hash->capacity;i++)
        if(g_MoMo_Hash->keys[i]!=-1 && g_MoMo_Hash->keys[i]!=-2)
            momo++;
    for(int i=0;i<g_MoRe_Hash->capacity;i++)
        if(g_MoRe_Hash->keys[i]!=-1 && g_MoRe_Hash->keys[i]!=-2)
            more++;
    printf("==current hash and pairs========\n");
    if(flag == 1)
        printf("after:");
    printf("arr_size:%ld,pair: %ld\t momo %ld\t more %ld\n",arr_size,pair_cnt,momo,more);
    printf("==============================\n");
}

__global__ void print_momo_and_more(){
    printf("----------------------after momo-----------------------\n");
    for(int i=0;i<g_MoMo_Hash->capacity;i++)
        if(g_MoMo_Hash->keys[i]!=-1 && g_MoMo_Hash->keys[i]!=-2)
            printf("%ld\t",g_MoMo_Hash->keys[i]);
    printf("\n");
    printf("----------------------after more-----------------------\n");
    for(int i=0;i<g_MoRe_Hash->capacity;i++)
        if(g_MoRe_Hash->keys[i]!=-1 && g_MoRe_Hash->keys[i]!=-2)
            printf("%ld\t",g_MoRe_Hash->keys[i]);
    printf("\n");
}

__global__ void printpointer(){
    if(g_MoMo_Hash == nullptr)
        printf("is a null\n");
    else
        printf("not null\n");
}