#pragma once

#include "gpu_simulate.h"

__device__ void checkpointers() {
    // 检查g_MoRe_Hash指针
    if (g_MoRe_Hash == nullptr) {
        printf("ERROR: g_MoRe_Hash is NULL!\n");
    } else {
        printf("g_MoRe_Hash is valid, add:%p capacity: %d\n", g_MoRe_Hash,g_MoRe_Hash->capacity);
    }
    
    // 检查g_MoMo_Hash指针
    if (g_MoMo_Hash == nullptr) {
        printf("ERROR: g_MoMo_Hash is NULL!\n");
    } else {
        printf("g_MoMo_Hash is valid, , add:%p,capacity: %d\n", g_MoMo_Hash,g_MoMo_Hash->capacity);
    }
    
    // 检查g_Re_Hash指针
    if (g_Re_Hash == nullptr) {
        printf("ERROR: g_Re_Hash is NULL!\n");
    } else {
        printf("g_Re_Hash is valid, capacity: %d\n", g_Re_Hash->capacity);
    }
    
    // 检查g_V_Hash指针
    if (g_V_Hash == nullptr) {
        printf("ERROR: g_V_Hash is NULL!\n");
    } else {
        printf("g_V_Hash is valid, capacity: %d\n", g_V_Hash->capacity);
    }
    
    // 检查g_meta指针
    if (g_meta == nullptr) {
        printf("ERROR: g_meta is NULL!\n");
    } else {
        printf("g_meta is valid, size_x: %lld\n", g_meta->size_x);
    }
    
    // 检查g_Busy_Set指针
    if (g_Busy_Set == nullptr) {
        printf("ERROR: g_Busy_Set is NULL!\n");
    } else {
        printf("g_Busy_Set is valid, capacity: %d\n", g_Busy_Set->capacity);
    }
    
    // 检查g_surface_Hash指针
    if (g_surface_Hash == nullptr) {
        printf("ERROR: g_surface_Hash is NULL!\n");
    } else {
        printf("g_surface_Hash is valid, capacity: %d\n", g_surface_Hash->capacity);
    }
    
    // 检查g_ghost_Hash指针
    if (g_ghost_Hash == nullptr) {
        printf("ERROR: g_ghost_Hash is NULL!\n");
    } else {
        printf("g_ghost_Hash is valid, capacity: %d\n", g_ghost_Hash->capacity);
    }
    
    // 检查g_Pair_Atoms指针
    if (g_Pair_Atoms == nullptr) {
        printf("ERROR: g_Pair_Atoms is NULL!\n");
    } else {
        printf("g_Pair_Atoms is valid, address: 0x%p\n", (void*)g_Pair_Atoms);
    }
}


__device__ int hash_set_contains_debug(GPUHashSet* table, long int key) {
    table = g_MoMo_Hash;
    if(table == nullptr)
        printf("momo is null1");
    printf("table addr:%p,key:%ld",table,key);
    if (table == nullptr) {
        printf("Error: table is nullptr\n");
        return 0;
    }
    if (table->keys == nullptr) {
        printf("Error: table->keys is nullptr\n");
        return 0;
    }
    if (table->capacity <= 0) {
        printf("Error: Invalid capacity (%d)\n", table->capacity);
        return 0;
    }
    int index = key % table->capacity;
    if (index < 0 || index >= table->capacity) {
        printf("Error: Computed index out of range (%d)\n", index);
        return 0;
    }

    printf("Searching for key: %d, start index: %d, capacity: %d\n", key, index, table->capacity);

    const int original_index = index;
    int attempts = 0;

    do {
        int current = table->keys[index];

        printf("Checking index: %d, key at index: %d\n", index, current);

        if (current == key) {
            printf("Key %d found at index %d\n", key, index);
            return 1;
        }
        if (current == table->empty_flag) {
            printf("Empty slot encountered at index %d, terminating search\n", index);
            break;  // 遇到空槽提前终止
        }

        index = (index + 1) % table->capacity;
        attempts++;

        if (attempts > table->capacity) {
            printf("Error: Infinite loop detected, breaking out\n");
            break;
        }

    }while (index != original_index);  // 完整遍历整个表

    printf("Key %d not found\n", key);
    return 0;
}

// 设备端存在性检查函数
__device__ int hash_set_contains(GPUHashSet* table, long int key) {
    int i=0;
    int index = key % table->capacity;
    const int original_index = index;
    volatile long int* keys = (volatile long int*)table->keys;  // 避免编译器优化

    while (true) {
        long int current = keys[index];  // 使用 volatile 方式读取 key

        if (current == table->empty_flag) {
            return 0;  // 空槽，说明 key 不存在
        }

        if (current == key) {
            return 1;  // 找到 key
        }

        index = (index + 1) % table->capacity;
        if (index == original_index) {
            return 0;  // 完整遍历哈希表后仍未找到 key
        }
    }
}

__device__ int hash_set_find(long int key){
    return 0;
}

// 设备端插入函数（线程安全）
__device__ int hash_set_insert(GPUHashSet* set, long int key) {
    int pos = abs(key) % set->capacity;  // 保持修正后的哈希计算
    int start_pos = pos;
    do {
        long int current = set->keys[pos];  // 直接内存访问
        if (current == key) {
            return 0; // 已存在相同key
        }

        if (current == set->empty_flag || current == set->tombstone_flag) {
            set->keys[pos] = key; // 直接赋值
            return 1;
        }

        pos = (pos + 1) % set->capacity;
    } while (pos != start_pos);

    return 0; // 哈希表已满
}

// 设备端删除函数（线程安全）
__device__ int hash_set_remove(GPUHashSet* table,long int key) {
    int index = abs(key) % table->capacity;
    const int original_index = index;

    do {
        long int current = table->keys[index]; // 直接内存访问
        if (current == table->empty_flag) {
            return 0;
        }
        if (current == key) {
            table->keys[index] = table->tombstone_flag; // 直接赋值
            return 1;
        }
    index = (index + 1) % table->capacity;
    } while (index != original_index);

    return 0;
}

__device__ long atomicExch(long* address, long val) {
    unsigned long long* addr_as_ull = reinterpret_cast<unsigned long long*>(address);
    unsigned long long val_as_ull = static_cast<unsigned long long>(val);
    
    unsigned long long old_val = atomicExch(addr_as_ull, val_as_ull);
    return static_cast<long>(old_val);
}
// 辅助函数
__device__ unsigned long long int_as_longlong(unsigned int low, unsigned int high) {
    return ((unsigned long long)high << 32) | low;
}

// 替代CUDA的__longlong_as_int函数
// 原子加载操作 - 支持long类型
__device__ long atomicLoad(long* address) {
    unsigned long long* addr_as_ull = reinterpret_cast<unsigned long long*>(address);
    
    // 使用两个32位原子操作模拟64位原子加载
    unsigned int* addr_as_int = reinterpret_cast<unsigned int*>(addr_as_ull);
    unsigned int low = atomicExch(&addr_as_int[0], addr_as_int[0]);
    unsigned int high = atomicExch(&addr_as_int[1], addr_as_int[1]);
    
    return static_cast<long>(int_as_longlong(low, high));
}

// 原子交换操作 - 支持long类型
__device__ int hash_set_remove_atom_op(GPUHashSet* table, long key) {
    unsigned long long index = key % table->capacity;
    unsigned long long original_index = index;
    
    do {
        // 使用适配long类型的原子加载
        long current = atomicLoad(&table->keys[index]);
        
        if (current == table->empty_flag) {
            return 0;
        }
        
        if (current == key) {
            // 使用适配long类型的原子交换
            atomicExch(&table->keys[index], table->tombstone_flag);
            return 1;
        }
        
        index = (index + 1) % table->capacity;
    } while (index != original_index);
    
    return 0;
}

__device__ int hash_set_insert_atom_op(GPUHashSet* set, long int key) {
    unsigned long long pos = key % set->capacity;  // 保持修正后的哈希计算
    unsigned long long start_pos = pos;

    do {
        // long int current = set->keys[pos];  // 直接内存访问
        long current = atomicLoad(&set->keys[pos]);
        if (current == key) {
            return 0; // 已存在相同key
        }
        if (current == set->empty_flag || current == set->tombstone_flag) {
            // set->keys[pos] = key; // 直接赋值
            atomicExch(&set->keys[pos], key);
            return 1;
        }
        pos = (pos + 1) % set->capacity;
    } while (pos != start_pos);

  return 0; // 哈希表已满
}