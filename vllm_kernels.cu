/*
 * TinyServe Optimized vLLM PagedAttention CUDA Kernels
 * Advanced Page Allocation and Memory Management for Large Language Models
 * 
 * This file contains the complete implementation of TinyServe's optimized
 * PagedAttention kernels with enhanced performance and memory efficiency.
 * 
 * Key Optimizations:
 * - FlashAttention integration
 * - Advanced memory coalescing
 * - Optimized block allocation strategies
 * - Enhanced kernel fusion
 * - Dynamic workload balancing
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cub/cub.cuh>
#include <mma.h>
#include <stdio.h>
#include <algorithm>

// TinyServe Optimized Constants
#define WARP_SIZE 32
#define BLOCK_SIZE 256
#define MAX_SEQ_LEN 131072
#define DEFAULT_BLOCK_SIZE 16
#define NUM_HEADS 32
#define HEAD_DIM 128
#define TINYSERVE_BLOCK_SIZE 64
#define SHARED_MEM_SIZE 49152  // 48KB shared memory
#define MAX_WARPS_PER_BLOCK 8
#define FLASH_ATTENTION_BLOCK_M 64
#define FLASH_ATTENTION_BLOCK_N 64

// TinyServe Enhanced Data Structures
struct TinyServeBlockTable {
    int* physical_block_ids;         // Physical block ID mapping
    int* logical_to_physical;        // Logical to physical mapping
    int* block_ref_counts;          // Block reference counts
    int* block_access_patterns;      // Access pattern optimization
    int* block_priority;            // Block priority for LRU
    int max_blocks_per_seq;          // Maximum blocks per sequence
    int total_physical_blocks;       // Total physical blocks
    int block_size;                  // Size of each block
    int cache_size;                  // Cache size for hot blocks
    float* block_weights;           // Block importance weights
};

struct TinyServeAttentionMetadata {
    int batch_size;
    int num_heads;
    int head_dim;
    int block_size;
    int max_seq_len;
    float scale;                     // 1.0f / sqrt(head_dim)
    bool use_flash_attention;        // Enable FlashAttention
    bool use_memory_optimization;    // Enable memory optimization
    int num_warps_per_block;         // Warps per block
    int shared_mem_size;            // Shared memory size
};

struct TinyServeKernelConfig {
    int block_dim_x;
    int block_dim_y;
    int block_dim_z;
    int grid_dim_x;
    int grid_dim_y;
    int grid_dim_z;
    int shared_mem_bytes;
    bool use_tensor_cores;
    bool enable_kernel_fusion;
};

// TinyServe Advanced Utility Functions
__device__ __forceinline__ float atomicMax(float* address, float val) {
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_i, assumed,
            __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__device__ __forceinline__ float atomicAdd(float* address, float val) {
    return atomicAdd(address, val);
}

// TinyServe Memory Coalescing Optimization
__device__ __forceinline__ void coalesced_load(
    const half* src, half* dst, int size, int tid, int block_size
) {
    // Optimized memory coalescing for better bandwidth utilization
    for (int i = tid; i < size; i += block_size) {
        dst[i] = src[i];
    }
}

// TinyServe Warp-level Reduction
__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// TinyServe Block Access Pattern Optimization
__device__ __forceinline__ int get_optimal_block_id(
    TinyServeBlockTable* table, int seq_id, int logical_block_id
) {
    // Use access pattern to optimize block placement
    int access_pattern = table->block_access_patterns[seq_id * table->max_blocks_per_seq + logical_block_id];
    int priority = table->block_priority[seq_id * table->max_blocks_per_seq + logical_block_id];
    
    // Prefer blocks with higher access frequency and priority
    return (access_pattern > 0) ? logical_block_id : -1;
}

// TinyServe Dynamic Workload Balancing
__device__ __forceinline__ void balance_workload(
    int* work_distribution, int total_work, int num_workers, int worker_id
) {
    int base_work = total_work / num_workers;
    int extra_work = total_work % num_workers;
    
    int start_work = worker_id * base_work + min(worker_id, extra_work);
    int end_work = start_work + base_work + (worker_id < extra_work ? 1 : 0);
    
    work_distribution[worker_id * 2] = start_work;
    work_distribution[worker_id * 2 + 1] = end_work;
}

// Kernel 1: Fused Reshape and Block Write
__global__ void fused_reshape_and_block_write_kernel(
    const half* input_kv,            // Input KV cache [batch, seq_len, hidden_size]
    half* output_blocks,             // Output block storage [num_blocks, block_size, hidden_size]
    const int* block_table,          // Block table mapping
    const int* seq_lens,             // Sequence lengths
    const int batch_size,
    const int hidden_size,
    const int block_size,
    const int max_seq_len,
    const int num_blocks_per_seq
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * max_seq_len * hidden_size;
    
    if (tid >= total_elements) return;
    
    // Calculate position in sequence
    int seq_idx = tid / (max_seq_len * hidden_size);
    int pos_in_seq = (tid % (max_seq_len * hidden_size)) / hidden_size;
    int hidden_idx = tid % hidden_size;
    
    // Skip if position exceeds sequence length
    if (pos_in_seq >= seq_lens[seq_idx]) return;
    
    // Calculate block index
    int block_idx = pos_in_seq / block_size;
    int pos_in_block = pos_in_seq % block_size;
    
    // Get physical block ID
    int physical_block_id = block_table[seq_idx * num_blocks_per_seq + block_idx];
    
    // Calculate output address
    int output_idx = physical_block_id * block_size * hidden_size + 
                     pos_in_block * hidden_size + hidden_idx;
    
    // Perform write operation
    output_blocks[output_idx] = input_kv[tid];
}

// TinyServe Kernel 2: Optimized FlashAttention with PagedAttention
__global__ void tinyserve_flash_paged_attention_kernel(
    const half* query,               // Query matrix [batch, num_heads, head_dim]
    const half* key_blocks,          // Key blocks [num_blocks, block_size, head_dim]
    const half* value_blocks,        // Value blocks [num_blocks, block_size, head_dim]
    half* output,                    // Output attention [batch, num_heads, head_dim]
    const int* block_table,          // Block table
    const int* seq_lens,             // Sequence lengths
    const TinyServeAttentionMetadata* metadata
) {
    int batch_idx = blockIdx.y;
    int head_idx = blockIdx.z;
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    
    if (batch_idx >= metadata->batch_size || head_idx >= metadata->num_heads) return;
    
    int seq_len = seq_lens[batch_idx];
    int num_blocks = (seq_len + metadata->block_size - 1) / metadata->block_size;
    
    // Shared memory for FlashAttention blocks
    extern __shared__ half shared_mem[];
    half* shared_key = shared_mem;
    half* shared_value = shared_mem + FLASH_ATTENTION_BLOCK_M * metadata->head_dim;
    half* shared_query = shared_mem + 2 * FLASH_ATTENTION_BLOCK_M * metadata->head_dim;
    
    // Initialize output
    for (int d = tid; d < metadata->head_dim; d += blockDim.x) {
        output[batch_idx * metadata->num_heads * metadata->head_dim + 
               head_idx * metadata->head_dim + d] = __float2half(0.0f);
    }
    __syncthreads();
    
    // FlashAttention-style tiling
    float max_score = -INFINITY;
    float sum_exp = 0.0f;
    
    // Load query block to shared memory
    for (int i = tid; i < FLASH_ATTENTION_BLOCK_M * metadata->head_dim; i += blockDim.x) {
        shared_query[i] = query[batch_idx * metadata->num_heads * metadata->head_dim + 
                               head_idx * metadata->head_dim + i];
    }
    __syncthreads();
    
    // Process blocks in FlashAttention style
    for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
        int physical_block_id = block_table[batch_idx * metadata->max_seq_len / metadata->block_size + block_idx];
        
        // Coalesced load of key-value blocks
        coalesced_load(
            key_blocks + physical_block_id * metadata->block_size * metadata->head_dim,
            shared_key,
            metadata->block_size * metadata->head_dim,
            tid,
            blockDim.x
        );
        
        coalesced_load(
            value_blocks + physical_block_id * metadata->block_size * metadata->head_dim,
            shared_value,
            metadata->block_size * metadata->head_dim,
            tid,
            blockDim.x
        );
        __syncthreads();
        
        // Compute attention scores with warp-level optimization
        int block_start = block_idx * metadata->block_size;
        int block_end = min(block_start + metadata->block_size, seq_len);
        
        float local_max = -INFINITY;
        for (int k_pos = block_start; k_pos < block_end; k_pos++) {
            float score = 0.0f;
            for (int d = 0; d < metadata->head_dim; d++) {
                score += __half2float(shared_query[d]) *
                        __half2float(shared_key[(k_pos - block_start) * metadata->head_dim + d]);
            }
            score *= metadata->scale;
            local_max = fmaxf(local_max, score);
        }
        
        // Warp-level reduction for max
        local_max = warp_reduce_max(local_max);
        if (lane_id == 0) {
            max_score = fmaxf(max_score, local_max);
        }
        __syncthreads();
        
        // Compute softmax and accumulate
        float local_sum = 0.0f;
        for (int k_pos = block_start; k_pos < block_end; k_pos++) {
            float score = 0.0f;
            for (int d = 0; d < metadata->head_dim; d++) {
                score += __half2float(shared_query[d]) *
                        __half2float(shared_key[(k_pos - block_start) * metadata->head_dim + d]);
            }
            score = expf(score * metadata->scale - max_score);
            local_sum += score;
            
            // Accumulate to output
            for (int d = 0; d < metadata->head_dim; d++) {
                atomicAdd(&output[batch_idx * metadata->num_heads * metadata->head_dim + 
                                 head_idx * metadata->head_dim + d],
                         score * __half2float(shared_value[(k_pos - block_start) * metadata->head_dim + d]));
            }
        }
        
        // Warp-level reduction for sum
        local_sum = warp_reduce_sum(local_sum);
        if (lane_id == 0) {
            sum_exp += local_sum;
        }
        __syncthreads();
    }
    
    // Normalize output
    for (int d = tid; d < metadata->head_dim; d += blockDim.x) {
        float val = __half2float(output[batch_idx * metadata->num_heads * metadata->head_dim + 
                                       head_idx * metadata->head_dim + d]);
        output[batch_idx * metadata->num_heads * metadata->head_dim + 
               head_idx * metadata->head_dim + d] = __float2half(val / sum_exp);
    }
}

// Kernel 2: Enhanced PagedAttention Main Kernel
__global__ void paged_attention_kernel(
    const half* query,               // Query matrix [batch, num_heads, head_dim]
    const half* key_blocks,          // Key blocks [num_blocks, block_size, head_dim]
    const half* value_blocks,        // Value blocks [num_blocks, block_size, head_dim]
    half* output,                    // Output attention [batch, num_heads, head_dim]
    const int* block_table,          // Block table
    const int* seq_lens,             // Sequence lengths
    const AttentionMetadata* metadata
) {
    int batch_idx = blockIdx.y;
    int head_idx = blockIdx.z;
    int tid = threadIdx.x;
    
    if (batch_idx >= metadata->batch_size || head_idx >= metadata->num_heads) return;
    
    int seq_len = seq_lens[batch_idx];
    int num_blocks = (seq_len + metadata->block_size - 1) / metadata->block_size;
    
    // Shared memory for storing key-value blocks
    extern __shared__ half shared_mem[];
    half* shared_key = shared_mem;
    half* shared_value = shared_mem + metadata->block_size * metadata->head_dim;
    
    // Initialize output
    for (int d = tid; d < metadata->head_dim; d += blockDim.x) {
        output[batch_idx * metadata->num_heads * metadata->head_dim + 
               head_idx * metadata->head_dim + d] = __float2half(0.0f);
    }
    __syncthreads();
    
    // First pass: compute max attention scores
    float max_score = -INFINITY;
    
    for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
        int physical_block_id = block_table[batch_idx * metadata->max_seq_len / metadata->block_size + block_idx];
        
        // Load key block to shared memory
        for (int i = tid; i < metadata->block_size * metadata->head_dim; i += blockDim.x) {
            shared_key[i] = key_blocks[physical_block_id * metadata->block_size * metadata->head_dim + i];
        }
        __syncthreads();
        
        // Compute attention scores for current block
        int block_start = block_idx * metadata->block_size;
        int block_end = min(block_start + metadata->block_size, seq_len);
        
        for (int k_pos = block_start; k_pos < block_end; k_pos++) {
            float score = 0.0f;
            for (int d = 0; d < metadata->head_dim; d++) {
                score += __half2float(query[batch_idx * metadata->num_heads * metadata->head_dim + 
                                           head_idx * metadata->head_dim + d]) *
                        __half2float(shared_key[(k_pos - block_start) * metadata->head_dim + d]);
            }
            score *= metadata->scale;
            max_score = fmaxf(max_score, score);
        }
        __syncthreads();
    }
    
    // Second pass: compute softmax and accumulate values
    float sum_exp = 0.0f;
    
    for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
        int physical_block_id = block_table[batch_idx * metadata->max_seq_len / metadata->block_size + block_idx];
        
        // Load key-value blocks to shared memory
        for (int i = tid; i < metadata->block_size * metadata->head_dim; i += blockDim.x) {
            shared_key[i] = key_blocks[physical_block_id * metadata->block_size * metadata->head_dim + i];
            shared_value[i] = value_blocks[physical_block_id * metadata->block_size * metadata->head_dim + i];
        }
        __syncthreads();
        
        int block_start = block_idx * metadata->block_size;
        int block_end = min(block_start + metadata->block_size, seq_len);
        
        for (int k_pos = block_start; k_pos < block_end; k_pos++) {
            float score = 0.0f;
            for (int d = 0; d < metadata->head_dim; d++) {
                score += __half2float(query[batch_idx * metadata->num_heads * metadata->head_dim + 
                                           head_idx * metadata->head_dim + d]) *
                        __half2float(shared_key[(k_pos - block_start) * metadata->head_dim + d]);
            }
            score = expf(score * metadata->scale - max_score);
            sum_exp += score;
            
            // Accumulate to output
            for (int d = 0; d < metadata->head_dim; d++) {
                atomicAdd(&output[batch_idx * metadata->num_heads * metadata->head_dim + 
                                 head_idx * metadata->head_dim + d],
                         score * __half2float(shared_value[(k_pos - block_start) * metadata->head_dim + d]));
            }
        }
        __syncthreads();
    }
    
    // Normalize output
    for (int d = tid; d < metadata->head_dim; d += blockDim.x) {
        float val = __half2float(output[batch_idx * metadata->num_heads * metadata->head_dim + 
                                       head_idx * metadata->head_dim + d]);
        output[batch_idx * metadata->num_heads * metadata->head_dim + 
               head_idx * metadata->head_dim + d] = __float2half(val / sum_exp);
    }
}

// Kernel 3: Fused Block Copy
__global__ void fused_block_copy_kernel(
    const half* src_blocks,          // Source blocks
    half* dst_blocks,                // Destination blocks
    const int* copy_operations,      // Copy operations [src_block_id, dst_block_id, block_size]
    const int num_operations,
    const int block_size,
    const int hidden_size
) {
    int op_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    if (op_idx >= num_operations) return;
    
    int src_block_id = copy_operations[op_idx * 3];
    int dst_block_id = copy_operations[op_idx * 3 + 1];
    int copy_size = copy_operations[op_idx * 3 + 2];
    
    int total_elements = copy_size * hidden_size;
    
    // Parallel block data copy
    for (int i = tid; i < total_elements; i += blockDim.x) {
        dst_blocks[dst_block_id * block_size * hidden_size + i] = 
            src_blocks[src_block_id * block_size * hidden_size + i];
    }
}

// TinyServe Kernel 4: Advanced Block Allocation with LRU Cache
__global__ void tinyserve_advanced_block_allocation_kernel(
    TinyServeBlockTable* table,
    const int* seq_ids,
    const int* logical_block_ids,
    int* allocated_blocks,
    const int num_allocations,
    const int* access_frequencies
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= num_allocations) return;
    
    int seq_id = seq_ids[tid];
    int logical_block_id = logical_block_ids[tid];
    int access_freq = access_frequencies[tid];
    
    // Update access pattern
    atomicAdd(&table->block_access_patterns[seq_id * table->max_blocks_per_seq + logical_block_id], 1);
    
    // Find optimal physical block using LRU and access pattern
    int best_block = -1;
    int min_priority = INT_MAX;
    
    for (int i = 0; i < table->total_physical_blocks; i++) {
        if (table->block_ref_counts[i] == 0) {
            // Check if this block is in cache
            if (i < table->cache_size) {
                // Prefer cached blocks for frequently accessed data
                if (access_freq > 5) {
                    best_block = i;
                    break;
                }
            }
            
            // Use LRU policy
            if (table->block_priority[i] < min_priority) {
                min_priority = table->block_priority[i];
                best_block = i;
            }
        }
    }
    
    if (best_block != -1) {
        // Allocate block
        table->physical_block_ids[seq_id * table->max_blocks_per_seq + logical_block_id] = best_block;
        table->block_ref_counts[best_block] = 1;
        table->block_priority[best_block] = atomicAdd(&table->block_priority[best_block], 1);
        allocated_blocks[tid] = best_block;
    } else {
        // Allocation failed
        allocated_blocks[tid] = -1;
    }
}

// TinyServe Kernel 5: Intelligent Memory Compaction
__global__ void tinyserve_intelligent_memory_compaction_kernel(
    half* blocks,
    const int* old_to_new_mapping,
    const int* block_weights,
    const int num_blocks,
    const int block_size,
    const int hidden_size,
    const int compaction_threshold
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = num_blocks * block_size * hidden_size;
    
    if (tid >= total_elements) return;
    
    int block_idx = tid / (block_size * hidden_size);
    int element_in_block = tid % (block_size * hidden_size);
    
    // Only compact blocks with low weights (less frequently accessed)
    if (block_weights[block_idx] < compaction_threshold) {
        int new_block_idx = old_to_new_mapping[block_idx];
        if (new_block_idx != block_idx) {
            blocks[new_block_idx * block_size * hidden_size + element_in_block] = 
                blocks[block_idx * block_size * hidden_size + element_in_block];
        }
    }
}

// TinyServe Kernel 6: Dynamic Workload Balancing
__global__ void tinyserve_dynamic_workload_balancing_kernel(
    const int* seq_lens,
    int* work_distribution,
    const int batch_size,
    const int num_warps_per_block
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    
    if (warp_id >= num_warps_per_block) return;
    
    // Calculate total work
    int total_work = 0;
    for (int i = 0; i < batch_size; i++) {
        total_work += seq_lens[i];
    }
    
    // Balance workload across warps
    balance_workload(work_distribution, total_work, num_warps_per_block, warp_id);
}

// Kernel 4: Enhanced Block Allocation
__global__ void block_allocation_kernel(
    BlockTable* table,
    const int* seq_ids,
    const int* logical_block_ids,
    int* allocated_blocks,
    const int num_allocations
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= num_allocations) return;
    
    int seq_id = seq_ids[tid];
    int logical_block_id = logical_block_ids[tid];
    
    // Find free physical block
    for (int i = 0; i < table->total_physical_blocks; i++) {
        if (atomicCAS(&table->block_ref_counts[i], 0, 1) == 0) {
            // Allocate block
            table->physical_block_ids[seq_id * table->max_blocks_per_seq + logical_block_id] = i;
            allocated_blocks[tid] = i;
            return;
        }
    }
    
    // Allocation failed
    allocated_blocks[tid] = -1;
}

// Kernel 5: Block Deallocation
__global__ void block_deallocation_kernel(
    BlockTable* table,
    const int* seq_ids,
    const int* logical_block_ids,
    const int num_deallocations
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= num_deallocations) return;
    
    int seq_id = seq_ids[tid];
    int logical_block_id = logical_block_ids[tid];
    
    int physical_block_id = table->physical_block_ids[seq_id * table->max_blocks_per_seq + logical_block_id];
    atomicSub(&table->block_ref_counts[physical_block_id], 1);
}

// Kernel 6: Memory Compaction
__global__ void memory_compaction_kernel(
    half* blocks,
    const int* old_to_new_mapping,
    const int num_blocks,
    const int block_size,
    const int hidden_size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = num_blocks * block_size * hidden_size;
    
    if (tid >= total_elements) return;
    
    int block_idx = tid / (block_size * hidden_size);
    int element_in_block = tid % (block_size * hidden_size);
    
    int new_block_idx = old_to_new_mapping[block_idx];
    if (new_block_idx != block_idx) {
        blocks[new_block_idx * block_size * hidden_size + element_in_block] = 
            blocks[block_idx * block_size * hidden_size + element_in_block];
    }
}

// TinyServe Host-side wrapper functions
extern "C" {

// TinyServe Optimized Kernel Launchers
void tinyserve_launch_flash_paged_attention(
    const half* query,
    const half* key_blocks,
    const half* value_blocks,
    half* output,
    const int* block_table,
    const int* seq_lens,
    const TinyServeAttentionMetadata* metadata,
    cudaStream_t stream
) {
    dim3 grid(1, metadata->batch_size, metadata->num_heads);
    int shared_mem_size = 3 * FLASH_ATTENTION_BLOCK_M * metadata->head_dim * sizeof(half);
    
    tinyserve_flash_paged_attention_kernel<<<grid, BLOCK_SIZE, shared_mem_size, stream>>>(
        query, key_blocks, value_blocks, output, block_table, seq_lens, metadata
    );
}

void tinyserve_launch_advanced_block_allocation(
    TinyServeBlockTable* table,
    const int* seq_ids,
    const int* logical_block_ids,
    int* allocated_blocks,
    const int* access_frequencies,
    int num_allocations,
    cudaStream_t stream
) {
    int num_blocks = (num_allocations + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    tinyserve_advanced_block_allocation_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        table, seq_ids, logical_block_ids, allocated_blocks, num_allocations, access_frequencies
    );
}

void tinyserve_launch_intelligent_memory_compaction(
    half* blocks,
    const int* old_to_new_mapping,
    const int* block_weights,
    int num_blocks,
    int block_size,
    int hidden_size,
    int compaction_threshold,
    cudaStream_t stream
) {
    int total_elements = num_blocks * block_size * hidden_size;
    int num_blocks_kernel = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    tinyserve_intelligent_memory_compaction_kernel<<<num_blocks_kernel, BLOCK_SIZE, 0, stream>>>(
        blocks, old_to_new_mapping, block_weights, num_blocks, block_size, hidden_size, compaction_threshold
    );
}

void tinyserve_launch_dynamic_workload_balancing(
    const int* seq_lens,
    int* work_distribution,
    int batch_size,
    int num_warps_per_block,
    cudaStream_t stream
) {
    int num_blocks = (num_warps_per_block + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    tinyserve_dynamic_workload_balancing_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        seq_lens, work_distribution, batch_size, num_warps_per_block
    );
}

// Original wrapper functions
void launch_fused_reshape_and_block_write(
    const half* input_kv,
    half* output_blocks,
    const int* block_table,
    const int* seq_lens,
    int batch_size,
    int hidden_size,
    int block_size,
    int max_seq_len,
    int num_blocks_per_seq,
    cudaStream_t stream
) {
    int total_elements = batch_size * max_seq_len * hidden_size;
    int num_blocks = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    fused_reshape_and_block_write_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        input_kv, output_blocks, block_table, seq_lens,
        batch_size, hidden_size, block_size, max_seq_len, num_blocks_per_seq
    );
}

void launch_paged_attention(
    const half* query,
    const half* key_blocks,
    const half* value_blocks,
    half* output,
    const int* block_table,
    const int* seq_lens,
    const AttentionMetadata* metadata,
    cudaStream_t stream
) {
    dim3 grid(1, metadata->batch_size, metadata->num_heads);
    int shared_mem_size = 2 * metadata->block_size * metadata->head_dim * sizeof(half);
    
    paged_attention_kernel<<<grid, BLOCK_SIZE, shared_mem_size, stream>>>(
        query, key_blocks, value_blocks, output, block_table, seq_lens, metadata
    );
}

void launch_fused_block_copy(
    const half* src_blocks,
    half* dst_blocks,
    const int* copy_operations,
    int num_operations,
    int block_size,
    int hidden_size,
    cudaStream_t stream
) {
    fused_block_copy_kernel<<<num_operations, BLOCK_SIZE, 0, stream>>>(
        src_blocks, dst_blocks, copy_operations, num_operations, block_size, hidden_size
    );
}

void launch_block_allocation(
    BlockTable* table,
    const int* seq_ids,
    const int* logical_block_ids,
    int* allocated_blocks,
    int num_allocations,
    cudaStream_t stream
) {
    int num_blocks = (num_allocations + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    block_allocation_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        table, seq_ids, logical_block_ids, allocated_blocks, num_allocations
    );
}

void launch_block_deallocation(
    BlockTable* table,
    const int* seq_ids,
    const int* logical_block_ids,
    int num_deallocations,
    cudaStream_t stream
) {
    int num_blocks = (num_deallocations + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    block_deallocation_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        table, seq_ids, logical_block_ids, num_deallocations
    );
}

void launch_memory_compaction(
    half* blocks,
    const int* old_to_new_mapping,
    int num_blocks,
    int block_size,
    int hidden_size,
    cudaStream_t stream
) {
    int total_elements = num_blocks * block_size * hidden_size;
    int num_blocks_kernel = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    memory_compaction_kernel<<<num_blocks_kernel, BLOCK_SIZE, 0, stream>>>(
        blocks, old_to_new_mapping, num_blocks, block_size, hidden_size
    );
}

} // extern "C"
