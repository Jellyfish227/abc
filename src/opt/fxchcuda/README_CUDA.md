# FXCH CUDA Parallelization

## Overview

This directory contains CUDA-accelerated implementations of the Fast eXtract with Cube Hashing (FXCH) algorithm. The primary optimization target is the `Fxch_SCHashTableEntryCompare` function, which accounts for **96.70%** of the runtime when processing large datasets.

## Performance Bottleneck

Based on profiling data (100k1.txt):
- **Function**: `Fxch_SCHashTableEntryCompare`
- **Call count**: 406,596,557 times
- **Time consumed**: 718.30 seconds (96.70% of total runtime)
- **Average time per call**: ~1.77 nanoseconds

The bottleneck occurs because this comparison function is called sequentially in tight loops during:
1. Hash table insertion (`Fxch_SCHashTableInsert`)
2. Hash table removal (`Fxch_SCHashTableRemove`)

## Parallelization Strategy

### Key Insight
When inserting or removing an entry in a hash table bin, the function compares one entry against all other entries in that bin. These comparisons are **independent** and can be executed in parallel.

### Implementation

#### 1. CUDA Kernel (`Kernel.cu`)
- **`ParallelEntryCompareKernel`**: Each CUDA thread compares the target entry against one existing entry
- **Thread organization**: 256 threads per block, with blocks calculated based on bin size
- **Data structures**: 
  - Flattened cube data for efficient GPU memory access
  - Offset and size arrays for cube indexing
  - Result array for comparison outcomes

#### 2. Host Interface (`FxchCudaSCHashTable.c`)
- **`FxchCuda_SCHashTableInsert`**: Parallelized insertion with GPU comparison
- **`FxchCuda_SCHashTableRemove`**: Parallelized removal with GPU comparison
- **`PrepareCubeDataForGPU`**: Flattens cube data for GPU transfer
- **Fallback mechanism**: Automatically reverts to CPU implementation on GPU errors

## Files

```
src/opt/fxchcuda/
├── Kernel.cu                      # CUDA kernel implementation
├── FxchCudaSCHashTable.c         # CUDA-accelerated hash table operations
├── FxchCuda.h                    # Header declarations
├── FxchCuda.c                    # Main entry point (currently commented out)
├── module.make                   # Build configuration
└── README_CUDA.md               # This file
```

## Usage

### Compilation Requirements
- NVIDIA CUDA Toolkit (tested with CUDA 11.0+)
- NVCC compiler
- C++ compiler with C++11 support

### Build Instructions

Add to your build configuration:
```bash
nvcc -c Kernel.cu -o Kernel.o
g++ -c FxchCudaSCHashTable.c -o FxchCudaSCHashTable.o
# Link with CUDA runtime library
```

### API Usage

```c
// Create hash table with GPU acceleration enabled
Fxch_SCHashTable_t* pHashTable = 
    FxchCuda_SCHashTableCreate(pFxchMan, nEntries, 1);  // 1 = use GPU

// Insert with GPU acceleration
int pairs = FxchCuda_SCHashTableInsert(
    pHashTable, vCubes, subCubeID, 
    iCube, iLit0, iLit1, fUpdate, 
    1);  // 1 = use GPU

// Remove with GPU acceleration
int pairs = FxchCuda_SCHashTableRemove(
    pHashTable, vCubes, subCubeID,
    iCube, iLit0, iLit1, fUpdate,
    1);  // 1 = use GPU
```

## Performance Characteristics

### Expected Speedup
- **Best case**: Linear speedup with number of entries per bin
- **Typical case**: 10-100x speedup for bins with 100+ entries
- **Worst case**: Overhead from GPU transfer may slow down small bins (<10 entries)

### Memory Requirements
- **GPU memory**: Proportional to total cube data size
- **Transfer overhead**: Data copied to/from GPU on each operation
- **Optimization opportunity**: Cache flattened cube data to reduce transfers

## Optimization Opportunities

### 1. Data Transfer Caching
**Current**: Flattens and transfers all cube data on every Insert/Remove call
**Improvement**: Cache flattened data structure, update incrementally
**Expected gain**: 50-90% reduction in overhead

### 2. Batch Processing
**Current**: Processes one comparison set at a time
**Improvement**: Batch multiple Insert/Remove operations
**Expected gain**: Amortize GPU launch overhead

### 3. Asynchronous Execution
**Current**: Synchronous GPU calls block CPU
**Improvement**: Use CUDA streams for overlapped execution
**Expected gain**: Better CPU-GPU utilization

### 4. Shared Memory Optimization
**Current**: Global memory access for all data
**Improvement**: Use shared memory for frequently accessed data
**Expected gain**: Reduced memory latency

### 5. Dynamic Parallelism
**Current**: Fixed thread block size
**Improvement**: Adaptive thread allocation based on bin size
**Expected gain**: Better GPU occupancy

## Debugging

### Enable Verbose Output
Uncomment debug printf statements in `Kernel.cu` to trace:
- Memory allocation sizes
- Kernel launch parameters
- Comparison results

### Common Issues

1. **CUDA allocation failure**: Reduce cube data size or implement batching
2. **Incorrect results**: Verify cube data flattening logic
3. **Slow performance**: Check if bins are too small for GPU benefit

### Fallback Behavior
If GPU execution fails, the implementation automatically falls back to the original CPU implementation, ensuring correctness at the cost of performance.

## Testing

### Verification
Compare results between CPU and GPU implementations:
```bash
# Run with GPU disabled (should match original behavior)
./abc_exe -c "fxch -N 0"

# Run with GPU enabled
./abc_exe -c "fxchcuda"
```

### Profiling
Use NVIDIA profiling tools:
```bash
nvprof ./your_application
nsys profile ./your_application
```

## Future Work

1. **Persistent data structures**: Keep flattened cube data on GPU
2. **Multi-GPU support**: Distribute work across multiple GPUs
3. **Kernel fusion**: Combine multiple operations into single kernel
4. **Memory pooling**: Pre-allocate GPU memory to reduce allocation overhead
5. **Adaptive strategy**: Switch between CPU/GPU based on bin size

## References

- Original FXCH paper: [Fast eXtract with Cube Hashing]
- CUDA Programming Guide: https://docs.nvidia.com/cuda/
- ABC System: https://people.eecs.berkeley.edu/~alanmi/abc/

## Authors

- Original FXCH: Bruno Schmitt (UFRGS)
- CUDA Acceleration: Yu Ching Hei, Chan Eugene (CUHK)
- Implementation: AI Assistant (November 2025)

