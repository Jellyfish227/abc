/**CFile****************************************************************

  FileName    [ Kernel.cu ]

  PackageName [ Fast eXtract with GPU Accelerated Cube Hashing (FXCHCUDA) ]

  Synopsis    [ CUDA kernels for parallelized entry comparison - ASYNC VERSION ]

  Author      [ AI Assistant ]

  Affiliation [ CUHK ]

  Date        [ Ver. 2.0. Started - November 2025. ]

  Revision    [ Async execution with streams and persistent memory ]

***********************************************************************/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>

////////////////////////////////////////////////////////////////////////
///                     PERSISTENT GPU MEMORY                        ///
////////////////////////////////////////////////////////////////////////

// Persistent GPU memory pool to avoid repeated allocation
typedef struct {
    // Device pointers for bulk data
    int* d_cubeData;
    int* d_cubeOffsets;
    int* d_cubeSizes;
    int* d_outputID;
    
    // Device pointers for per-call data (now persistent)
    void* d_newEntry;
    void* d_binEntries;
    void* d_results;
    
    // Allocated sizes for bulk data
    size_t cubeDataCapacity;
    size_t cubeMetaCapacity;
    size_t outputIDCapacity;
    
    // Allocated sizes for per-call data
    size_t newEntryCapacity;
    size_t binEntriesCapacity;
    size_t resultsCapacity;
    
    // CUDA stream and events for async operations
    cudaStream_t stream;
    cudaEvent_t transferComplete;
    
    // Validity flag
    int initialized;
} GPUMemoryPool_t;

static GPUMemoryPool_t g_gpuPool = {NULL, NULL, NULL, NULL, NULL, NULL, NULL, 
                                     0, 0, 0, 0, 0, 0, NULL, NULL, 0};

// Device-side subcube data structure
typedef struct {
    uint32_t Id;
    uint32_t iCube;
    uint32_t iLit0 : 16;
    uint32_t iLit1 : 16;
} DevSubCube_t;

// Result structure for each comparison
typedef struct {
    int match;
    int shouldContinue;
    int cubeIndex;
} ComparisonResult_t;

////////////////////////////////////////////////////////////////////////
///                     HELPER FUNCTIONS                             ///
////////////////////////////////////////////////////////////////////////

/**Function*************************************************************

  Synopsis    [ Initialize GPU memory pool ]

***********************************************************************/
static int InitGPUMemoryPool()
{
    if (g_gpuPool.initialized)
        return 0;
    
    // Create CUDA stream for async operations
    cudaError_t err = cudaStreamCreate(&g_gpuPool.stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to create CUDA stream: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    // Create CUDA event for transfer tracking
    err = cudaEventCreate(&g_gpuPool.transferComplete);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to create CUDA event: %s\n", cudaGetErrorString(err));
        cudaStreamDestroy(g_gpuPool.stream);
        return -1;
    }
    
    // Pre-allocate reasonable sizes for bulk data (will grow if needed)
    g_gpuPool.cubeDataCapacity = 10 * 1024 * 1024;  // 10M ints = 40MB
    g_gpuPool.cubeMetaCapacity = 100000;             // 100K cubes
    g_gpuPool.outputIDCapacity = 100000 * 32;        // 100K cubes * 32 ints
    
    // Initialize per-call data capacities (will allocate on first use)
    g_gpuPool.newEntryCapacity = 0;
    g_gpuPool.binEntriesCapacity = 0;
    g_gpuPool.resultsCapacity = 0;
    
    g_gpuPool.initialized = 1;
    return 0;
}

/**Function*************************************************************

  Synopsis    [ Free GPU memory pool ]

***********************************************************************/
static void FreeGPUMemoryPool()
{
    if (!g_gpuPool.initialized)
        return;
    
    // Free bulk data
    if (g_gpuPool.d_cubeData) cudaFree(g_gpuPool.d_cubeData);
    if (g_gpuPool.d_cubeOffsets) cudaFree(g_gpuPool.d_cubeOffsets);
    if (g_gpuPool.d_cubeSizes) cudaFree(g_gpuPool.d_cubeSizes);
    if (g_gpuPool.d_outputID) cudaFree(g_gpuPool.d_outputID);
    
    // Free per-call data
    if (g_gpuPool.d_newEntry) cudaFree(g_gpuPool.d_newEntry);
    if (g_gpuPool.d_binEntries) cudaFree(g_gpuPool.d_binEntries);
    if (g_gpuPool.d_results) cudaFree(g_gpuPool.d_results);
    
    // Destroy stream and event
    if (g_gpuPool.stream) cudaStreamDestroy(g_gpuPool.stream);
    if (g_gpuPool.transferComplete) cudaEventDestroy(g_gpuPool.transferComplete);
    
    // Reset all pointers and capacities
    g_gpuPool.d_cubeData = NULL;
    g_gpuPool.d_cubeOffsets = NULL;
    g_gpuPool.d_cubeSizes = NULL;
    g_gpuPool.d_outputID = NULL;
    g_gpuPool.d_newEntry = NULL;
    g_gpuPool.d_binEntries = NULL;
    g_gpuPool.d_results = NULL;
    g_gpuPool.stream = NULL;
    g_gpuPool.transferComplete = NULL;
    g_gpuPool.cubeDataCapacity = 0;
    g_gpuPool.cubeMetaCapacity = 0;
    g_gpuPool.outputIDCapacity = 0;
    g_gpuPool.newEntryCapacity = 0;
    g_gpuPool.binEntriesCapacity = 0;
    g_gpuPool.resultsCapacity = 0;
    g_gpuPool.initialized = 0;
}

////////////////////////////////////////////////////////////////////////
///                     CUDA KERNEL FUNCTIONS                       ///
////////////////////////////////////////////////////////////////////////

__device__ int DeviceVecIntEqual(const int* arr1, int size1, const int* arr2, int size2)
{
    if (size1 != size2)
        return 0;
    
    for (int i = 0; i < size1; i++)
        if (arr1[i] != arr2[i])
            return 0;
    
    return 1;
}

__global__ void ParallelEntryCompareKernel(
    DevSubCube_t* pNewEntry,
    DevSubCube_t* pBinEntries,
    int nBinSize,
    int* pCubeData,
    int* pCubeOffsets,
    int* pCubeSizes,
    int* pOutputID,
    int nSizeOutputID,
    ComparisonResult_t* pResults,
    int iNewEntry)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= nBinSize)
        return;
    
    pResults[idx].match = 0;
    pResults[idx].shouldContinue = 0;
    pResults[idx].cubeIndex = idx;
    
    DevSubCube_t* pEntry = &pBinEntries[idx];
    
    // Early exit conditions
    if ((pEntry->iLit1 != 0 && pNewEntry->iLit1 == 0) || 
        (pEntry->iLit1 == 0 && pNewEntry->iLit1 != 0))
        return;
    
    int iCube0 = pEntry->iCube;
    int iCube1 = pNewEntry->iCube;
    
    if (iCube0 >= nSizeOutputID || iCube1 >= nSizeOutputID)
        return;
    
    int* vCube0 = &pCubeData[pCubeOffsets[iCube0]];
    int* vCube1 = &pCubeData[pCubeOffsets[iCube1]];
    int nSize0 = pCubeSizes[iCube0];
    int nSize1 = pCubeSizes[iCube1];
    
    if (nSize0 == 0 || nSize1 == 0)
        return;
    
    if (vCube0[0] != vCube1[0])
        return;
    
    if (pEntry->Id != pNewEntry->Id)
        return;
    
    // Check OutputID intersection
    int* pOutputID0 = &pOutputID[iCube0 * nSizeOutputID];
    int* pOutputID1 = &pOutputID[iCube1 * nSizeOutputID];
    int Result = 0;
    
    for (int i = 0; i < nSizeOutputID; i++)
        Result |= (pOutputID0[i] & pOutputID1[i]);
    
    if (Result == 0)
        return;
    
    // Check for literal conflicts
    if (pEntry->iLit1 > 0 && pNewEntry->iLit1 > 0) {
        int lit0_0 = (pEntry->iLit0 < nSize0) ? vCube0[pEntry->iLit0] : 0;
        int lit1_0 = (pEntry->iLit1 < nSize0) ? vCube0[pEntry->iLit1] : 0;
        int lit0_1 = (pNewEntry->iLit0 < nSize1) ? vCube1[pNewEntry->iLit0] : 0;
        int lit1_1 = (pNewEntry->iLit1 < nSize1) ? vCube1[pNewEntry->iLit1] : 0;
        
        if (lit0_0 == lit0_1 || lit0_0 == lit1_1 ||
            lit1_0 == lit0_1 || lit1_0 == lit1_1)
            return;
    }
    
    // Build subcubes (stack allocation)
    int subCube0[256];
    int subCube1[256];
    int subSize0 = 0;
    int subSize1 = 0;
    
    // Build subCube0
    if (pEntry->iLit0 > 0) {
        for (int i = 0; i < nSize0; i++) {
            if (i != pEntry->iLit0)
                subCube0[subSize0++] = vCube0[i];
        }
    } else {
        for (int i = 0; i < nSize0; i++)
            subCube0[subSize0++] = vCube0[i];
    }
    
    if (pEntry->iLit1 > 0) {
        int dropIdx = (pEntry->iLit0 < pEntry->iLit1) ? 
                      pEntry->iLit1 - 1 : pEntry->iLit1;
        if (dropIdx < subSize0) {
            for (int i = dropIdx; i < subSize0 - 1; i++)
                subCube0[i] = subCube0[i + 1];
            subSize0--;
        }
    }
    
    // Build subCube1
    if (pNewEntry->iLit0 > 0) {
        for (int i = 0; i < nSize1; i++) {
            if (i != pNewEntry->iLit0)
                subCube1[subSize1++] = vCube1[i];
        }
    } else {
        for (int i = 0; i < nSize1; i++)
            subCube1[subSize1++] = vCube1[i];
    }
    
    if (pNewEntry->iLit1 > 0) {
        int dropIdx = (pNewEntry->iLit0 < pNewEntry->iLit1) ? 
                      pNewEntry->iLit1 - 1 : pNewEntry->iLit1;
        if (dropIdx < subSize1) {
            for (int i = dropIdx; i < subSize1 - 1; i++)
                subCube1[i] = subCube1[i + 1];
            subSize1--;
        }
    }
    
    // Final comparison
    if (DeviceVecIntEqual(subCube0, subSize0, subCube1, subSize1)) {
        pResults[idx].match = 1;
        pResults[idx].shouldContinue = 1;
    }
}

////////////////////////////////////////////////////////////////////////
///                     HOST INTERFACE FUNCTIONS                    ///
////////////////////////////////////////////////////////////////////////

extern "C" {

/**Function*************************************************************

  Synopsis    [ Initialize CUDA system ]

***********************************************************************/
int InitCUDASystem()
{
    return InitGPUMemoryPool();
}

/**Function*************************************************************

  Synopsis    [ Cleanup CUDA system ]

***********************************************************************/
void CleanupCUDASystem()
{
    FreeGPUMemoryPool();
}

/**Function*************************************************************

  Synopsis    [ Async entry comparison with persistent memory ]

***********************************************************************/
int LaunchParallelEntryCompare(
    void* pNewEntry,
    void* pBinEntries,
    int nBinSize,
    int* pCubeData,
    int* pCubeOffsets,
    int* pCubeSizes,
    int nMaxCube,
    int* pOutputID,
    int nSizeOutputID,
    void* pResults,
    int bSkipCubeTransfer)
{
    if (nBinSize == 0)
        return 0;
    
    // Initialize GPU pool if needed
    if (!g_gpuPool.initialized) {
        if (InitGPUMemoryPool() != 0)
            return -1;
    }
    
    cudaError_t err;
    
    // Calculate sizes
    size_t newEntrySize = sizeof(DevSubCube_t);
    size_t binEntriesSize = nBinSize * sizeof(DevSubCube_t);
    size_t resultsSize = nBinSize * sizeof(ComparisonResult_t);
    
    int totalCubeData = 0;
    for (int i = 0; i < nMaxCube; i++)
        totalCubeData += pCubeSizes[i];
    
    size_t cubeDataSize = totalCubeData * sizeof(int);
    size_t cubeOffsetsSize = nMaxCube * sizeof(int);
    size_t cubeSizesSize = nMaxCube * sizeof(int);
    size_t outputIDSize = nMaxCube * nSizeOutputID * sizeof(int);
    
    // Allocate or reuse persistent memory for cube data
    if (cubeDataSize > g_gpuPool.cubeDataCapacity || !g_gpuPool.d_cubeData) {
        if (g_gpuPool.d_cubeData) cudaFree(g_gpuPool.d_cubeData);
        err = cudaMalloc(&g_gpuPool.d_cubeData, cubeDataSize);
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed for cubeData: %s\n", cudaGetErrorString(err));
            return -1;
        }
        g_gpuPool.cubeDataCapacity = cubeDataSize;
    }
    
    if (cubeOffsetsSize > g_gpuPool.cubeMetaCapacity * sizeof(int) || !g_gpuPool.d_cubeOffsets) {
        if (g_gpuPool.d_cubeOffsets) cudaFree(g_gpuPool.d_cubeOffsets);
        if (g_gpuPool.d_cubeSizes) cudaFree(g_gpuPool.d_cubeSizes);
        
        err = cudaMalloc(&g_gpuPool.d_cubeOffsets, cubeOffsetsSize);
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed for cubeOffsets: %s\n", cudaGetErrorString(err));
            return -1;
        }
        err = cudaMalloc(&g_gpuPool.d_cubeSizes, cubeSizesSize);
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed for cubeSizes: %s\n", cudaGetErrorString(err));
            return -1;
        }
        g_gpuPool.cubeMetaCapacity = nMaxCube;
    }
    
    if (outputIDSize > g_gpuPool.outputIDCapacity || !g_gpuPool.d_outputID) {
        if (g_gpuPool.d_outputID) cudaFree(g_gpuPool.d_outputID);
        err = cudaMalloc(&g_gpuPool.d_outputID, outputIDSize);
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed for outputID: %s\n", cudaGetErrorString(err));
            return -1;
        }
        g_gpuPool.outputIDCapacity = outputIDSize;
    }
    
    // Allocate or reuse persistent memory for per-call data
    if (newEntrySize > g_gpuPool.newEntryCapacity || !g_gpuPool.d_newEntry) {
        if (g_gpuPool.d_newEntry) cudaFree(g_gpuPool.d_newEntry);
        err = cudaMalloc(&g_gpuPool.d_newEntry, newEntrySize);
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed for newEntry: %s\n", cudaGetErrorString(err));
            return -1;
        }
        g_gpuPool.newEntryCapacity = newEntrySize;
    }
    
    if (binEntriesSize > g_gpuPool.binEntriesCapacity || !g_gpuPool.d_binEntries) {
        if (g_gpuPool.d_binEntries) cudaFree(g_gpuPool.d_binEntries);
        err = cudaMalloc(&g_gpuPool.d_binEntries, binEntriesSize);
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed for binEntries: %s\n", cudaGetErrorString(err));
            return -1;
        }
        g_gpuPool.binEntriesCapacity = binEntriesSize;
    }
    
    if (resultsSize > g_gpuPool.resultsCapacity || !g_gpuPool.d_results) {
        if (g_gpuPool.d_results) cudaFree(g_gpuPool.d_results);
        err = cudaMalloc(&g_gpuPool.d_results, resultsSize);
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed for results: %s\n", cudaGetErrorString(err));
            return -1;
        }
        g_gpuPool.resultsCapacity = resultsSize;
    }
    
    // Cast to proper types for use
    DevSubCube_t* d_newEntry = (DevSubCube_t*)g_gpuPool.d_newEntry;
    DevSubCube_t* d_binEntries = (DevSubCube_t*)g_gpuPool.d_binEntries;
    ComparisonResult_t* d_results = (ComparisonResult_t*)g_gpuPool.d_results;
    
    // Async copy to device using stream (always copy per-call data)
    cudaMemcpyAsync(d_newEntry, pNewEntry, newEntrySize, 
                    cudaMemcpyHostToDevice, g_gpuPool.stream);
    cudaMemcpyAsync(d_binEntries, pBinEntries, binEntriesSize, 
                    cudaMemcpyHostToDevice, g_gpuPool.stream);
    
    // Only transfer bulk data if not already on GPU (huge optimization!)
    if (!bSkipCubeTransfer) {
        cudaMemcpyAsync(g_gpuPool.d_cubeData, pCubeData, cubeDataSize, 
                        cudaMemcpyHostToDevice, g_gpuPool.stream);
        cudaMemcpyAsync(g_gpuPool.d_cubeOffsets, pCubeOffsets, cubeOffsetsSize, 
                        cudaMemcpyHostToDevice, g_gpuPool.stream);
        cudaMemcpyAsync(g_gpuPool.d_cubeSizes, pCubeSizes, cubeSizesSize, 
                        cudaMemcpyHostToDevice, g_gpuPool.stream);
        cudaMemcpyAsync(g_gpuPool.d_outputID, pOutputID, outputIDSize, 
                        cudaMemcpyHostToDevice, g_gpuPool.stream);
    }
    
    // Launch kernel on stream (async)
    int threadsPerBlock = 256;
    int blocks = (nBinSize + threadsPerBlock - 1) / threadsPerBlock;
    
    ParallelEntryCompareKernel<<<blocks, threadsPerBlock, 0, g_gpuPool.stream>>>(
        d_newEntry,
        d_binEntries,
        nBinSize,
        g_gpuPool.d_cubeData,
        g_gpuPool.d_cubeOffsets,
        g_gpuPool.d_cubeSizes,
        g_gpuPool.d_outputID,
        nSizeOutputID,
        d_results,
        nBinSize);
    
    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    // Async copy results back
    cudaMemcpyAsync(pResults, d_results, resultsSize, 
                    cudaMemcpyDeviceToHost, g_gpuPool.stream);
    
    // Record event after all async operations are queued
    err = cudaEventRecord(g_gpuPool.transferComplete, g_gpuPool.stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "Event record failed: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    // Wait for event (more efficient than stream sync for profiling)
    err = cudaEventSynchronize(g_gpuPool.transferComplete);
    if (err != cudaSuccess) {
        fprintf(stderr, "Event sync failed: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    // Persistent buffers are reused, no need to free
    return 0;
}

} // extern "C"

////////////////////////////////////////////////////////////////////////
///                       END OF FILE                                ///
////////////////////////////////////////////////////////////////////////
