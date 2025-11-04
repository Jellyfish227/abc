/**CFile****************************************************************

  FileName    [ Kernel.cu ]

  PackageName [ Fast eXtract with GPU Accelerated Cube Hashing (FXCHCUDA) ]

  Synopsis    [ CUDA kernels for parallelized entry comparison ]

  Author      [ AI Assistant ]

  Affiliation [ CUHK ]

  Date        [ Ver. 1.0. Started - November 4, 2025. ]

  Revision    []

***********************************************************************/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            return -1; \
        } \
    } while(0)

// Device-side subcube data structure
typedef struct {
    uint32_t Id;
    uint32_t iCube;
    uint32_t iLit0 : 16;
    uint32_t iLit1 : 16;
} DevSubCube_t;

// Device-side cube data - simplified representation
typedef struct {
    int* pData;      // Pointer to cube data
    int  nSize;      // Size of the cube
    int  nAlloc;     // Allocated size
} DevCube_t;

// Result structure for each comparison
typedef struct {
    int match;           // 1 if entries match, 0 otherwise
    int shouldContinue;  // 1 if comparison should proceed, 0 if early exit
    int cubeIndex;       // Index of the comparison
} ComparisonResult_t;

////////////////////////////////////////////////////////////////////////
///                     CUDA KERNEL FUNCTIONS                       ///
////////////////////////////////////////////////////////////////////////

/**Function*************************************************************

  Synopsis    [ Device function to compare two integer arrays ]

  Description [ Returns 1 if arrays are equal, 0 otherwise ]

***********************************************************************/
__device__ int DeviceVecIntEqual(const int* arr1, int size1, const int* arr2, int size2)
{
    if (size1 != size2)
        return 0;
    
    for (int i = 0; i < size1; i++)
        if (arr1[i] != arr2[i])
            return 0;
    
    return 1;
}

/**Function*************************************************************

  Synopsis    [ CUDA kernel for parallel entry comparison ]

  Description [ Each thread compares the new entry against one existing entry ]

  Parameters:
    - pNewEntry: The new entry to compare against
    - pBinEntries: Array of existing entries in the bin
    - nBinSize: Number of entries in the bin (excluding the new one)
    - pCubeData: Flattened array of all cube data
    - pCubeOffsets: Offset for each cube in pCubeData
    - pCubeSizes: Size of each cube
    - pOutputID: Output ID data for all cubes
    - nSizeOutputID: Size of output ID per cube
    - pResults: Output array of comparison results
    - iNewEntry: Index of the new entry (typically nBinSize)

***********************************************************************/
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
    
    // Initialize result
    pResults[idx].match = 0;
    pResults[idx].shouldContinue = 0;
    pResults[idx].cubeIndex = idx;
    
    DevSubCube_t* pEntry = &pBinEntries[idx];
    
    // Check iLit1 compatibility (early exit condition)
    if ((pEntry->iLit1 != 0 && pNewEntry->iLit1 == 0) || 
        (pEntry->iLit1 == 0 && pNewEntry->iLit1 != 0))
        return;
    
    // Get cube pointers and sizes
    int iCube0 = pEntry->iCube;
    int iCube1 = pNewEntry->iCube;
    
    int* vCube0 = &pCubeData[pCubeOffsets[iCube0]];
    int* vCube1 = &pCubeData[pCubeOffsets[iCube1]];
    int nSize0 = pCubeSizes[iCube0];
    int nSize1 = pCubeSizes[iCube1];
    
    // Early exit checks
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
    
    // Build subcubes and compare
    // Allocate temporary arrays in shared or local memory
    int subCube0[256];  // Stack allocation - assumes max cube size
    int subCube1[256];
    int subSize0 = 0;
    int subSize1 = 0;
    
    // Build subCube0
    if (pEntry->iLit0 > 0) {
        // AppendSkip
        for (int i = 0; i < nSize0; i++) {
            if (i != pEntry->iLit0)
                subCube0[subSize0++] = vCube0[i];
        }
    } else {
        // Append all
        for (int i = 0; i < nSize0; i++)
            subCube0[subSize0++] = vCube0[i];
    }
    
    // Drop iLit1 if needed
    if (pEntry->iLit1 > 0) {
        int dropIdx = (pEntry->iLit0 < pEntry->iLit1) ? 
                      pEntry->iLit1 - 1 : pEntry->iLit1;
        if (dropIdx < subSize0) {
            // Remove element at dropIdx
            for (int i = dropIdx; i < subSize0 - 1; i++)
                subCube0[i] = subCube0[i + 1];
            subSize0--;
        }
    }
    
    // Build subCube1
    if (pNewEntry->iLit0 > 0) {
        // AppendSkip
        for (int i = 0; i < nSize1; i++) {
            if (i != pNewEntry->iLit0)
                subCube1[subSize1++] = vCube1[i];
        }
    } else {
        // Append all
        for (int i = 0; i < nSize1; i++)
            subCube1[subSize1++] = vCube1[i];
    }
    
    // Drop iLit1 if needed
    if (pNewEntry->iLit1 > 0) {
        int dropIdx = (pNewEntry->iLit0 < pNewEntry->iLit1) ? 
                      pNewEntry->iLit1 - 1 : pNewEntry->iLit1;
        if (dropIdx < subSize1) {
            // Remove element at dropIdx
            for (int i = dropIdx; i < subSize1 - 1; i++)
                subCube1[i] = subCube1[i + 1];
            subSize1--;
        }
    }
    
    // Final comparison
    int isEqual = DeviceVecIntEqual(subCube0, subSize0, subCube1, subSize1);
    
    if (isEqual) {
        pResults[idx].match = 1;
        pResults[idx].shouldContinue = 1;
    }
}

////////////////////////////////////////////////////////////////////////
///                     HOST INTERFACE FUNCTIONS                    ///
////////////////////////////////////////////////////////////////////////

extern "C" {

/**Function*************************************************************

  Synopsis    [ Host function to launch parallel entry comparison ]

  Description [ Returns array of comparison results, or NULL on error ]

  Parameters:
    - pNewEntry: New subcube entry
    - pBinEntries: Array of existing entries
    - nBinSize: Number of existing entries
    - pCubeData: Flattened cube data
    - pCubeOffsets: Offsets for each cube
    - pCubeSizes: Sizes for each cube
    - nMaxCube: Maximum cube index
    - pOutputID: Output ID array
    - nSizeOutputID: Size of output ID per cube
    - pResults: Output results array (must be pre-allocated)

  Returns: 0 on success, -1 on failure

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
    void* pResults)
{
    if (nBinSize == 0)
        return 0;
    
    DevSubCube_t* d_newEntry;
    DevSubCube_t* d_binEntries;
    int* d_cubeData;
    int* d_cubeOffsets;
    int* d_cubeSizes;
    int* d_outputID;
    ComparisonResult_t* d_results;
    
    // Calculate sizes
    size_t newEntrySize = sizeof(DevSubCube_t);
    size_t binEntriesSize = nBinSize * sizeof(DevSubCube_t);
    size_t resultsSize = nBinSize * sizeof(ComparisonResult_t);
    
    // Calculate total cube data size
    int totalCubeData = 0;
    for (int i = 0; i < nMaxCube; i++)
        totalCubeData += pCubeSizes[i];
    
    size_t cubeDataSize = totalCubeData * sizeof(int);
    size_t cubeOffsetsSize = nMaxCube * sizeof(int);
    size_t cubeSizesSize = nMaxCube * sizeof(int);
    size_t outputIDSize = nMaxCube * nSizeOutputID * sizeof(int);
    
    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_newEntry, newEntrySize));
    CUDA_CHECK(cudaMalloc(&d_binEntries, binEntriesSize));
    CUDA_CHECK(cudaMalloc(&d_cubeData, cubeDataSize));
    CUDA_CHECK(cudaMalloc(&d_cubeOffsets, cubeOffsetsSize));
    CUDA_CHECK(cudaMalloc(&d_cubeSizes, cubeSizesSize));
    CUDA_CHECK(cudaMalloc(&d_outputID, outputIDSize));
    CUDA_CHECK(cudaMalloc(&d_results, resultsSize));
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_newEntry, pNewEntry, newEntrySize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_binEntries, pBinEntries, binEntriesSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cubeData, pCubeData, cubeDataSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cubeOffsets, pCubeOffsets, cubeOffsetsSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cubeSizes, pCubeSizes, cubeSizesSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_outputID, pOutputID, outputIDSize, cudaMemcpyHostToDevice));
    
    // Launch kernel
    int threadsPerBlock = 256;
    int blocks = (nBinSize + threadsPerBlock - 1) / threadsPerBlock;
    
    ParallelEntryCompareKernel<<<blocks, threadsPerBlock>>>(
        d_newEntry,
        d_binEntries,
        nBinSize,
        d_cubeData,
        d_cubeOffsets,
        d_cubeSizes,
        d_outputID,
        nSizeOutputID,
        d_results,
        nBinSize);
    
    // Check for kernel errors
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy results back
    CUDA_CHECK(cudaMemcpy(pResults, d_results, resultsSize, cudaMemcpyDeviceToHost));
    
    // Free device memory
    cudaFree(d_newEntry);
    cudaFree(d_binEntries);
    cudaFree(d_cubeData);
    cudaFree(d_cubeOffsets);
    cudaFree(d_cubeSizes);
    cudaFree(d_outputID);
    cudaFree(d_results);
    
    return 0;
}

} // extern "C"

////////////////////////////////////////////////////////////////////////
///                       END OF FILE                                ///
////////////////////////////////////////////////////////////////////////

