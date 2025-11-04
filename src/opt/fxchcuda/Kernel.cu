/**CFile****************************************************************

  FileName    [ Kernel.cu ]

  PackageName [ Fast eXtract with GPU Accelerated Cube Hashing (FXCHCUDA) ]

  Synopsis    [ CUDA kernel for parallel entry comparison ]

  Author      [ Yu Ching Hei, Chan Eugene ]

  Affiliation [ CUHK ]

  Date        [ Ver. 1.0. Started - October 22, 2025. ]

  Revision    []

***********************************************************************/

#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdint>

// Structure matching Fxch_SubCube_t for GPU
struct GPU_SubCube_t {
    uint32_t Id;
    uint32_t iCube;
    uint32_t iLit0 : 16;
    uint32_t iLit1 : 16;
};

// Structure for flattened cube data
struct GPU_CubeData {
    int* cube_offsets;      // Offset in cube_data for each cube
    int* cube_sizes;        // Size of each cube
    int* cube_data;         // Flattened cube data
    int* output_ids;        // Flattened output IDs (nCubes * nSizeOutputID)
    int nCubes;
    int nSizeOutputID;
};

// Device function to get cube data
__device__ inline int getCubeEntry(GPU_CubeData* cubeData, int cubeIdx, int litIdx) {
    int offset = cubeData->cube_offsets[cubeIdx];
    return cubeData->cube_data[offset + litIdx];
}

__device__ inline int getCubeSize(GPU_CubeData* cubeData, int cubeIdx) {
    return cubeData->cube_sizes[cubeIdx];
}

__device__ inline int* getOutputID(GPU_CubeData* cubeData, int cubeIdx) {
    return &cubeData->output_ids[cubeIdx * cubeData->nSizeOutputID];
}

// Device function to build subcube and compare
__device__ int compareSubCubes(GPU_CubeData* cubeData, GPU_SubCube_t* sc0, GPU_SubCube_t* sc1) {
    // Quick checks
    int size0 = getCubeSize(cubeData, sc0->iCube);
    int size1 = getCubeSize(cubeData, sc1->iCube);
    
    if (size0 == 0 || size1 == 0)
        return 0;
    
    if (getCubeEntry(cubeData, sc0->iCube, 0) != getCubeEntry(cubeData, sc1->iCube, 0))
        return 0;
    
    if (sc0->Id != sc1->Id)
        return 0;
    
    // Check output ID overlap
    int* pOutputID0 = getOutputID(cubeData, sc0->iCube);
    int* pOutputID1 = getOutputID(cubeData, sc1->iCube);
    int result = 0;
    
    for (int i = 0; i < cubeData->nSizeOutputID && result == 0; i++)
        result = (pOutputID0[i] & pOutputID1[i]);
    
    if (result == 0)
        return 0;
    
    // Check literal conflicts
    if (sc0->iLit1 > 0 && sc1->iLit1 > 0) {
        int lit0_0 = (sc0->iLit0 > 0) ? getCubeEntry(cubeData, sc0->iCube, sc0->iLit0) : -1;
        int lit1_0 = (sc0->iLit1 > 0) ? getCubeEntry(cubeData, sc0->iCube, sc0->iLit1) : -1;
        int lit0_1 = (sc1->iLit0 > 0) ? getCubeEntry(cubeData, sc1->iCube, sc1->iLit0) : -1;
        int lit1_1 = (sc1->iLit1 > 0) ? getCubeEntry(cubeData, sc1->iCube, sc1->iLit1) : -1;
        
        if (lit0_0 == lit0_1 || lit0_0 == lit1_1 || lit1_0 == lit0_1 || lit1_0 == lit1_1)
            return 0;
    }
    
    // Build subcubes and compare using local memory
    const int MAX_SUBCUBE_SIZE = 256;  // Maximum expected subcube size
    int subcube0[MAX_SUBCUBE_SIZE];
    int subcube1[MAX_SUBCUBE_SIZE];
    int idx0 = 0, idx1 = 0;
    
    // Build subcube0
    if (sc0->iLit0 > 0) {
        for (int i = 0; i < sc0->iLit0 && idx0 < MAX_SUBCUBE_SIZE; i++) {
            subcube0[idx0++] = getCubeEntry(cubeData, sc0->iCube, i);
        }
    } else {
        for (int i = 0; i < size0 && idx0 < MAX_SUBCUBE_SIZE; i++) {
            subcube0[idx0++] = getCubeEntry(cubeData, sc0->iCube, i);
        }
    }
    
    // Drop literal if needed
    if (sc0->iLit1 > 0 && idx0 > 0) {
        int dropIdx = (sc0->iLit0 < sc0->iLit1) ? sc0->iLit1 - 1 : sc0->iLit1;
        if (dropIdx < idx0) {
            for (int i = dropIdx; i < idx0 - 1; i++) {
                subcube0[i] = subcube0[i + 1];
            }
            idx0--;
        }
    }
    
    // Build subcube1
    if (sc1->iLit0 > 0) {
        for (int i = 0; i < sc1->iLit0 && idx1 < MAX_SUBCUBE_SIZE; i++) {
            subcube1[idx1++] = getCubeEntry(cubeData, sc1->iCube, i);
        }
    } else {
        for (int i = 0; i < size1 && idx1 < MAX_SUBCUBE_SIZE; i++) {
            subcube1[idx1++] = getCubeEntry(cubeData, sc1->iCube, i);
        }
    }
    
    // Drop literal if needed
    if (sc1->iLit1 > 0 && idx1 > 0) {
        int dropIdx = (sc1->iLit0 < sc1->iLit1) ? sc1->iLit1 - 1 : sc1->iLit1;
        if (dropIdx < idx1) {
            for (int i = dropIdx; i < idx1 - 1; i++) {
                subcube1[i] = subcube1[i + 1];
            }
            idx1--;
        }
    }
    
    // Compare subcubes
    if (idx0 != idx1)
        return 0;
    
    for (int i = 0; i < idx0; i++) {
        if (subcube0[i] != subcube1[i])
            return 0;
    }
    
    return 1;
}

// CUDA kernel for parallel entry comparison
__global__ void entryCompareKernel(
    GPU_CubeData* cubeData,
    GPU_SubCube_t* entries0,
    GPU_SubCube_t* entries1,
    int* results,
    int nComparisons
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < nComparisons) {
        results[idx] = compareSubCubes(cubeData, &entries0[idx], &entries1[idx]);
    }
}

// Host wrapper function
extern "C" {
    void launchEntryCompareKernel(
        GPU_CubeData* h_cubeData,
        GPU_SubCube_t* h_entries0,
        GPU_SubCube_t* h_entries1,
        int* h_results,
        int nComparisons
    ) {
        // Allocate device memory
        GPU_CubeData* d_cubeData;
        GPU_SubCube_t* d_entries0;
        GPU_SubCube_t* d_entries1;
        int* d_results;
        
        cudaError_t err;
        err = cudaMalloc(&d_cubeData, sizeof(GPU_CubeData));
        if (err != cudaSuccess) return;
        err = cudaMalloc(&d_entries0, nComparisons * sizeof(GPU_SubCube_t));
        if (err != cudaSuccess) { cudaFree(d_cubeData); return; }
        err = cudaMalloc(&d_entries1, nComparisons * sizeof(GPU_SubCube_t));
        if (err != cudaSuccess) { cudaFree(d_cubeData); cudaFree(d_entries0); return; }
        err = cudaMalloc(&d_results, nComparisons * sizeof(int));
        if (err != cudaSuccess) { cudaFree(d_cubeData); cudaFree(d_entries0); cudaFree(d_entries1); return; }
        
        // Allocate and copy cube data structure
        GPU_CubeData cubeDataHost = *h_cubeData;
        
        int* d_cube_offsets;
        int* d_cube_sizes;
        int* d_cube_data;
        int* d_output_ids;
        
        int totalCubeDataSize = 0;
        for (int i = 0; i < h_cubeData->nCubes; i++) {
            totalCubeDataSize += h_cubeData->cube_sizes[i];
        }
        
        err = cudaMalloc(&d_cube_offsets, h_cubeData->nCubes * sizeof(int));
        if (err != cudaSuccess) goto cleanup1;
        err = cudaMalloc(&d_cube_sizes, h_cubeData->nCubes * sizeof(int));
        if (err != cudaSuccess) goto cleanup2;
        err = cudaMalloc(&d_cube_data, totalCubeDataSize * sizeof(int));
        if (err != cudaSuccess) goto cleanup3;
        err = cudaMalloc(&d_output_ids, h_cubeData->nCubes * h_cubeData->nSizeOutputID * sizeof(int));
        if (err != cudaSuccess) goto cleanup4;
        
        cudaMemcpy(d_cube_offsets, h_cubeData->cube_offsets, h_cubeData->nCubes * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_cube_sizes, h_cubeData->cube_sizes, h_cubeData->nCubes * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_cube_data, h_cubeData->cube_data, totalCubeDataSize * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_output_ids, h_cubeData->output_ids, h_cubeData->nCubes * h_cubeData->nSizeOutputID * sizeof(int), cudaMemcpyHostToDevice);
        
        cubeDataHost.cube_offsets = d_cube_offsets;
        cubeDataHost.cube_sizes = d_cube_sizes;
        cubeDataHost.cube_data = d_cube_data;
        cubeDataHost.output_ids = d_output_ids;
        
        cudaMemcpy(d_cubeData, &cubeDataHost, sizeof(GPU_CubeData), cudaMemcpyHostToDevice);
        cudaMemcpy(d_entries0, h_entries0, nComparisons * sizeof(GPU_SubCube_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_entries1, h_entries1, nComparisons * sizeof(GPU_SubCube_t), cudaMemcpyHostToDevice);
        
        // Launch kernel
        int threadsPerBlock = 256;
        int blocksPerGrid = (nComparisons + threadsPerBlock - 1) / threadsPerBlock;
        
        entryCompareKernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_cubeData, d_entries0, d_entries1, d_results, nComparisons
        );
        
        // Synchronize to ensure kernel completion
        cudaDeviceSynchronize();
        
        // Copy results back
        cudaMemcpy(h_results, d_results, nComparisons * sizeof(int), cudaMemcpyDeviceToHost);
        
        // Free device memory
        cudaFree(d_output_ids);
        cleanup4:
        cudaFree(d_cube_data);
        cleanup3:
        cudaFree(d_cube_sizes);
        cleanup2:
        cudaFree(d_cube_offsets);
        cleanup1:
        cudaFree(d_cubeData);
        cudaFree(d_entries0);
        cudaFree(d_entries1);
        cudaFree(d_results);
    }
}

