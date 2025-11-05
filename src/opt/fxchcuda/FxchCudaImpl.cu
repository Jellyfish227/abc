#include "FxchCuda.h



__global__ void insert_subcubes_kernel(
    cuco::static_multimap<uint32_t, Fxch_SubCube_t>::device_ref map,
    uint32_t* d_subcubeIds,
    uint32_t* d_cubeIndices,
    uint32_t* d_iLit0,
    uint32_t* d_iLit1,
    int num_subcubes) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_subcubes) return;
    
    Fxch_SubCube_t entry = {
        .Id = d_subcubeIds[idx],
        .iCube = d_cubeIndices[idx],
        .iLit0 = d_iLit0[idx],
        .iLit1 = d_iLit1[idx]
    };
    
    map.insert(d_subcubeIds[idx], entry);
}


int FxchCuda_SCHashTableInsert(
    Fxch_SCHashTable_GPU_t* d_table,
    uint32_t* h_subcubeIds,
    uint32_t* h_cubeIndices,
    uint32_t* h_iLit0,
    uint32_t* h_iLit1,
    int numSubcubes)
{
    // Transfer to GPU
    uint32_t *d_subcubeIds, *d_cubeIndices, *d_iLit0, *d_iLit1;
    cudaMalloc(&d_subcubeIds, numSubcubes * sizeof(uint32_t));
    cudaMalloc(&d_cubeIndices, numSubcubes * sizeof(uint32_t));
    cudaMalloc(&d_iLit0, numSubcubes * sizeof(uint32_t));
    cudaMalloc(&d_iLit1, numSubcubes * sizeof(uint32_t));
    
    cudaMemcpy(d_subcubeIds, h_subcubeIds, numSubcubes * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cubeIndices, h_cubeIndices, numSubcubes * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_iLit0, h_iLit0, numSubcubes * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_iLit1, h_iLit1, numSubcubes * sizeof(uint32_t), cudaMemcpyHostToDevice);
    
    // Launch kernel
    auto map_ref = d_table->d_subcubeMap->get_device_ref();
    int blockSize = 256;
    int numBlocks = (numSubcubes + blockSize - 1) / blockSize;
    
    insert_subcubes_kernel<<<numBlocks, blockSize>>>(
        map_ref, d_subcubeIds, d_cubeIndices, d_iLit0, d_iLit1, numSubcubes);
    
    cudaDeviceSynchronize();
    
    // Cleanup
    cudaFree(d_subcubeIds);
    cudaFree(d_cubeIndices);
    cudaFree(d_iLit0);
    cudaFree(d_iLit1);
    
    return numSubcubes;
}

void FxchCuda_SCHashTableDeleteGPU(Fxch_SCHashTable_GPU_t* d_table)
{
    if (!d_table) return;
    
    cudaFree(d_table->d_flatData);
    cudaFree(d_table->d_levelSizes);
    cudaFree(d_table->d_cubeOffsets);
    delete d_table->d_subcubeMap;
    delete d_table;
}

// Stub implementations for other functions
void FxchCuda_SCHashTableRemoveGPU(Fxch_SCHashTable_GPU_t* d_table,
                                   uint32_t SubCubeID, uint32_t iCube,
                                   uint32_t iLit0, uint32_t iLit1)
{
    // TODO: Implement removal kernel if needed
}


// this kernel is 100% fucking wrong
__global__ void compare_subcubes_kernel(
    cuco::static_multimap<uint32_t, Fxch_SubCube_t>::device_ref map,
    int* d_flatData,
    int* d_cubeOffsets,
    int* d_levelSizes,
    uint32_t* d_newSubcubeIds,
    uint32_t* d_newCubeIndices,
    uint32_t* d_newILit0,
    uint32_t* d_newILit1,
    int num_new_subcubes,
    /* output: matching pairs */) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_new_subcubes) return;
    
    uint32_t subcubeId = d_newSubcubeIds[idx];
    
    // Find all entries with this SubCubeID
    auto [it, it_end] = map.find(subcubeId);
    
    // Compare new entry against each existing match
    Fxch_SubCube_t new_entry = {
        .Id = subcubeId,
        .iCube = d_newCubeIndices[idx],
        .iLit0 = d_newILit0[idx],
        .iLit1 = d_newILit1[idx]
    };
    
    while (it != it_end) {
        Fxch_SubCube_t existing_entry = *it;
        
        if (gpu_semantic_compare(new_entry, existing_entry, 
                                 d_flatData, d_cubeOffsets, d_levelSizes)) {
            // Record match (write to output buffer atomically)
            record_divisor_pair(new_entry.iCube, existing_entry.iCube, /* ... */);
        }
        ++it;
    }
}

__device__ bool gpu_semantic_compare(Fxch_SubCube_t entry0,
                                      Fxch_SubCube_t entry1,
                                      int* flatData,
                                      int* cubeOffsets,
                                      int* levelSizes) {
    // Stage 1: Basic checks
    if (entry0.Id != entry1.Id) return false;
    
    int offset0 = cubeOffsets[entry0.iCube];
    int offset1 = cubeOffsets[entry1.iCube];
    int size0 = levelSizes[entry0.iCube];
    int size1 = levelSizes[entry1.iCube];
    
    if (size0 == 0 || size1 == 0) return false;
    if (flatData[offset0] != flatData[offset1]) return false;
    
    // Stage 2: Output compatibility (if you have vOutputID, compare it)
    // ... (requires passing vOutputID to GPU)
    
    // Stage 3: Conflicting literals check
    if (entry0.iLit1 > 0 && entry1.iLit1 > 0) {
        int lit0_0 = flatData[offset0 + entry0.iLit0];
        int lit1_0 = flatData[offset0 + entry0.iLit1];
        int lit0_1 = flatData[offset1 + entry1.iLit0];
        int lit1_1 = flatData[offset1 + entry1.iLit1];
        
        if (lit0_0 == lit0_1 || lit0_0 == lit1_1 || 
            lit1_0 == lit0_1 || lit1_0 == lit1_1) {
            return false;
        }
    }
    
    // Stage 4-5: Normalize and compare cores
    // Build temporary normalized vectors and compare
    // (This requires local arrays or shared memory)
    
    return gpu_compare_normalized_cores(entry0, entry1, flatData, cubeOffsets, levelSizes);
}


void FxchCuda_SCHashTableDelete(Fxch_SCHashTable_GPU_t* d_table) {
    cudaFree(d_table->d_flatData);
    cudaFree(d_table->d_levelSizes);
    cudaFree(d_table->d_cubeOffsets);
    delete d_table->d_subcubeMap;
    free(d_table);
}

unsigned int FxchCuda_SCHashTableMemoryGPU(Fxch_SCHashTable_GPU_t* d_table)
{
    return d_table->totalElements * sizeof(int) +
           d_table->numCubes * 2 * sizeof(int) +
           d_table->nEntries * sizeof(Fxch_SubCube_t);
}

void FxchCuda_SCHashTablePrintGPU(Fxch_SCHashTable_GPU_t* d_table)
{
    printf("GPU Hash Table: %d cubes, %d elements\n", 
           d_table->numCubes, d_table->totalElements);
}


