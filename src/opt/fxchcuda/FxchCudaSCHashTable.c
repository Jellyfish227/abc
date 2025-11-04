/**CFile****************************************************************

  FileName    [ FxchCudaSCHashTable.c ]

  PackageName [ Fast eXtract with GPU Accelerated Cube Hashing (FXCHCUDA) ]

  Synopsis    [ Sub-cubes hash table implementation ]

  Author      [ Yu Ching Hei, Chan Eugene ]

  Affiliation [ CUHK ]

  Date        [ Ver. 1.0. Started - October 22, 2025. ]

  Revision    []

***********************************************************************/
#include "FxchCuda.h"
#include <stdlib.h>
#include <string.h>

#if (__GNUC__ >= 8)
  #pragma GCC diagnostic ignored "-Wimplicit-fallthrough"
#endif

ABC_NAMESPACE_IMPL_START

////////////////////////////////////////////////////////////////////////
///                     FUNCTION DEFINITIONS                         ///
////////////////////////////////////////////////////////////////////////

// Forward declaration for CUDA kernel wrapper
extern void launchEntryCompareKernel(
    GPU_CubeData_t* h_cubeData,
    GPU_SubCube_t* h_entries0,
    GPU_SubCube_t* h_entries1,
    int* h_results,
    int nComparisons
);

// Helper functions (copied from FxchSCHashTable.c since they're static inline)
static inline void MurmurHash3_x86_32(const void* key, int len, uint32_t seed, void* out) {
    const uint8_t* data = (const uint8_t*)key;
    const int nblocks = len / 4;
    uint32_t h1 = seed;
    const uint32_t c1 = 0xcc9e2d51;
    const uint32_t c2 = 0x1b873593;
    const uint8_t* tail;
    uint32_t k1;
    const uint32_t* blocks = (const uint32_t*)(data + nblocks * 4);
    int i;

    for (i = -nblocks; i; i++) {
        uint32_t k1 = blocks[i];
        k1 *= c1;
        k1 = (k1 << 15) | (k1 >> (32 - 15));
        k1 *= c2;
        h1 ^= k1;
        h1 = (h1 << 13) | (h1 >> (32 - 13));
        h1 = h1 * 5 + 0xe6546b64;
    }

    tail = (const uint8_t*)(data + nblocks * 4);
    k1 = 0;

    switch (len & 3) {
        case 3: k1 ^= tail[2] << 16;
        case 2: k1 ^= tail[1] << 8;
        case 1: k1 ^= tail[0];
              k1 *= c1; k1 = (k1 << 15) | (k1 >> (32 - 15)); k1 *= c2; h1 ^= k1;
    }

    h1 ^= len;
    h1 ^= h1 >> 16;
    h1 *= 0x85ebca6b;
    h1 ^= h1 >> 13;
    h1 *= 0xc2b2ae35;
    h1 ^= h1 >> 16;

    *(uint32_t*)out = h1;
}

static inline Fxch_SCHashTable_Entry_t* Fxch_SCHashTableBin(Fxch_SCHashTable_t* pSCHashTable,
                                                             unsigned int SubCubeID) {
    return pSCHashTable->pBins + (SubCubeID & pSCHashTable->SizeMask);
}

// Structure matching GPU structures
typedef struct {
    int* cube_offsets;
    int* cube_sizes;
    int* cube_data;
    int* output_ids;
    int nCubes;
    int nSizeOutputID;
} GPU_CubeData_t;

typedef struct {
    uint32_t Id;
    uint32_t iCube;
    uint32_t iLit0 : 16;
    uint32_t iLit1 : 16;
} GPU_SubCube_t;

// Helper function to prepare cube data for GPU
static GPU_CubeData_t* prepareCubeDataForGPU(Fxch_SCHashTable_t* pSCHashTable, Vec_Wec_t* vCubes) {
    GPU_CubeData_t* gpuData = ABC_CALLOC(GPU_CubeData_t, 1);
    int nCubes = Vec_WecSize(vCubes);
    gpuData->nCubes = nCubes;
    gpuData->nSizeOutputID = pSCHashTable->pFxchMan->nSizeOutputID;
    
    // Allocate arrays
    gpuData->cube_offsets = ABC_CALLOC(int, nCubes);
    gpuData->cube_sizes = ABC_CALLOC(int, nCubes);
    
    // Calculate total size and offsets
    int totalSize = 0;
    int i;
    for (i = 0; i < nCubes; i++) {
        Vec_Int_t* vCube = Vec_WecEntry(vCubes, i);
        gpuData->cube_offsets[i] = totalSize;
        gpuData->cube_sizes[i] = Vec_IntSize(vCube);
        totalSize += Vec_IntSize(vCube);
    }
    
    // Allocate and fill cube data
    gpuData->cube_data = ABC_CALLOC(int, totalSize);
    int offset = 0;
    for (i = 0; i < nCubes; i++) {
        Vec_Int_t* vCube = Vec_WecEntry(vCubes, i);
        int size = Vec_IntSize(vCube);
        int j;
        for (j = 0; j < size; j++) {
            gpuData->cube_data[offset + j] = Vec_IntEntry(vCube, j);
        }
        offset += size;
    }
    
    // Allocate and fill output IDs
    gpuData->output_ids = ABC_CALLOC(int, nCubes * gpuData->nSizeOutputID);
    for (i = 0; i < nCubes; i++) {
        int* pOutputID = Vec_IntEntryP(pSCHashTable->pFxchMan->vOutputID, i * gpuData->nSizeOutputID);
        int j;
        for (j = 0; j < gpuData->nSizeOutputID; j++) {
            gpuData->output_ids[i * gpuData->nSizeOutputID + j] = pOutputID[j];
        }
    }
    
    return gpuData;
}

// Helper function to free GPU cube data
static void freeCubeDataForGPU(GPU_CubeData_t* gpuData) {
    if (gpuData) {
        ABC_FREE(gpuData->cube_offsets);
        ABC_FREE(gpuData->cube_sizes);
        ABC_FREE(gpuData->cube_data);
        ABC_FREE(gpuData->output_ids);
        ABC_FREE(gpuData);
    }
}

// CUDA-accelerated batch entry comparison
static int cudaBatchEntryCompare(
    Fxch_SCHashTable_t* pSCHashTable,
    Vec_Wec_t* vCubes,
    Fxch_SubCube_t* entries0,
    Fxch_SubCube_t* entries1,
    int nComparisons,
    int* results
) {
    // Prepare GPU data (can be cached, but for now we prepare each time)
    GPU_CubeData_t* gpuCubeData = prepareCubeDataForGPU(pSCHashTable, vCubes);
    
    // Convert entries to GPU format
    GPU_SubCube_t* gpuEntries0 = ABC_CALLOC(GPU_SubCube_t, nComparisons);
    GPU_SubCube_t* gpuEntries1 = ABC_CALLOC(GPU_SubCube_t, nComparisons);
    
    int i;
    for (i = 0; i < nComparisons; i++) {
        gpuEntries0[i].Id = entries0[i].Id;
        gpuEntries0[i].iCube = entries0[i].iCube;
        gpuEntries0[i].iLit0 = entries0[i].iLit0;
        gpuEntries0[i].iLit1 = entries0[i].iLit1;
        
        gpuEntries1[i].Id = entries1[i].Id;
        gpuEntries1[i].iCube = entries1[i].iCube;
        gpuEntries1[i].iLit0 = entries1[i].iLit0;
        gpuEntries1[i].iLit1 = entries1[i].iLit1;
    }
    
    // Launch CUDA kernel
    launchEntryCompareKernel(gpuCubeData, gpuEntries0, gpuEntries1, results, nComparisons);
    
    // Cleanup
    ABC_FREE(gpuEntries0);
    ABC_FREE(gpuEntries1);
    freeCubeDataForGPU(gpuCubeData);
    
    return 0;
}

// These are all wrappers with fallbacks to the original fxch implementation

Fxch_SCHashTable_t* FxchCuda_SCHashTableCreate( Fxch_Man_t* pFxchMan, int nEntries, short int usingGpu )
{
    // Early exit
    if (!usingGpu) {
        return Fxch_SCHashTableCreate(pFxchMan, nEntries);
    }
    
    return Fxch_SCHashTableCreate(pFxchMan, nEntries);
}


void FxchCuda_SCHashTableDelete( Fxch_SCHashTable_t* pSCHashTable, short int usingGpu)
{
    // Early exit
    if (!usingGpu) {
        Fxch_SCHashTableDelete(pSCHashTable);
        return;
    }
    
    Fxch_SCHashTableDelete(pSCHashTable);
}

int FxchCuda_SCHashTableInsert( Fxch_SCHashTable_t* pSCHashTable,
                            Vec_Wec_t* vCubes,
                            uint32_t SubCubeID,
                            uint32_t iCube,
                            uint32_t iLit0,
                            uint32_t iLit1,
                            char fUpdate,
                            short int usingGpu )
{
    // Early exit
    if (!usingGpu) {
        return Fxch_SCHashTableInsert(pSCHashTable, vCubes, SubCubeID, iCube, iLit0, iLit1, fUpdate);
    }
    
    // GPU-accelerated version
    int iNewEntry;
    int Pairs = 0;
    uint32_t BinID;
    Fxch_SCHashTable_Entry_t* pBin;
    Fxch_SubCube_t* pNewEntry;
    int iEntry;

    // Use same hash function as original
    MurmurHash3_x86_32((void*)&SubCubeID, sizeof(int), 0x9747b28c, &BinID);
    pBin = Fxch_SCHashTableBin(pSCHashTable, BinID);

    if (pBin->vSCData == NULL) {
        pBin->vSCData = ABC_CALLOC(Fxch_SubCube_t, 16);
        pBin->Size = 0;
        pBin->Cap = 16;
    } else if (pBin->Size == pBin->Cap) {
        assert(pBin->Cap <= 0xAAAA);
        pBin->Cap = (pBin->Cap >> 1) * 3;
        pBin->vSCData = ABC_REALLOC(Fxch_SubCube_t, pBin->vSCData, pBin->Cap);
    }

    iNewEntry = pBin->Size++;
    pBin->vSCData[iNewEntry].Id = SubCubeID;
    pBin->vSCData[iNewEntry].iCube = iCube;
    pBin->vSCData[iNewEntry].iLit0 = iLit0;
    pBin->vSCData[iNewEntry].iLit1 = iLit1;
    pSCHashTable->nEntries++;

    if (pBin->Size == 1)
        return 0;

    pNewEntry = &(pBin->vSCData[iNewEntry]);
    
    // Batch comparisons using CUDA
    int nComparisons = pBin->Size - 1;
    if (nComparisons > 0) {
        Fxch_SubCube_t* entries0 = ABC_CALLOC(Fxch_SubCube_t, nComparisons);
        Fxch_SubCube_t* entries1 = ABC_CALLOC(Fxch_SubCube_t, nComparisons);
        int* results = ABC_CALLOC(int, nComparisons);
        
        // Prepare comparison pairs
        for (iEntry = 0; iEntry < nComparisons; iEntry++) {
            entries0[iEntry] = pBin->vSCData[iEntry];
            entries1[iEntry] = *pNewEntry;
        }
        
        // Launch CUDA kernel
        cudaBatchEntryCompare(pSCHashTable, vCubes, entries0, entries1, nComparisons, results);
        
        // Process results
        for (iEntry = 0; iEntry < nComparisons; iEntry++) {
            Fxch_SubCube_t* pEntry = &(pBin->vSCData[iEntry]);
            int* pOutputID0 = Vec_IntEntryP(pSCHashTable->pFxchMan->vOutputID, pEntry->iCube * pSCHashTable->pFxchMan->nSizeOutputID);
            int* pOutputID1 = Vec_IntEntryP(pSCHashTable->pFxchMan->vOutputID, pNewEntry->iCube * pSCHashTable->pFxchMan->nSizeOutputID);
            int Result = 0;
            int Base;
            int iNewDiv = -1, i, z;

            if ((pEntry->iLit1 != 0 && pNewEntry->iLit1 == 0) || (pEntry->iLit1 == 0 && pNewEntry->iLit1 != 0))
                continue;

            // Use CUDA result
            if (!results[iEntry])
                continue;

            if ((pEntry->iLit0 == 0) || (pNewEntry->iLit0 == 0)) {
                Vec_Int_t* vCube0 = Fxch_ManGetCube(pSCHashTable->pFxchMan, pEntry->iCube),
                         * vCube1 = Fxch_ManGetCube(pSCHashTable->pFxchMan, pNewEntry->iCube);

                if (Vec_IntSize(vCube0) > Vec_IntSize(vCube1)) {
                    Vec_IntPush(pSCHashTable->pFxchMan->vSCC, pEntry->iCube);
                    Vec_IntPush(pSCHashTable->pFxchMan->vSCC, pNewEntry->iCube);
                } else {
                    Vec_IntPush(pSCHashTable->pFxchMan->vSCC, pNewEntry->iCube);
                    Vec_IntPush(pSCHashTable->pFxchMan->vSCC, pEntry->iCube);
                }

                continue;
            }

            Base = Fxch_DivCreate(pSCHashTable->pFxchMan, pEntry, pNewEntry);

            if (Base < 0)
                continue;

            for (i = 0; i < pSCHashTable->pFxchMan->nSizeOutputID; i++)
                Result += Fxch_CountOnes(pOutputID0[i] & pOutputID1[i]);

            for (z = 0; z < Result; z++)
                iNewDiv = Fxch_DivAdd(pSCHashTable->pFxchMan, fUpdate, 0, Base);

            Vec_WecPush(pSCHashTable->pFxchMan->vDivCubePairs, iNewDiv, pEntry->iCube);
            Vec_WecPush(pSCHashTable->pFxchMan->vDivCubePairs, iNewDiv, pNewEntry->iCube);

            Pairs++;
        }
        
        ABC_FREE(entries0);
        ABC_FREE(entries1);
        ABC_FREE(results);
    }

    return Pairs;
}


int FxchCuda_SCHashTableRemove( Fxch_SCHashTable_t* pSCHashTable,
                            Vec_Wec_t* vCubes,
                            uint32_t SubCubeID,
                            uint32_t iCube,
                            uint32_t iLit0,
                            uint32_t iLit1,
                            char fUpdate,
                            short int usingGpu )
{
    // Early exit
    if (!usingGpu) {
        return Fxch_SCHashTableRemove(pSCHashTable, vCubes, SubCubeID, iCube, iLit0, iLit1, fUpdate);
    }
    
    // For now, fallback to CPU version for remove
    return Fxch_SCHashTableRemove(pSCHashTable, vCubes, SubCubeID, iCube, iLit0, iLit1, fUpdate);
}


unsigned int FxchCuda_SCHashTableMemory( Fxch_SCHashTable_t* pHashTable, short int usingGpu) 
{
    // Early exit
    if (!usingGpu) {
        return Fxch_SCHashTableMemory(pHashTable);
    }
    
    return Fxch_SCHashTableMemory(pHashTable);
}


void FxchCuda_SCHashTablePrint( Fxch_SCHashTable_t* pHashTable, short int usingGpu)
{
    // Early exit
    if (!usingGpu) {
        Fxch_SCHashTablePrint(pHashTable);
        return;
    }
    
    Fxch_SCHashTablePrint(pHashTable);
}

////////////////////////////////////////////////////////////////////////
///                       END OF FILE                                ///
////////////////////////////////////////////////////////////////////////

ABC_NAMESPACE_IMPL_END
