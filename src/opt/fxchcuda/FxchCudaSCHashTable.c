/**CFile****************************************************************

  FileName    [ FxchCudaSCHashTable.c ]

  PackageName [ Fast eXtract with GPU Accelerated Cube Hashing (FXCHCUDA) ]

  Synopsis    [ Sub-cubes hash table implementation with CUDA acceleration ]

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
///                     TYPE DEFINITIONS                             ///
////////////////////////////////////////////////////////////////////////

// Comparison result structure
typedef struct {
    int match;
    int shouldContinue;
    int cubeIndex;
} ComparisonResult_t;

// GPU data cache to avoid repeated transfers
typedef struct {
    int* pCubeData;
    int* pCubeOffsets;
    int* pCubeSizes;
    int  nTotalSize;
    int  nMaxCube;
    int  nCacheVersion;  // Invalidate when cubes change
    int  nCachedCubeCount;
} GPUDataCache_t;

// Global cache (could be per-hash-table for better encapsulation)
static GPUDataCache_t g_GPUCache = {NULL, NULL, NULL, 0, 0, 0, 0};

////////////////////////////////////////////////////////////////////////
///                     EXTERNAL DECLARATIONS                        ///
////////////////////////////////////////////////////////////////////////

// External CUDA kernel interface
// Note: extern "C" is in the Kernel.cu file
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
    void* pResults);

// MurmurHash function from original implementation
static inline void MurmurHash3_x86_32 ( const void* key,
                                        int len,
                                        uint32_t seed,
                                        void* out )
{
    const uint8_t* data = (const uint8_t*)key;
    const int nblocks = len / 4;

    uint32_t h1 = seed;

    const uint32_t c1 = 0xcc9e2d51;
    const uint32_t c2 = 0x1b873593;

    const uint8_t * tail;
    uint32_t k1;

    const uint32_t * blocks = (const uint32_t *)(data + nblocks*4);
    int i;

    for(i = -nblocks; i; i++)
    {
        uint32_t k1 = blocks[i];

        k1 *= c1;
        k1 = (k1 << 15) | (k1 >> (32 - 15));
        k1 *= c2;

        h1 ^= k1;
        h1 = (h1 << 13) | (h1 >> (32 - 13));
        h1 = h1*5+0xe6546b64;
    }

    tail = (const uint8_t*)(data + nblocks*4);

    k1 = 0;

    switch(len & 3)
    {
        case 3: k1 ^= tail[2] << 16;
        case 2: k1 ^= tail[1] << 8;
        case 1: k1 ^= tail[0];
              k1 *= c1; k1 = (k1 << 15) | (k1 >> (32 - 15)); k1 *= c2; h1 ^= k1;
    };

    h1 ^= len;

    h1 ^= h1 >> 16;
    h1 *= 0x85ebca6b;
    h1 ^= h1 >> 13;
    h1 *= 0xc2b2ae35;
    h1 ^= h1 >> 16;

    *(uint32_t*)out = h1;
}

////////////////////////////////////////////////////////////////////////
///                     HELPER FUNCTIONS                             ///
////////////////////////////////////////////////////////////////////////

/**Function*************************************************************

  Synopsis    [ Free GPU data cache ]

  Description []

***********************************************************************/
static void FreeGPUDataCache()
{
    if (g_GPUCache.pCubeData) {
        ABC_FREE(g_GPUCache.pCubeData);
        ABC_FREE(g_GPUCache.pCubeOffsets);
        ABC_FREE(g_GPUCache.pCubeSizes);
        g_GPUCache.pCubeData = NULL;
        g_GPUCache.pCubeOffsets = NULL;
        g_GPUCache.pCubeSizes = NULL;
        g_GPUCache.nTotalSize = 0;
        g_GPUCache.nMaxCube = 0;
        g_GPUCache.nCachedCubeCount = 0;
    }
}

/**Function*************************************************************

  Synopsis    [ Prepare cube data for GPU transfer with caching ]

  Description [ Flattens cube data into contiguous arrays. Uses cache
                to avoid repeated work. ]

***********************************************************************/
static int PrepareCubeDataForGPU(
    Fxch_Man_t* pFxchMan,
    Vec_Wec_t* vCubes,
    int** ppCubeData,
    int** ppCubeOffsets,
    int** ppCubeSizes,
    int* pTotalSize,
    int* pMaxCube)
{
    int nCubes = Vec_WecSize(vCubes);
    int i, j, offset = 0;
    int totalSize = 0;
    
    // Check if we can use cached data
    if (g_GPUCache.pCubeData != NULL && g_GPUCache.nCachedCubeCount == nCubes) {
        // Use cached data
        *ppCubeData = g_GPUCache.pCubeData;
        *ppCubeOffsets = g_GPUCache.pCubeOffsets;
        *ppCubeSizes = g_GPUCache.pCubeSizes;
        *pTotalSize = g_GPUCache.nTotalSize;
        *pMaxCube = g_GPUCache.nMaxCube;
        return 0;  // Return 0 to indicate using cache (don't free)
    }
    
    // Free old cache if cube count changed
    FreeGPUDataCache();
    
    // Calculate total size needed
    for (i = 0; i < nCubes; i++)
        totalSize += Vec_IntSize(Vec_WecEntry(vCubes, i));
    
    // Allocate arrays
    *ppCubeData = ABC_ALLOC(int, totalSize == 0 ? 1 : totalSize);
    *ppCubeOffsets = ABC_ALLOC(int, nCubes == 0 ? 1 : nCubes);
    *ppCubeSizes = ABC_ALLOC(int, nCubes == 0 ? 1 : nCubes);
    
    // Fill arrays
    for (i = 0; i < nCubes; i++) {
        Vec_Int_t* vCube = Vec_WecEntry(vCubes, i);
        int size = Vec_IntSize(vCube);
        
        (*ppCubeOffsets)[i] = offset;
        (*ppCubeSizes)[i] = size;
        
        for (j = 0; j < size; j++)
            (*ppCubeData)[offset++] = Vec_IntEntry(vCube, j);
    }
    
    *pTotalSize = totalSize;
    *pMaxCube = nCubes;
    
    // Update cache
    g_GPUCache.pCubeData = *ppCubeData;
    g_GPUCache.pCubeOffsets = *ppCubeOffsets;
    g_GPUCache.pCubeSizes = *ppCubeSizes;
    g_GPUCache.nTotalSize = totalSize;
    g_GPUCache.nMaxCube = nCubes;
    g_GPUCache.nCachedCubeCount = nCubes;
    
    return 1;  // Return 1 to indicate new allocation (cache updated, don't free)
}

////////////////////////////////////////////////////////////////////////
///                     FUNCTION DEFINITIONS                         ///
////////////////////////////////////////////////////////////////////////

// These are all wrappers with fallbacks to the original fxch implementation

Fxch_SCHashTable_t* FxchCuda_SCHashTableCreate( Fxch_Man_t* pFxchMan, int nEntries, short int usingGpu )
{
    // For now, always use original implementation
    // GPU-specific initialization can be added here if needed
    return Fxch_SCHashTableCreate(pFxchMan, nEntries);
}


void FxchCuda_SCHashTableDelete( Fxch_SCHashTable_t* pSCHashTable, short int usingGpu)
{
    // Clean up GPU cache
    if (usingGpu) {
        FreeGPUDataCache();
    }
    
    // Call original delete function
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
    // Early exit - use CPU version
    if (!usingGpu) {
        return Fxch_SCHashTableInsert(pSCHashTable, vCubes, SubCubeID, iCube, iLit0, iLit1, fUpdate);
    }
    
    int iNewEntry;
    int Pairs = 0;
    uint32_t BinID;
    Fxch_SCHashTable_Entry_t* pBin;
    Fxch_SubCube_t* pNewEntry;
    
    // Hash the subcube ID to get bin
    MurmurHash3_x86_32( ( void* ) &SubCubeID, sizeof( int ), 0x9747b28c, &BinID);
    pBin = pSCHashTable->pBins + (BinID & pSCHashTable->SizeMask);
    
    // Initialize bin if needed
    if ( pBin->vSCData == NULL )
    {
        pBin->vSCData = ABC_CALLOC( Fxch_SubCube_t, 16 );
        pBin->Size = 0;
        pBin->Cap = 16;
    }
    else if ( pBin->Size == pBin->Cap )
    {
        assert(pBin->Cap <= 0xAAAA);
        pBin->Cap = ( pBin->Cap >> 1 ) * 3;
        pBin->vSCData = ABC_REALLOC( Fxch_SubCube_t, pBin->vSCData, pBin->Cap );
    }
    
    // Add new entry
    iNewEntry = pBin->Size++;
    pBin->vSCData[iNewEntry].Id = SubCubeID;
    pBin->vSCData[iNewEntry].iCube = iCube;
    pBin->vSCData[iNewEntry].iLit0 = iLit0;
    pBin->vSCData[iNewEntry].iLit1 = iLit1;
    pSCHashTable->nEntries++;
    
    if ( pBin->Size == 1 )
        return 0;
    
    // Use GPU for parallel comparison
    pNewEntry = &( pBin->vSCData[iNewEntry] );
    
    // Prepare data for GPU
    int* pCubeData = NULL;
    int* pCubeOffsets = NULL;
    int* pCubeSizes = NULL;
    int totalSize, maxCube;
    int cacheStatus;  // Not used but good to track
    
    cacheStatus = PrepareCubeDataForGPU(pSCHashTable->pFxchMan, vCubes, 
                                        &pCubeData, &pCubeOffsets, &pCubeSizes,
                                        &totalSize, &maxCube);
    (void)cacheStatus;  // Suppress unused warning
    
    // Allocate results array
    ComparisonResult_t* pResults = ABC_ALLOC(ComparisonResult_t, pBin->Size - 1);
    
    // Launch CUDA kernel
    int cudaResult = LaunchParallelEntryCompare(
        (void*)pNewEntry,
        (void*)pBin->vSCData,
        pBin->Size - 1,
        pCubeData,
        pCubeOffsets,
        pCubeSizes,
        maxCube,
        Vec_IntArray(pSCHashTable->pFxchMan->vOutputID),
        pSCHashTable->pFxchMan->nSizeOutputID,
        (void*)pResults);
    
    if (cudaResult == 0) {
        // Process results from GPU
        int iEntry;
        for (iEntry = 0; iEntry < (int)pBin->Size - 1; iEntry++) {
            if (!pResults[iEntry].match)
                continue;
            
            Fxch_SubCube_t* pEntry = &( pBin->vSCData[iEntry] );
            int* pOutputID0 = Vec_IntEntryP( pSCHashTable->pFxchMan->vOutputID, 
                                            pEntry->iCube * pSCHashTable->pFxchMan->nSizeOutputID );
            int* pOutputID1 = Vec_IntEntryP( pSCHashTable->pFxchMan->vOutputID, 
                                            pNewEntry->iCube * pSCHashTable->pFxchMan->nSizeOutputID );
            
            // Handle the matching entry
            if ( ( pEntry->iLit0 == 0 ) || ( pNewEntry->iLit0 == 0 ) )
            {
                Vec_Int_t* vCube0 = Fxch_ManGetCube( pSCHashTable->pFxchMan, pEntry->iCube );
                Vec_Int_t* vCube1 = Fxch_ManGetCube( pSCHashTable->pFxchMan, pNewEntry->iCube );

                if ( Vec_IntSize( vCube0 ) > Vec_IntSize( vCube1 ) )
                {
                    Vec_IntPush( pSCHashTable->pFxchMan->vSCC, pEntry->iCube );
                    Vec_IntPush( pSCHashTable->pFxchMan->vSCC, pNewEntry->iCube );
                }
                else
                {
                    Vec_IntPush( pSCHashTable->pFxchMan->vSCC, pNewEntry->iCube );
                    Vec_IntPush( pSCHashTable->pFxchMan->vSCC, pEntry->iCube );
                }

                continue;
            }

            int Base = Fxch_DivCreate( pSCHashTable->pFxchMan, pEntry, pNewEntry );

            if ( Base < 0 )
                continue;

            int Result = 0;
            int i, z;
            for ( i = 0; i < pSCHashTable->pFxchMan->nSizeOutputID; i++ )
                Result += Fxch_CountOnes( pOutputID0[i] & pOutputID1[i] );

            int iNewDiv = -1;
            for ( z = 0; z < Result; z++ )
                iNewDiv = Fxch_DivAdd( pSCHashTable->pFxchMan, fUpdate, 0, Base );

            Vec_WecPush( pSCHashTable->pFxchMan->vDivCubePairs, iNewDiv, pEntry->iCube );
            Vec_WecPush( pSCHashTable->pFxchMan->vDivCubePairs, iNewDiv, pNewEntry->iCube );

            Pairs++;
        }
    } else {
        // CUDA failed, fall back to CPU version
        printf("CUDA kernel failed, falling back to CPU\n");
        ABC_FREE(pResults);
        // Don't free cached data
        
        // Remove the entry we added and use original CPU version
        pBin->Size--;
        pSCHashTable->nEntries--;
        return Fxch_SCHashTableInsert(pSCHashTable, vCubes, SubCubeID, iCube, iLit0, iLit1, fUpdate);
    }
    
    // Clean up (but not cached data)
    ABC_FREE(pResults);
    
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
    // Early exit - use CPU version
    if (!usingGpu) {
        return Fxch_SCHashTableRemove(pSCHashTable, vCubes, SubCubeID, iCube, iLit0, iLit1, fUpdate);
    }
    
    int iEntry;
    int Pairs = 0;
    uint32_t BinID;
    Fxch_SCHashTable_Entry_t* pBin;
    Fxch_SubCube_t* pEntry;
    
    MurmurHash3_x86_32( ( void* ) &SubCubeID, sizeof( int ), 0x9747b28c, &BinID);
    pBin = pSCHashTable->pBins + (BinID & pSCHashTable->SizeMask);

    if ( pBin->Size == 1 )
    {
        pBin->Size = 0;
        return 0;
    }

    // Find the entry to remove
    for ( iEntry = 0; iEntry < (int)pBin->Size; iEntry++ )
        if ( pBin->vSCData[iEntry].iCube == iCube )
            break;

    assert( ( iEntry != (int)pBin->Size ) && ( pBin->Size != 0 ) );

    pEntry = &( pBin->vSCData[iEntry] );
    
    // Prepare data for GPU
    int* pCubeData = NULL;
    int* pCubeOffsets = NULL;
    int* pCubeSizes = NULL;
    int totalSize, maxCube;
    int cacheStatus;
    
    cacheStatus = PrepareCubeDataForGPU(pSCHashTable->pFxchMan, vCubes, 
                                        &pCubeData, &pCubeOffsets, &pCubeSizes,
                                        &totalSize, &maxCube);
    (void)cacheStatus;  // Suppress unused warning
    
    // Allocate results array - compare against all other entries
    ComparisonResult_t* pResults = ABC_ALLOC(ComparisonResult_t, pBin->Size);
    
    // Launch CUDA kernel
    int cudaResult = LaunchParallelEntryCompare(
        (void*)pEntry,
        (void*)pBin->vSCData,
        pBin->Size,
        pCubeData,
        pCubeOffsets,
        pCubeSizes,
        maxCube,
        Vec_IntArray(pSCHashTable->pFxchMan->vOutputID),
        pSCHashTable->pFxchMan->nSizeOutputID,
        (void*)pResults);
    
    if (cudaResult == 0) {
        // Process results from GPU
        int idx;
        for ( idx = 0; idx < (int)pBin->Size; idx++ )
        {
            if ( idx == iEntry || !pResults[idx].match )
                continue;

            Fxch_SubCube_t* pNextEntry = &( pBin->vSCData[idx] );
            int* pOutputID0 = Vec_IntEntryP( pSCHashTable->pFxchMan->vOutputID, 
                                            pEntry->iCube * pSCHashTable->pFxchMan->nSizeOutputID );
            int* pOutputID1 = Vec_IntEntryP( pSCHashTable->pFxchMan->vOutputID, 
                                            pNextEntry->iCube * pSCHashTable->pFxchMan->nSizeOutputID );
            
            if ( pEntry->iLit0 == 0 || pNextEntry->iLit0 == 0 )
                continue;

            int Base = Fxch_DivCreate( pSCHashTable->pFxchMan, pNextEntry, pEntry );

            if ( Base < 0 )
                continue;

            int Result = 0;
            int i, z;
            for ( i = 0; i < pSCHashTable->pFxchMan->nSizeOutputID; i++ )
                Result += Fxch_CountOnes( pOutputID0[i] & pOutputID1[i] );

            int iDiv = -1;
            for ( z = 0; z < Result; z++ )
                iDiv = Fxch_DivRemove( pSCHashTable->pFxchMan, fUpdate, 0, Base );

            Vec_Int_t* vDivCubePairs = Vec_WecEntry( pSCHashTable->pFxchMan->vDivCubePairs, iDiv );
            int iCube0, iCube1;
            Vec_IntForEachEntryDouble( vDivCubePairs, iCube0, iCube1, i )
                if ( ( iCube0 == (int)pNextEntry->iCube && iCube1 == (int)pEntry->iCube )  ||
                     ( iCube0 == (int)pEntry->iCube && iCube1 == (int)pNextEntry->iCube ) )
                {
                    Vec_IntDrop( vDivCubePairs, i+1 );
                    Vec_IntDrop( vDivCubePairs, i );
                }
            if ( Vec_IntSize( vDivCubePairs ) == 0 )
                Vec_IntErase( vDivCubePairs );

            Pairs++;
        }
    } else {
        // CUDA failed, fall back to CPU version
        printf("CUDA kernel failed in Remove, falling back to CPU\n");
        ABC_FREE(pResults);
        // Don't free cached data
        return Fxch_SCHashTableRemove(pSCHashTable, vCubes, SubCubeID, iCube, iLit0, iLit1, fUpdate);
    }

    // Clean up (but not cached data)
    ABC_FREE(pResults);

    // Remove the entry from the bin
    memmove(pBin->vSCData + iEntry, pBin->vSCData + iEntry + 1, 
            (pBin->Size - iEntry - 1) * sizeof(*pBin->vSCData));
    pBin->Size -= 1;

    return Pairs;
}


unsigned int FxchCuda_SCHashTableMemory( Fxch_SCHashTable_t* pHashTable, short int usingGpu) 
{
    // Memory usage is same as original plus cache
    unsigned int memory = Fxch_SCHashTableMemory(pHashTable);
    
    if (usingGpu && g_GPUCache.pCubeData != NULL) {
        // Add cache memory
        memory += g_GPUCache.nTotalSize * sizeof(int);
        memory += g_GPUCache.nMaxCube * sizeof(int) * 2; // offsets + sizes
    }
    
    return memory;
}


void FxchCuda_SCHashTablePrint( Fxch_SCHashTable_t* pHashTable, short int usingGpu)
{
    // Early exit
    if (!usingGpu) {
        Fxch_SCHashTablePrint(pHashTable);
        return;
    }
};

////////////////////////////////////////////////////////////////////////
///                       END OF FILE                                ///
////////////////////////////////////////////////////////////////////////

ABC_NAMESPACE_IMPL_END
