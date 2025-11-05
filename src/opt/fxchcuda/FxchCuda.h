/**CFile****************************************************************

  FileName    [ FxchCuda.h ]

  PackageName [ Fast eXtract with GPU accelerated Cube Hashing (FXCHCUDA) ]

  Synopsis    [ External declarations of fast extract with GPU accelerated cube hashing. ]

  Author      [ Yu Ching Hei, Chan Eugene ]

  Affiliation [ CUHK ]

  Date        [ Ver. 1.0. Started - October 22, 2025. ]

  Revision    []

***********************************************************************/

#ifndef ABC__opt__fxchcuda__fxchcuda_h
#define ABC__opt_fxchcuda__fxchcuda_h

#include "base/abc/abc.h"
#include "opt/fxch/Fxch.h"
#include <cuco/static_multimap.cuh>
#include <thrust/scan.h>

ABC_NAMESPACE_HEADER_START

/*===== FxchCudaMan.c ============================================*/

// Entry point of FxchCuda essentially
void FxchCuda_ManDivDoubleCube( Fxch_Man_t* pFxchMan, int fAdd, int fUpdate );


/*===== FxchCudaSCHashTable.c ============================================*/

// GPU-specific structure (opaque to CPU code)
typedef struct {
    int* d_flatData;           // GPU flattened cube literals
    int* d_levelSizes;         // GPU literals per cube
    int* d_cubeOffsets;        // GPU cumulative offsets (for myself: we can avoid O(n) lookup for cubes by precomputing cubeOffsets for each cube)
    cuco::static_multimap<uint32_t, Fxch_SubCube_t>* d_subcubeMap;
    int totalElements;
    int numCubes;
    int nEntries;
} Fxch_SCHashTable_GPU_t;

typedef struct Fxch_SCHashTable_GPU_t_ Fxch_SCHashTable_GPU_t;

// GPU function declarations (implemented in FxchCuda.cu)
Fxch_SCHashTable_t* FxchCuda_TransferAndAllocateGPU(
    int* flatData,
    int* levelSizes,
    int totalElements,
    int numCubes,
    int nEntries
);

void FxchCuda_SCHashTableDeleteGPU(Fxch_SCHashTable_GPU_t* d_table);

int FxchCuda_SCHashTableInsertGPU(
    Fxch_SCHashTable_GPU_t* d_table,
    uint32_t* h_subcubeIds,      // Host arrays
    uint32_t* h_cubeIndices,
    uint32_t* h_iLit0,
    uint32_t* h_iLit1,
    int numSubcubes
);

void FxchCuda_SCHashTableRemoveGPU(
    Fxch_SCHashTable_GPU_t* d_table,
    uint32_t SubCubeID,
    uint32_t iCube,
    uint32_t iLit0,
    uint32_t iLit1
);

unsigned int FxchCuda_SCHashTableMemoryGPU(Fxch_SCHashTable_GPU_t* d_table);

void FxchCuda_SCHashTablePrintGPU(Fxch_SCHashTable_GPU_t* d_table);

void FxchCuda_SCHashTableDelete( Fxch_SCHashTable_t*, short int usingGpu );

int FxchCuda_SCHashTableInsert( Fxch_SCHashTable_t* pSCHashTable,
                            Vec_Wec_t* vCubes,
                            uint32_t SubCubeID,
                            uint32_t iCube,
                            uint32_t iLit0,
                            uint32_t iLit1,
                            char fUpdate,
                            short int usingGpu );


int FxchCuda_SCHashTableRemove( Fxch_SCHashTable_t* pSCHashTable,
                            Vec_Wec_t* vCubes,
                            uint32_t SubCubeID,
                            uint32_t iCube,
                            uint32_t iLit0,
                            uint32_t iLit1,
                            char fUpdate,
                            short int usingGpu );

unsigned int FxchCuda_SCHashTableMemory( Fxch_SCHashTable_t* , short int usingGpu);
void FxchCuda_SCHashTablePrint( Fxch_SCHashTable_t* , short int usingGpu);

#ifdef __cplusplus
extern "C" {
#endif

Fxch_SCHashTable_t* FxchCuda_TransferAndAllocateGPU(
    int* flatData, 
    int* levelSizes, 
    int totalElements, 
    int numLevels, 
    int nEntries
);

#ifdef __cplusplus
}
#endif

ABC_NAMESPACE_HEADER_END

#endif

////////////////////////////////////////////////////////////////////////
///                       END OF FILE                                ///
////////////////////////////////////////////////////////////////////////

