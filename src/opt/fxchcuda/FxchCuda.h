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
void FxchCuda_ManDivDoubleCube(Fxch_Man_t *pFxchMan, int fAdd, int fUpdate);

/*===== FxchCudaSCHashTable.c ============================================*/

// Simplified GPU structure - subcube metadata only
typedef struct
{
    // Hash table for subcube insertion and lookup
    cuco::static_multimap<uint32_t, Fxch_SubCube_t> *d_subcubeMap;
    
    // Hash table sizing
    size_t capacity;      // Allocated capacity of hash table
    int nSubcubes;        // Number of subcubes stored (for validation/debugging)
} Fxch_SCHashTable_GPU_t;


typedef struct Fxch_SCHashTable_GPU_t_ Fxch_SCHashTable_GPU_t;

void FxchCuda_SCHashTableRemoveGPU(
    Fxch_SCHashTable_GPU_t *d_table,
    uint32_t SubCubeID,
    uint32_t iCube,
    uint32_t iLit0,
    uint32_t iLit1);

int FxchCuda_SCHashTableInsert(
    Fxch_SCHashTable_GPU_t *d_table,
    uint32_t *h_subcubeIds,
    uint32_t *h_cubeIndices,
    uint32_t *h_iLit0,
    uint32_t *h_iLit1,
    int numSubcubes);

int FxchCuda_SCHashTableCompare(
    Fxch_SCHashTable_GPU_t *d_table,
    int matchingPairs);


unsigned int FxchCuda_SCHashTableMemory(Fxch_SCHashTable_t *, short int usingGpu);
void FxchCuda_SCHashTablePrint(Fxch_SCHashTable_t *, short int usingGpu);

ABC_NAMESPACE_HEADER_END

#endif

////////////////////////////////////////////////////////////////////////
///                       END OF FILE                                ///
////////////////////////////////////////////////////////////////////////
