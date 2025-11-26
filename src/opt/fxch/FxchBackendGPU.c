#include "FxchBackend.h"

#if FXCH_USE_CUDA
#include "../fxchcuda/FxchCuda.h"

static Fxch_SCHashTable_t* FxchGPU_SCHashTableCreate(Fxch_Man_t* pFxchMan, int nEntries)
{
    return FxchCuda_SCHashTableCreate(pFxchMan, nEntries, 1);
}

static void FxchGPU_SCHashTableDelete(Fxch_SCHashTable_t* pSCHashTable)
{
    FxchCuda_SCHashTableDelete(pSCHashTable, 1);
}

static int FxchGPU_SCHashTableInsert(Fxch_SCHashTable_t* pSCHashTable, Vec_Wec_t* vCubes,
                                     uint32_t SubCubeID, uint32_t iCube, 
                                     uint32_t iLit0, uint32_t iLit1, char fUpdate)
{
    return FxchCuda_SCHashTableInsert(pSCHashTable, vCubes, SubCubeID, iCube, iLit0, iLit1, fUpdate, 1);
}

static int FxchGPU_SCHashTableRemove(Fxch_SCHashTable_t* pSCHashTable, Vec_Wec_t* vCubes,
                                     uint32_t SubCubeID, uint32_t iCube,
                                     uint32_t iLit0, uint32_t iLit1, char fUpdate)
{
    return FxchCuda_SCHashTableRemove(pSCHashTable, vCubes, SubCubeID, iCube, iLit0, iLit1, fUpdate, 1);
}

static Fxch_Backend_t g_FxchBackendGPU = {
    .SCHashTableCreate  = FxchGPU_SCHashTableCreate,
    .SCHashTableDelete  = FxchGPU_SCHashTableDelete,
    .SCHashTableInsert  = FxchGPU_SCHashTableInsert,
    .SCHashTableRemove  = FxchGPU_SCHashTableRemove,
};

Fxch_Backend_t* Fxch_GetBackendGPU()
{
    return &g_FxchBackendGPU;
}

#endif