#include "FxchBackend.h"

static Fxch_SCHashTable_t* FxchCPU_SCHashTableCreate(Fxch_Man_t* pFxchMan, int nEntries)
{
    return Fxch_SCHashTableCreate(pFxchMan, nEntries);
}

static void FxchCPU_SCHashTableDelete(Fxch_SCHashTable_t* pSCHashTable)
{
    Fxch_SCHashTableDelete(pSCHashTable);
}

static int FxchCPU_SCHashTableInsert(Fxch_SCHashTable_t* pSCHashTable, Vec_Wec_t* vCubes,
                                     uint32_t SubCubeID, uint32_t iCube, 
                                     uint32_t iLit0, uint32_t iLit1, char fUpdate)
{
    return Fxch_SCHashTableInsert(pSCHashTable, vCubes, SubCubeID, iCube, iLit0, iLit1, fUpdate);
}

static int FxchCPU_SCHashTableRemove(Fxch_SCHashTable_t* pSCHashTable, Vec_Wec_t* vCubes,
                                     uint32_t SubCubeID, uint32_t iCube,
                                     uint32_t iLit0, uint32_t iLit1, char fUpdate)
{
    return Fxch_SCHashTableRemove(pSCHashTable, vCubes, SubCubeID, iCube, iLit0, iLit1, fUpdate);
}

static Fxch_Backend_t g_FxchBackendCPU = {
    .SCHashTableCreate  = FxchCPU_SCHashTableCreate,
    .SCHashTableDelete  = FxchCPU_SCHashTableDelete,
    .SCHashTableInsert  = FxchCPU_SCHashTableInsert,
    .SCHashTableRemove  = FxchCPU_SCHashTableRemove,
};

Fxch_Backend_t* Fxch_GetBackendCPU()
{
    return &g_FxchBackendCPU;
}