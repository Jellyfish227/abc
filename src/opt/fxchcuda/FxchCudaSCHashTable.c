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

#if (__GNUC__ >= 8)
  #pragma GCC diagnostic ignored "-Wimplicit-fallthrough"
#endif

ABC_NAMESPACE_IMPL_START

////////////////////////////////////////////////////////////////////////
///                     FUNCTION DEFINITIONS                         ///
////////////////////////////////////////////////////////////////////////

// These are all wrappers with fallbacks to the original fxch implementation

Fxch_SCHashTable_t* FxchCuda_SCHashTableCreate( Fxch_Man_t* pFxchMan, int nEntries, bool usingGpu )
{
    return usingGpu ? NULL : Fxch_SCHashTableCreate(pFxchMan, nEntries);
}


void FxchCuda_SCHashTableDelete( Fxch_SCHashTable_t* pSCHashTable, bool usingGpu)
{
    return usingGpu ? NULL : Fxch_SCHashTableDelete(pSCHashTable);
}

int FxchCuda_SCHashTableInsert( Fxch_SCHashTable_t* pSCHashTable,
                            Vec_Wec_t* vCubes,
                            uint32_t SubCubeID,
                            uint32_t iCube,
                            uint32_t iLit0,
                            uint32_t iLit1,
                            char fUpdate,
                            bool usingGpu )
{
    return usingGpu ? NULL : Fxch_SCHashTableInsert(pSCHashTable, vCubes, SubCubeID, iCube, iLit0, iLit1, fUpdate);
}


int FxchCuda_SCHashTableRemove( Fxch_SCHashTable_t* pSCHashTable,
                            Vec_Wec_t* vCubes,
                            uint32_t SubCubeID,
                            uint32_t iCube,
                            uint32_t iLit0,
                            uint32_t iLit1,
                            char fUpdate,
                            bool usingGpu )
{
    return usingGpu ? NULL : Fxch_SCHashTableRemove(pSCHashTable, vCubes, SubCubeID, iCube, iLit0, iLit1, fUpdate);
}


unsigned int FxchCuda_SCHashTableMemory( Fxch_SCHashTable_t* pHashTable, bool usingGpu) 
{
    return usingGpu ? NULL : Fxch_SCHashTableMemory(pHashTable);
}


void FxchCuda_SCHashTablePrint( Fxch_SCHashTable_t* pHashTable, bool usingGpu)
{
    return usingGpu ? NULL : Fxch_SCHashTablePrint(pHashTable);
};
////////////////////////////////////////////////////////////////////////
///                       END OF FILE                                ///
////////////////////////////////////////////////////////////////////////

ABC_NAMESPACE_IMPL_END
