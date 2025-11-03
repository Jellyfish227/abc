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
#include <cuda_runtime.h>

ABC_NAMESPACE_HEADER_START

/*===== FxChCuda.c =======================================================*/

// int FxchCuda_FastExtract( Vec_Wec_t* vCubes,
//                       int ObjIdMax,
//                       int nMaxDivExt,
//                       int fVerbose,
//                       int fVeryVerbose );

// int Abc_NtkFxchCudaPerform( Abc_Ntk_t* pNtk,
//                         int nMaxDivExt,
//                         int fVerbose,
//                         int fVeryVerbose );


/*===== FxchCudaSCHashTable.c ============================================*/
Fxch_SCHashTable_t* FxchCuda_SCHashTableCreate( Fxch_Man_t* pFxchMan, int nEntries, short int usingGpu );

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

/*===== Kernel.cu ============================================*/
extern int launch_kernel(int *pOutputID0, int *pOutputID1, int size);

ABC_NAMESPACE_HEADER_END

#endif

////////////////////////////////////////////////////////////////////////
///                       END OF FILE                                ///
////////////////////////////////////////////////////////////////////////

