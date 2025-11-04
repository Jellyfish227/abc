// /**CFile****************************************************************

//   FileName    [ FxchCuda.c ]

//   PackageName [ Fast eXtract with GPU accelerated Cube Hashing (FXCHCUDA) ]

//   Synopsis    [ The entrance into the GPU accelerated fast extract module. ]

//   Author      [ Yu Ching Hei, Chan Eugene ]

//   Affiliation [ CUHK ]

//   Date        [ Ver. 1.0. Started - October 22, 2025. ]

//   Revision    []

// ***********************************************************************/
// #include "FxchCuda.h"

// ABC_NAMESPACE_IMPL_START

// ////////////////////////////////////////////////////////////////////////
// ///                     FUNCTION DEFINITIONS                         ///
// ////////////////////////////////////////////////////////////////////////

// /**Function*************************************************************

//   Synopsis    [ Performs fast extract with gpu accelerated cube hashing on a set
//                 of covers. ]

//   Description []

//   SideEffects []

//   SeeAlso     []

// ***********************************************************************/
// int FxchCuda_FastExtract( Vec_Wec_t* vCubes,
//                       int ObjIdMax,
//                       int nMaxDivExt,
//                       int fVerbose,
//                       int fVeryVerbose )
// {
//     abctime TempTime;
//     Fxch_Man_t* pFxchMan = Fxch_ManAlloc( vCubes );
//     int i;

//     TempTime = Abc_Clock();
//     Fxch_CubesGruping( pFxchMan );
//     Fxch_ManMapLiteralsIntoCubes( pFxchMan, ObjIdMax );
//     Fxch_ManGenerateLitHashKeys( pFxchMan );
//     Fxch_ManComputeLevel( pFxchMan );
//     FxchCuda_ManSCHashTablesInit( pFxchMan ); // made a func name change here
//     Fxch_ManDivCreate( pFxchMan );
//     pFxchMan->timeInit = Abc_Clock() - TempTime;

//     if ( fVeryVerbose )
//         Fxch_ManPrintDivs( pFxchMan );

//     if ( fVerbose )
//         Fxch_ManPrintStats( pFxchMan );

//     TempTime = Abc_Clock();
    
//     for ( i = 0; (!nMaxDivExt || i < nMaxDivExt) && Vec_QueTopPriority( pFxchMan->vDivPrio ) > 0.0; i++ )
//     {
//         int iDiv = Vec_QuePop( pFxchMan->vDivPrio );

//         if ( fVeryVerbose )
//             Fxch_DivPrint( pFxchMan, iDiv );

//         Fxch_ManUpdate( pFxchMan, iDiv );
//     }
   
//     pFxchMan->timeExt = Abc_Clock() - TempTime;

//     if ( fVerbose )
//     {
//         Fxch_ManPrintStats( pFxchMan );
//         Abc_PrintTime( 1, "\n[FXCHCUDA] Elapsed Time", pFxchMan->timeInit + pFxchMan->timeExt );
//         Abc_PrintTime( 1, "[FXCHCUDA]    +-> Init", pFxchMan->timeInit );
//         Abc_PrintTime( 1, "[FXCHCUDA]    +-> Extr", pFxchMan->timeExt );
//     }

//     Fxch_CubesUnGruping( pFxchMan );
//     Fxch_ManCudaSCHashTablesFree( pFxchMan ); // made a func name change here
//     Fxch_ManFree( pFxchMan );

//     Vec_WecRemoveEmpty( vCubes );
//     Vec_WecSortByFirstInt( vCubes, 0 );

//     return 1;
// }

// /**Function*************************************************************

//   Synopsis    [ Retrives the necessary information for the fast extract
//                 with cube hashing. ]

//   Description []

//   SideEffects []

//   SeeAlso     []

// ***********************************************************************/
// int Abc_NtkFxchCudaPerform( Abc_Ntk_t* pNtk,
//                         int nMaxDivExt,
//                         int fVerbose,
//                         int fVeryVerbose ) // made a func name change here
// {
//     Vec_Wec_t* vCubes;

//     assert( Abc_NtkIsSopLogic( pNtk ) );

//     if ( !Abc_NtkFxCheck( pNtk ) )
//     {
//         printf( "Abc_NtkFxchPerform(): Nodes have duplicated fanins. FXCHCUDA is not performed.\n" );
//         return 0;
//     }

//     vCubes = Abc_NtkFxRetrieve( pNtk );
//     if ( FxchCuda_FastExtract( vCubes, Abc_NtkObjNumMax( pNtk ), nMaxDivExt, fVerbose, fVeryVerbose ) > 0 ) // made a func name change here
//     {
//         Abc_NtkFxInsert( pNtk, vCubes );
//         Vec_WecFree( vCubes );

//         if ( !Abc_NtkCheck( pNtk ) )
//             printf( "Abc_NtkFxchPerform(): The network check has failed.\n" );

//         return 1;
//     }
//     else
//         printf( "Warning: The network has not been changed by \"fxchcuda\".\n" );

//     Vec_WecFree( vCubes );

//     return 0;
// }
// ////////////////////////////////////////////////////////////////////////
// ///                       END OF FILE                                ///
// ////////////////////////////////////////////////////////////////////////

// ABC_NAMESPACE_IMPL_END
