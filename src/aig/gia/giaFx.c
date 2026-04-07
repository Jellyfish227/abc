/**CFile****************************************************************

  FileName    [giaFx.c]

  SystemName  [ABC: Logic synthesis and verification system.]

  PackageName [Scalable AIG package.]

  Synopsis    [Interface to fast_extract package.]

  Author      [Alan Mishchenko]
  
  Affiliation [UC Berkeley]

  Date        [Ver. 1.0. Started - June 20, 2005.]

  Revision    [$Id: giaFx.c,v 1.00 2013/09/29 00:00:00 alanmi Exp $]

***********************************************************************/

#include "gia.h"
#include "bool/kit/kit.h"
#include "misc/vec/vecWec.h"
#include "bool/dec/dec.h"
#include "opt/dau/dau.h"
#include "misc/util/utilTruth.h"
#include <omp.h>

ABC_NAMESPACE_IMPL_START


////////////////////////////////////////////////////////////////////////
///                        DECLARATIONS                              ///
////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////
///                     FUNCTION DEFINITIONS                         ///
////////////////////////////////////////////////////////////////////////

/**Function*************************************************************

  Synopsis    [Create GIA for SOP.]

  Description []
               
  SideEffects []

  SeeAlso     []

***********************************************************************/
int Gia_ManGraphToAig( Gia_Man_t * p, Dec_Graph_t * pGraph )
{
    Dec_Node_t * pNode = NULL; // Suppress "might be used uninitialized"
    int i, iAnd0, iAnd1;
    // check for constant function
    if ( Dec_GraphIsConst(pGraph) )
        return Abc_LitNotCond( 1, Dec_GraphIsComplement(pGraph) );
    // check for a literal
    if ( Dec_GraphIsVar(pGraph) )
        return Abc_LitNotCond( Dec_GraphVar(pGraph)->iFunc, Dec_GraphIsComplement(pGraph) );
    // build the AIG nodes corresponding to the AND gates of the graph
    Dec_GraphForEachNode( pGraph, pNode, i )
    {
        iAnd0 = Abc_LitNotCond( Dec_GraphNode(pGraph, pNode->eEdge0.Node)->iFunc, pNode->eEdge0.fCompl ); 
        iAnd1 = Abc_LitNotCond( Dec_GraphNode(pGraph, pNode->eEdge1.Node)->iFunc, pNode->eEdge1.fCompl ); 
        pNode->iFunc = Gia_ManHashAnd( p, iAnd0, iAnd1 );
    }
    // complement the result if necessary
    return Abc_LitNotCond( pNode->iFunc, Dec_GraphIsComplement(pGraph) );
}
int Gia_ManSopToAig( Gia_Man_t * p, char * pSop, Vec_Int_t * vLeaves )
{
    int i, iAnd, iSum, Value, nFanins;
    char * pCube;
    // get the number of variables
    nFanins = Kit_PlaGetVarNum(pSop);
    // go through the cubes of the node's SOP
    iSum = 0;
    Kit_PlaForEachCube( pSop, nFanins, pCube )
    {
        // create the AND of literals
        iAnd = 1;
        Kit_PlaCubeForEachVar( pCube, Value, i )
        {
            assert( Vec_IntEntry(vLeaves, i) >= 0 );
            if ( Value == '1' )
                iAnd = Gia_ManHashAnd( p, iAnd, Vec_IntEntry(vLeaves, i) );
            else if ( Value == '0' )
                iAnd = Gia_ManHashAnd( p, iAnd, Abc_LitNot(Vec_IntEntry(vLeaves, i)) );
            else assert( Value == '-' );
        }
        // add to the sum of cubes
        iSum = Gia_ManHashOr( p, iSum, iAnd );
    }
    // decide whether to complement the result
    if ( Kit_PlaIsComplement(pSop) )
        iSum = Abc_LitNot(iSum);
    return iSum;
}
int Gia_ManFactorGraph( Gia_Man_t * p, Dec_Graph_t * pFForm, Vec_Int_t * vLeaves )
{
    Dec_Node_t * pFFNode;
    int i, Lit;
    // assign fanins
    Dec_GraphForEachLeaf( pFForm, pFFNode, i )
    {
        assert( Vec_IntEntry(vLeaves, i) >= 0 );
        pFFNode->iFunc = Vec_IntEntry(vLeaves, i);
    }
    // perform strashing
    Lit = Gia_ManGraphToAig( p, pFForm );
    return Lit;
}
int Gia_ManFactorNode( Gia_Man_t * p, char * pSop, Vec_Int_t * vLeaves )
{
    if ( Kit_PlaGetVarNum(pSop) == 0 )
        return Abc_LitNotCond( 1, Kit_PlaIsConst0(pSop) );
    assert( Kit_PlaGetVarNum(pSop) == Vec_IntSize(vLeaves) );
    if ( Kit_PlaGetVarNum(pSop) > 2 && Kit_PlaGetCubeNum(pSop) > 1 )
    {
        Dec_Graph_t * pFForm = Dec_Factor( pSop );
        int Lit = Gia_ManFactorGraph( p, pFForm, vLeaves );
        Dec_GraphFree( pFForm );
        return Lit;
    }
    return Gia_ManSopToAig( p, pSop, vLeaves );
}

/**Function*************************************************************

  Synopsis    [Computing truth tables for the mapped network.]

  Description []
               
  SideEffects []

  SeeAlso     []

***********************************************************************/
Vec_Wrd_t * Gia_ManComputeTruths( Gia_Man_t * p, int nCutSize, int nLutNum, int fReverse )
{
    Vec_Wrd_t * vTruths;
    Vec_Int_t vLeaves;
    word * pTruth;
    int i, k, nWords;
    nWords = Abc_Truth6WordNum( nCutSize );
    vTruths = Vec_WrdAlloc( nWords * nLutNum );
    Gia_ObjComputeTruthTableStart( p, nCutSize );
    Gia_ManForEachLut( p, i )
    {
        // collect and sort fanins
        vLeaves.nCap = vLeaves.nSize = Gia_ObjLutSize( p, i );
        vLeaves.pArray = Gia_ObjLutFanins( p, i );
        if( !Vec_IntCheckUniqueSmall(&vLeaves) ) 
        {
            Vec_IntUniqify(&vLeaves);
            Vec_IntWriteEntry(p->vMapping, Vec_IntEntry(p->vMapping, i), vLeaves.nSize);
            for ( k = 0; k < vLeaves.nSize; k++ )
                Vec_IntWriteEntry(p->vMapping, Vec_IntEntry(p->vMapping, i) + 1 + k, vLeaves.pArray[k]);
        }
        assert( Vec_IntCheckUniqueSmall(&vLeaves) );
        Vec_IntSelectSort( Vec_IntArray(&vLeaves), Vec_IntSize(&vLeaves) );
        if ( !fReverse )
            Vec_IntReverseOrder( &vLeaves );
        // compute truth table
        pTruth = Gia_ObjComputeTruthTableCut( p, Gia_ManObj(p, i), &vLeaves );
        for ( k = 0; k < nWords; k++ )
            Vec_WrdPush( vTruths, pTruth[k] );
//        Kit_DsdPrintFromTruth( (unsigned *)pTruth, 6 ); printf( "\n" );
    }
    Gia_ObjComputeTruthTableStop( p );
    assert( Vec_WrdCap(vTruths) == 16 || Vec_WrdSize(vTruths) == Vec_WrdCap(vTruths) );
    return vTruths;
}

/**Function*************************************************************

  Synopsis    [Extracts information about the network.]

  Description []
               
  SideEffects []

  SeeAlso     []

***********************************************************************/
int Gia_ManAssignNumbers( Gia_Man_t * p )
{
    Gia_Obj_t * pObj;
    int i, Counter = 0;
    Gia_ManFillValue( p );
    Gia_ManForEachCi( p, pObj, i )
        pObj->Value = Counter++;
    Gia_ManForEachLut( p, i )
        Gia_ManObj(p, i)->Value = Counter++;
    return Counter;
}
Vec_Wec_t * Gia_ManFxRetrieve( Gia_Man_t * p, Vec_Str_t ** pvCompl, int fReverse )
{
    Vec_Wec_t * vCubes;
    Vec_Wrd_t * vTruths;
    Vec_Int_t * vCube, * vCover;
    int nItems, nCutSize, nWords;
    int i, c, v, Lit, Cube, Counter = 0;
//    abctime clk = Abc_Clock();
    nItems = Gia_ManAssignNumbers( p );
    // compute truth tables
    nCutSize = Gia_ManLutSizeMax( p );
    nWords = Abc_Truth6WordNum( nCutSize );
    vTruths = Gia_ManComputeTruths( p, nCutSize, nItems - Gia_ManCiNum(p), fReverse );
    vCover = Vec_IntAlloc( 1 << 16 );
    // collect cubes
    vCubes = Vec_WecAlloc( 1000 );
    *pvCompl = Vec_StrStart( nItems );
    Gia_ManForEachLut( p, i )
    {
        Gia_Obj_t * pObj = Gia_ManObj( p, i );
        int nVars = Gia_ObjLutSize( p, i );
        int * pVars = Gia_ObjLutFanins( p, i );
        word * pTruth = Vec_WrdEntryP( vTruths, Counter++ * nWords );
        Abc_TtFlipVar5( pTruth, nVars );
        int Status = Kit_TruthIsop( (unsigned *)pTruth, nVars, vCover, 1 );
        Abc_TtFlipVar5( pTruth, nVars );
        if ( Vec_IntSize(vCover) == 0 || (Vec_IntSize(vCover) == 1 && Vec_IntEntry(vCover,0) == 0) )
        {
            Vec_StrWriteEntry( *pvCompl, pObj->Value, (char)(Vec_IntSize(vCover) == 0) );
            vCube = Vec_WecPushLevel( vCubes );
            Vec_IntPush( vCube, pObj->Value );
            continue;
        }
        Vec_StrWriteEntry( *pvCompl, pObj->Value, (char)Status );
        Vec_IntForEachEntry( vCover, Cube, c )
        {
            vCube = Vec_WecPushLevel( vCubes );
            Vec_IntPush( vCube, pObj->Value );
            for ( v = 0; v < nVars; v++ )
            {
                Lit = 3 & (Cube >> (v << 1));
                if ( Lit == 1 )
                    Vec_IntPush( vCube, Abc_Var2Lit(Gia_ManObj(p, pVars[v])->Value, 1) );
                else if ( Lit == 2 )
                    Vec_IntPush( vCube, Abc_Var2Lit(Gia_ManObj(p, pVars[v])->Value, 0) );
                else if ( Lit != 0 )
                    assert( 0 );
            }
            Vec_IntSelectSort( Vec_IntArray(vCube) + 1, Vec_IntSize(vCube) - 1 );
        }        
    }
    assert( Counter * nWords == Vec_WrdSize(vTruths) );
    Vec_WrdFree( vTruths );
    Vec_IntFree( vCover );
//    Abc_PrintTime( 1, "Setup time", Abc_Clock() - clk );
    return vCubes;
}

/**Function*************************************************************

  Synopsis    [Generates GIA after factoring the resulting SOPs.]

  Description []
               
  SideEffects []

  SeeAlso     []

***********************************************************************/
void Gia_ManFxTopoOrder_rec( Vec_Wec_t * vCubes, Vec_Int_t * vFirst, Vec_Int_t * vCount, Vec_Int_t * vVisit, Vec_Int_t * vOrder, int iObj )
{
    int c, v, Lit;
    int iFirst = Vec_IntEntry( vFirst, iObj );
    int nCubes = Vec_IntEntry( vCount, iObj ); 
    assert( !Vec_IntEntry( vVisit, iObj ) );
    Vec_IntWriteEntry( vVisit, iObj, 1 );
    for ( c = 0; c < nCubes; c++ )
    {
        Vec_Int_t * vCube = Vec_WecEntry( vCubes, iFirst + c );
        assert( Vec_IntEntry(vCube, 0) == iObj );
        Vec_IntForEachEntryStart( vCube, Lit, v, 1 )
            if ( !Vec_IntEntry( vVisit, Abc_Lit2Var(Lit) ) )
                Gia_ManFxTopoOrder_rec( vCubes, vFirst, vCount, vVisit, vOrder, Abc_Lit2Var(Lit) );
    }
    Vec_IntPush( vOrder, iObj );
}
Vec_Int_t * Gia_ManFxTopoOrder( Vec_Wec_t * vCubes, int nInputs, int nStart, Vec_Int_t ** pvFirst, Vec_Int_t ** pvCount )
{
    Vec_Int_t * vOrder, * vFirst, * vCount, * vVisit, * vCube;
    int i, iFanin, nNodeMax = -1;
    // find the largest index
    Vec_WecForEachLevel( vCubes, vCube, i )
        nNodeMax = Abc_MaxInt( nNodeMax, Vec_IntEntry(vCube, 0) );
    nNodeMax++;
    // quit if there is no new nodes
    if ( nNodeMax == nStart )
    {
        //printf( "The network is unchanged by fast extract.\n" );
        return NULL;
    }
    // find first cube and how many cubes
    vFirst = Vec_IntStart( nNodeMax );
    vCount = Vec_IntStart( nNodeMax );
    Vec_WecForEachLevel( vCubes, vCube, i )
    {
        iFanin = Vec_IntEntry( vCube, 0 );
        assert( iFanin >= nInputs );
        if ( Vec_IntEntry(vCount, iFanin) == 0 )
            Vec_IntWriteEntry( vFirst, iFanin, i );
        Vec_IntAddToEntry( vCount, iFanin, 1 );
    }
    // put all of them in a topo order
    vOrder = Vec_IntStart( nInputs );
    vVisit = Vec_IntStart( nNodeMax );
    for ( i = 0; i < nInputs; i++ )
        Vec_IntWriteEntry( vVisit, i, 1 );
    for ( i = nInputs; i < nNodeMax; i++ )
        if ( !Vec_IntEntry( vVisit, i ) )
            Gia_ManFxTopoOrder_rec( vCubes, vFirst, vCount, vVisit, vOrder, i );
    assert( Vec_IntSize(vOrder) == nNodeMax );
    Vec_IntFree( vVisit );
    // return topological order of new nodes
    *pvFirst = vFirst;
    *pvCount = vCount;
    return vOrder;
}
Gia_Man_t * Gia_ManFxInsert( Gia_Man_t * p, Vec_Wec_t * vCubes, Vec_Str_t * vCompls )
{
    Gia_Man_t * pNew, * pTemp;
    Gia_Obj_t * pObj; Vec_Str_t * vSop;
    Vec_Int_t * vOrder, * vFirst, * vCount, * vFanins, * vCover;
    Vec_Int_t * vCopies, * vCube, * vMap;
    int k, c, v, Lit, Var, iItem;
//    abctime clk = Abc_Clock();
    // prepare the cubes
    vOrder = Gia_ManFxTopoOrder( vCubes, Gia_ManCiNum(p), Vec_StrSize(vCompls), &vFirst, &vCount );
    if ( vOrder == NULL )
        return Gia_ManDup( p );
    /* |vOrder| == nNodeMax; after parallel FX, vCompl is grown to one char per ObjId 0..max, so sizes match */
    assert( Vec_IntSize(vOrder) >= Vec_StrSize(vCompls) );
    // create new manager
    pNew = Gia_ManStart( Gia_ManObjNum(p) );
    pNew->pName = Abc_UtilStrsav( p->pName );
    pNew->pSpec = Abc_UtilStrsav( p->pSpec );
    pNew->vLevels = Vec_IntStart( 6*Gia_ManObjNum(p)/5 + 100 );
    Gia_ManHashStart( pNew );
    // create primary inputs
    vMap = Vec_IntStartFull( Vec_IntSize(vOrder) );
    vCopies = Vec_IntAlloc( Vec_IntSize(vOrder) );
    Gia_ManForEachCi( p, pObj, k )
        Vec_IntPush( vCopies, Gia_ManAppendCi(pNew) );
    Vec_IntFillExtra( vCopies, Vec_IntSize(vOrder), -1 );
    // add AIG nodes in the topological order
    vSop = Vec_StrAlloc( 1000 );
    vCover = Vec_IntAlloc( 1 << 16 );
    vFanins = Vec_IntAlloc( 100 );
    Vec_IntForEachEntryStart( vOrder, iItem, k, Gia_ManCiNum(p) )
    {
        int iFirst = Vec_IntEntry( vFirst, iItem );
        int nCubes = Vec_IntEntry( vCount, iItem );
        // collect fanins
        Vec_IntClear( vFanins );
        for ( c = 0; c < nCubes; c++ )
        {
            vCube = Vec_WecEntry( vCubes, iFirst + c );
            Vec_IntForEachEntryStart( vCube, Lit, v, 1 )
                if ( Vec_IntEntry(vMap, Abc_Lit2Var(Lit)) == -1 )
                {
                    Vec_IntWriteEntry( vMap, Abc_Lit2Var(Lit), Vec_IntSize(vFanins) );
                    Vec_IntPush( vFanins, Abc_Lit2Var(Lit) );
                }
        }
        if ( Vec_IntSize(vFanins) > 6 )
        {
            // create SOP
            Vec_StrClear( vSop );
            for ( c = 0; c < nCubes; c++ )
            {
                for ( v = 0; v < Vec_IntSize(vFanins); v++ )
                    Vec_StrPush( vSop, '-' );
                vCube = Vec_WecEntry( vCubes, iFirst + c );
                Vec_IntForEachEntryStart( vCube, Lit, v, 1 )
                {
                    Lit = Abc_Lit2LitV( Vec_IntArray(vMap), Lit );
                    assert( Lit >= 0 && Abc_Lit2Var(Lit) < Vec_IntSize(vFanins) );
                    Vec_StrWriteEntry( vSop, Vec_StrSize(vSop) - Vec_IntSize(vFanins) + Abc_Lit2Var(Lit), (char)(Abc_LitIsCompl(Lit)? '0' : '1') );
                }
                Vec_StrPush( vSop, ' ' );
                Vec_StrPush( vSop, '1' );
                Vec_StrPush( vSop, '\n' );
            }
            Vec_StrPush( vSop, '\0' );
            // collect fanins
            Vec_IntForEachEntry( vFanins, Var, v )
            {
                Vec_IntWriteEntry( vMap, Var, -1 );
                Vec_IntWriteEntry( vFanins, v, Vec_IntEntry(vCopies, Var) );
            }
            // derive new AIG
            Lit = Gia_ManFactorNode( pNew, Vec_StrArray(vSop), vFanins );
        }
        else
        {
            word uTruth = 0, uCube;
            for ( c = 0; c < nCubes; c++ )
            {
                uCube = ~(word)0;
                vCube = Vec_WecEntry( vCubes, iFirst + c );
                Vec_IntForEachEntryStart( vCube, Lit, v, 1 )
                {
                    Lit = Abc_Lit2LitV( Vec_IntArray(vMap), Lit );
                    assert( Lit >= 0 && Abc_Lit2Var(Lit) < Vec_IntSize(vFanins) );
                    uCube &= Abc_LitIsCompl(Lit) ? ~s_Truths6[Abc_Lit2Var(Lit)] : s_Truths6[Abc_Lit2Var(Lit)];
                }
                uTruth |= uCube;
            }
            // complement constant
            if ( uTruth == 0 )
                uTruth = ~uTruth;
            // collect fanins
            Vec_IntForEachEntry( vFanins, Var, v )
            {
                Vec_IntWriteEntry( vMap, Var, -1 );
                Vec_IntWriteEntry( vFanins, v, Vec_IntEntry(vCopies, Var) );
            }
            // create truth table
            Lit = Dsm_ManTruthToGia( pNew, &uTruth, vFanins, vCover );
        }
        // complement if the original SOP was complemented
        Lit = Abc_LitNotCond( Lit, (iItem < Vec_StrSize(vCompls)) && (Vec_StrEntry(vCompls, iItem) > 0) );
        // remeber this literal
        assert( Vec_IntEntry(vCopies, iItem) == -1 );
        Vec_IntWriteEntry( vCopies, iItem, Lit );
    }
    Gia_ManHashStop( pNew );
    // create primary outputs
    Gia_ManForEachCo( p, pObj, k )
    {
        Lit = Gia_ObjFaninId0p(p, pObj) ? Vec_IntEntry(vCopies, Gia_ObjFanin0(pObj)->Value) : 0;
        Gia_ManAppendCo( pNew, Abc_LitNotCond( Lit, Gia_ObjFaninC0(pObj) ) );
    }
    Gia_ManSetRegNum( pNew, Gia_ManRegNum(p) );
    // cleanup
    Vec_IntFree( vOrder );
    Vec_IntFree( vFirst );
    Vec_IntFree( vCount );
    Vec_IntFree( vFanins );
    Vec_IntFree( vCopies );
    Vec_IntFree( vMap );
    Vec_StrFree( vSop );
    Vec_IntFree( vCover );
    // remove dangling nodes
    pNew = Gia_ManCleanup( pTemp = pNew );
    Gia_ManStop( pTemp );
//    Abc_PrintTime( 1, "Setdn time", Abc_Clock() - clk );
    return pNew;
}

/**Function*************************************************************

  Synopsis    [Performs classical fast_extract on logic functions.]

  Description []
               
  SideEffects [Sorts the fanins of each cut in the increasing order.]

  SeeAlso     []

***********************************************************************/
Gia_Man_t * Gia_ManPerformFx( Gia_Man_t * p, int nNewNodesMax, int LitCountMax, int fReverse, int fVerbose, int fVeryVerbose )
{
    extern int Fx_FastExtract( Vec_Wec_t * vCubes, int ObjIdMax, int nNewNodesMax, int LitCountMax, int fCanonDivs, int fVerbose, int fVeryVerbose );
    extern Vec_Wec_t ** Fx_PartitionCubes( Vec_Wec_t * vCubes, int * pnParts, int * pLocalObjIdMax );
    extern void Fx_RemapNewNodes( Vec_Wec_t * vPart, int localObjIdMax, int globalOffset );
    extern Vec_Wec_t * Fx_MergeCubes( Vec_Wec_t ** vParts, int nParts, int * pLocalObjIdMax );
    Gia_Man_t * pNew = NULL;
    Vec_Wec_t * vCubes, * vMerged;
    Vec_Wec_t ** vParts;
    Vec_Str_t * vCompl;
    Vec_Int_t * vCube;
    int * pLocalObjIdMax, * pNewNodeCounts, * pOffsets;
    int nParts, globalObjIdMax, j, i, anyChanged;

    if ( Gia_ManAndNum(p) == 0 )
    {
        pNew = Gia_ManDup(p);
        Gia_ManTransferTiming( pNew, p );
        return pNew;
    }
    assert( Gia_ManHasMapping(p) );

    abctime tTotal = Abc_Clock();
    abctime tPhase;

    /* collect information about the covers */
    tPhase = Abc_Clock();
    vCubes = Gia_ManFxRetrieve( p, &vCompl, fReverse );
    // DEBUG Print
    // printf( "[GIA-FX] Retrieve: %.2f sec  (%d cubes)\n",
    //         (float)(Abc_Clock() - tPhase) / CLOCKS_PER_SEC, Vec_WecSize(vCubes) );

    /* determine number of partitions from OMP_NUM_THREADS */
    nParts = omp_get_max_threads();
    if ( nParts < 1 ) nParts = 1;
    // Debug Print
    // printf( "[GIA-FX] Starting parallel FX with %d partition(s)\n", nParts );

    /* global ObjId ceiling before extraction */
    globalObjIdMax = Vec_StrSize(vCompl) - 1;

    /* --- Phase 1: partition vCubes into deep-copied chunks --- */
    tPhase = Abc_Clock();
    pLocalObjIdMax = ABC_ALLOC( int, nParts );
    vParts = Fx_PartitionCubes( vCubes, &nParts, pLocalObjIdMax );
    if ( vParts == NULL || nParts == 0 )
    {
        ABC_FREE( pLocalObjIdMax );
        Vec_WecFree( vCubes );
        Vec_StrFree( vCompl );
        printf( "Warning: The network has not been changed by \"fx\".\n" );
        pNew = Gia_ManDup( p );
        Gia_ManTransferTiming( pNew, p );
        return pNew;
    }
    printf( "[GIA-FX] Phase 1 (partition): %.2f sec  (%d partitions)\n",
            (float)(Abc_Clock() - tPhase) / CLOCKS_PER_SEC, nParts );

    /* free original vCubes — work entirely on per-partition copies */
    Vec_WecFree( vCubes );
    vCubes = NULL;

    /* --- Phase 2: run Fx_FastExtract on each partition in parallel --- */
    {
        abctime tPhase2Start = Abc_Clock();
        #pragma omp parallel for schedule(dynamic, 1) num_threads(nParts)
        for ( j = 0; j < nParts; j++ )
        {
            int tid = omp_get_thread_num();
            int nthreads = omp_get_num_threads();
            abctime tPartStart = Abc_Clock();
            // DEBUG Print
            // printf( "[OMP] thread %d/%d starting partition %d (cubes=%d)\n",
                    // tid, nthreads, j, Vec_WecSize(vParts[j]) );
            fflush( stdout );
            Fx_FastExtract( vParts[j], pLocalObjIdMax[j] + 1,
                            nNewNodesMax, LitCountMax, 0, fVerbose, fVeryVerbose );
            // DEBUG Print
            // printf( "[OMP] thread %d/%d finished partition %d in %.2f sec\n",
                    // tid, nthreads, j,
                    // (float)(Abc_Clock() - tPartStart) / CLOCKS_PER_SEC );
            fflush( stdout );
        }
        printf( "[GIA-FX] Phase 2 (parallel extraction): %.2f sec\n",
                (float)(Abc_Clock() - tPhase2Start) / CLOCKS_PER_SEC );
    }

    /* --- Phase 3: count new nodes per partition, then prefix-sum ---
       Each new divisor is one ObjId but may have 1 or 2 SOP cubes; count cubes
       would overcount and oversize vCompl, breaking Gia_ManFxInsert's assert.
       New ObjIds in a partition are contiguous (localObjIdMax+1 .. max), so
       #new nodes = max(cube[0]) - localObjIdMax (0 if nothing above local). */
    tPhase = Abc_Clock();
    pNewNodeCounts = ABC_CALLOC( int, nParts );
    for ( j = 0; j < nParts; j++ )
    {
        int maxId = pLocalObjIdMax[j];
        Vec_WecForEachLevel( vParts[j], vCube, i )
        {
            if ( Vec_IntSize(vCube) == 0 )
                continue;
            maxId = Abc_MaxInt( maxId, Vec_IntEntry(vCube, 0) );
        }
        pNewNodeCounts[j] = maxId - pLocalObjIdMax[j];
    }

    /* prefix-sum: offsets[j] = first global new-node ID for partition j */
    pOffsets = ABC_ALLOC( int, nParts );
    pOffsets[0] = globalObjIdMax + 1;
    for ( j = 1; j < nParts; j++ )
        pOffsets[j] = pOffsets[j-1] + pNewNodeCounts[j-1];
    printf( "[GIA-FX] Phase 3 (count+prefix-sum): %.2f sec\n",
            (float)(Abc_Clock() - tPhase) / CLOCKS_PER_SEC );

    /* --- Phase 4: remap new-node IDs to global range (parallel) --- */
    tPhase = Abc_Clock();
    #pragma omp parallel for schedule(static) num_threads(nParts)
    for ( j = 0; j < nParts; j++ )
        Fx_RemapNewNodes( vParts[j], pLocalObjIdMax[j], pOffsets[j] );
    printf( "[GIA-FX] Phase 4 (remap): %.2f sec\n",
            (float)(Abc_Clock() - tPhase) / CLOCKS_PER_SEC );

    /* --- Phase 5: merge into a single sorted vCubes, then insert --- */
    tPhase = Abc_Clock();
    anyChanged = 0;
    for ( j = 0; j < nParts; j++ )
        if ( pNewNodeCounts[j] > 0 )
            { anyChanged = 1; break; }

    if ( anyChanged )
    {
        vMerged = Fx_MergeCubes( vParts, nParts, pLocalObjIdMax );
        /* grow vCompl to cover new node IDs so Gia_ManFxInsert can index it */
        {
            int nNewTotal = 0;
            for ( j = 0; j < nParts; j++ )
                nNewTotal += pNewNodeCounts[j];
            Vec_StrFillExtra( vCompl, Vec_StrSize(vCompl) + nNewTotal, 0 );
        }
        pNew = Gia_ManFxInsert( p, vMerged, vCompl );
        Vec_WecFree( vMerged );
    }
    else
    {
        printf( "Warning: The network has not been changed by \"fx\".\n" );
        pNew = Gia_ManDup( p );
    }
    printf( "[GIA-FX] Phase 5 (merge+insert): %.2f sec\n",
            (float)(Abc_Clock() - tPhase) / CLOCKS_PER_SEC );
    printf( "[GIA-FX] Total wall time: %.2f sec\n",
            (float)(Abc_Clock() - tTotal) / CLOCKS_PER_SEC );

    Gia_ManTransferTiming( pNew, p );

    /* cleanup */
    for ( j = 0; j < nParts; j++ )
        Vec_WecFree( vParts[j] );
    ABC_FREE( vParts );
    ABC_FREE( pLocalObjIdMax );
    ABC_FREE( pNewNodeCounts );
    ABC_FREE( pOffsets );
    Vec_StrFree( vCompl );

    return pNew;
}

////////////////////////////////////////////////////////////////////////
///                       END OF FILE                                ///
////////////////////////////////////////////////////////////////////////


ABC_NAMESPACE_IMPL_END

