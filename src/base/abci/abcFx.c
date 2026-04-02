/**CFile****************************************************************

  FileName    [abcFx.c]

  SystemName  [ABC: Logic synthesis and verification system.]

  PackageName [Network and node package.]

  Synopsis    [Implementation of traditional "fast_extract" algorithm.]

  Author      [Alan Mishchenko, Bruno Schmitt]
  
  Affiliation [UC Berkeley, UFRGS]

  Date        [Ver. 1.0. Started - April 26, 2013.]

  Revision    [$Id: abcFx.c,v 1.00 2013/04/26 00:00:00 alanmi Exp $]

***********************************************************************/

#include "base/abc/abc.h"
#include "misc/vec/vecWec.h"
#include "misc/vec/vecQue.h"
#include "misc/vec/vecHsh.h"
#include "opt/fxch/Fxch.h"
#ifdef ABC_USE_OMP
#include <omp.h>
#endif

ABC_NAMESPACE_IMPL_START

////////////////////////////////////////////////////////////////////////
///                        DECLARATIONS                              ///
////////////////////////////////////////////////////////////////////////

/*
    The code in this file implements the traditional "fast_extract" algorithm, 
    which extracts two-cube divisors concurrently with single-cube two-literal divisors,
    as proposed in the TCAD'92 paper by J. Rajski and J. Vasudevamurthi.

    Integration notes:

    It is assumed that each object (primary input or internal node) in the original network 
    is associated with a unique integer number, called object identifier (ObjId, for short).

    The user's input data given to 'fast_extract" is an array of cubes (pMan->vCubes).
    Each cube is an array of integers, in which the first entry contains ObjId of the node, 
    to which this cube belongs in the original network. The following entries of a cube are 
    SOP literals of this cube.  Each literal is represtned as 2*FaninId + ComplAttr, where FaninId 
    is ObjId of the fanin node and ComplAttr is 1 if literal is complemented, and 0 otherwise.

    The user's output data produced by 'fast_extract' is also an array of cubes (pMan->vCubes).
    If no divisors have been extracted, the output array is the same as the input array.
    If some divisors have been extracted, the output array contains updated old cubes and new cubes 
    representing the extracted divisors. The new divisors have their ObjId starting from the 
    largest ObjId used in the cubes. To give the user more flexibility, which may be needed when some 
    ObjIds are already used for primary output nodes, which do not participate in fast_extract,
    the parameter ObjIdMax is passed to procedure Fx_FastExtract().  The new divisors will receive
    their ObjId starting from ObjIdMax onward, as divisor extaction proceeds.

    The following two requirements are imposed on the input and output array of cubes:
    (1) The array of cubes should be sorted by the first entry in each cube (that is, cubes belonging 
    to the same node should form a contiguous range). 
    (2) Literals in a cube should be sorted in the increasing order of the integer numbers.
    
    To integrate this code into a calling application, such as ABC, the input cube array should 
    be generated (below this is done by the procedure Abc_NtkFxRetrieve) and the output cube array
    should be incorporated into the current network (below this is done by the procedure Abc_NtkFxInsert).
    In essence, the latter procedure performs the following:
    - removes the current fanins and SOPs of each node in the network
    - adds new nodes for each new divisor introduced by "fast_extract"
    - populates fanins and SOPs of each node, both old and new, as indicaded by the resulting cube array.

    Implementation notes:

    The implementation is optimized for simplicity and speed of computation.
    (1) Main input/output data-structure (pMan->vCubes) is the array of cubes which is dynamically updated by the algorithm.
    (2) Auxiliary data-structure (pMan->vLits) is the array of arrays. The i-th array contains IDs of cubes which have literal i.
    It may be convenient to think about the first (second) array as rows (columns) of a sparse matrix, 
    although the sparse matrix data-structure is not used in the proposed implementation.
    (3) Hash table (pMan->pHash) hashes the normalized divisors (represented as integer arrays) into integer numbers.
    (4) Array of divisor weights (pMan->vWeights), that is, the number of SOP literals to be saved by extacting each divisor.
    (5) Priority queue (pMan->vPrio), which sorts divisor (integer numbers) by their weight
    (6) Integer array (pMan->vVarCube), which maps each ObjId into the first cube of this object, 
    or -1, if there is no cubes as in the case of a primary input.

*/

typedef struct Fx_Man_t_ Fx_Man_t;
struct Fx_Man_t_
{
    // user's data
    Vec_Wec_t *     vCubes;     // cube -> lit
    int             LitCountMax;// max size of divisor to extract
    int             fCanonDivs; // use only AND/XOR/MUX
    // internal data
    Vec_Wec_t *     vLits;      // lit -> cube
    Vec_Int_t *     vCounts;    // literal counts (currently not used)
    Hsh_VecMan_t *  pHash;      // hash table for normalized divisors
    Vec_Flt_t *     vWeights;   // divisor weights
    Vec_Que_t *     vPrio;      // priority queue for divisors by weight
    Vec_Int_t *     vVarCube;   // mapping ObjId into its first cube
    Vec_Int_t *     vLevels;    // variable levels
    // temporary data to update the data-structure when a divisor is extracted
    Vec_Int_t *     vCubesS;    // single cubes for the given divisor
    Vec_Int_t *     vCubesD;    // cube pairs for the given divisor
    Vec_Int_t *     vCompls;    // complemented attribute of each cube pair
    Vec_Int_t *     vCubeFree;  // cube-free divisor
    Vec_Int_t *     vDiv;       // selected divisor
    Vec_Int_t *     vSCC;       // single cubes containment cubes
    // statistics 
    abctime         timeStart;  // starting time
    int             nVars;      // original problem variables
    int             nLits;      // the number of SOP literals
    int             nDivs;      // the number of extracted divisors
    int             nCompls;    // the number of complements
    int             nPairsS;    // number of lit pairs
    int             nPairsD;    // number of cube pairs
    int             nDivsS;     // single cube divisors
    int             nDivMux[3]; // 0 = mux, 1 = compl mux, 2 = no mux
};

static inline int Fx_ManGetFirstVarCube( Fx_Man_t * p, Vec_Int_t * vCube ) { return Vec_IntEntry( p->vVarCube, Vec_IntEntry(vCube, 0) ); }

#ifdef ABC_USE_OMP
// Per-thread context for parallel batch extraction
typedef struct Fx_ThreadCtx_t_ Fx_ThreadCtx_t;
struct Fx_ThreadCtx_t_
{
    Vec_Int_t *     vCubeFree;    // thread-local scratch for divisor normalization
    Vec_Int_t *     vDeltaDiv;    // divisor IDs with weight changes
    Vec_Flt_t *     vDeltaWt;     // corresponding weight deltas
    Vec_Int_t *     vSCC;         // thread-local SCC accumulator
    int             nPairsS;      // thread-local SCD pair counter delta
    int             nPairsD;      // thread-local DCD pair counter delta
    int             fWarning;
};

// Per-candidate data for batch extraction
typedef struct Fx_Candidate_t_ Fx_Candidate_t;
struct Fx_Candidate_t_
{
    int             iDiv;
    int             Lit0, Lit1;
    Vec_Int_t *     vCubesS;      // affected SCD cubes (owned)
    Vec_Int_t *     vCubesD;      // affected DCD cube pairs (owned)
    Vec_Int_t *     vCompls;      // complement flags (owned)
    Vec_Int_t *     vDiv;         // divisor content (owned)
    int             iVarNew;      // pre-allocated variable ID
    int             iNewCubeBase; // pre-allocated cube slot base index
    int             fWarning;
};
#endif

#define Fx_ManForEachCubeVec( vVec, vCubes, vCube, i )           \
    for ( i = 0; (i < Vec_IntSize(vVec)) && ((vCube) = Vec_WecEntry(vCubes, Vec_IntEntry(vVec, i))); i++ )

////////////////////////////////////////////////////////////////////////
///                     FUNCTION DEFINITIONS                         ///
////////////////////////////////////////////////////////////////////////

/**Function*************************************************************

  Synopsis    [Retrieves SOP information for fast_extract.]

  Description []
               
  SideEffects []

  SeeAlso     []

***********************************************************************/
Vec_Wec_t * Abc_NtkFxRetrieve( Abc_Ntk_t * pNtk )
{
    Vec_Wec_t * vCubes;
    Vec_Int_t * vCube;
    Abc_Obj_t * pNode;
    char * pCube, * pSop;
    int nVars, i, v, Lit;
    assert( Abc_NtkIsSopLogic(pNtk) );
    vCubes = Vec_WecAlloc( 1000 );
    Abc_NtkForEachNode( pNtk, pNode, i )
    {
        pSop = (char *)pNode->pData;
        nVars = Abc_SopGetVarNum(pSop);
        assert( nVars == Abc_ObjFaninNum(pNode) );
//        if ( nVars < 2 ) continue;
        Abc_SopForEachCube( pSop, nVars, pCube )
        {
            vCube = Vec_WecPushLevel( vCubes );
            Vec_IntPush( vCube, Abc_ObjId(pNode) );
            Abc_CubeForEachVar( pCube, Lit, v )
            {
                if ( Lit == '0' )
                    Vec_IntPush( vCube, Abc_Var2Lit(Abc_ObjFaninId(pNode, v), 1) );
                else if ( Lit == '1' )
                    Vec_IntPush( vCube, Abc_Var2Lit(Abc_ObjFaninId(pNode, v), 0) );
            }
            Vec_IntSelectSort( Vec_IntArray(vCube) + 1, Vec_IntSize(vCube) - 1 );
        }
    }
    return vCubes;
}

/**Function*************************************************************

  Synopsis    [Inserts SOP information after fast_extract.]

  Description []
               
  SideEffects []

  SeeAlso     []

***********************************************************************/
void Abc_NtkFxInsert( Abc_Ntk_t * pNtk, Vec_Wec_t * vCubes )
{
    Vec_Int_t * vCube, * vPres, * vFirst, * vCount;
    Abc_Obj_t * pNode, * pFanin;
    char * pCube, * pSop;
    int i, k, v, Lit, iFanin, iNodeMax = 0;
    assert( Abc_NtkIsSopLogic(pNtk) );
    // check that cubes have no gaps and are ordered by first node
    Lit = -1;
    Vec_WecForEachLevel( vCubes, vCube, i )
    {
        assert( Vec_IntSize(vCube) > 0 );
        assert( Lit <= Vec_IntEntry(vCube, 0) );
        Lit = Vec_IntEntry(vCube, 0);
    }
    // find the largest index
    Vec_WecForEachLevel( vCubes, vCube, i )
        iNodeMax = Abc_MaxInt( iNodeMax, Vec_IntEntry(vCube, 0) );
    // quit if nothing changes
    if ( iNodeMax < Abc_NtkObjNumMax(pNtk) )
    {
        //printf( "The network is unchanged by fast extract.\n" );
        return;
    }
    // create new nodes
    for ( i = Abc_NtkObjNumMax(pNtk); i <= iNodeMax; i++ )
    {
        pNode = Abc_NtkCreateNode( pNtk );
        assert( i == (int)Abc_ObjId(pNode) );
    }
    // create node fanins
    vFirst = Vec_IntStart( Abc_NtkObjNumMax(pNtk) );
    vCount = Vec_IntStart( Abc_NtkObjNumMax(pNtk) );
    Vec_WecForEachLevel( vCubes, vCube, i )
    {
        iFanin = Vec_IntEntry( vCube, 0 );
        if ( Vec_IntEntry(vCount, iFanin) == 0 )
            Vec_IntWriteEntry( vFirst, iFanin, i );
        Vec_IntAddToEntry( vCount, iFanin, 1 );
    }
    // create node SOPs
    vPres = Vec_IntStartFull( Abc_NtkObjNumMax(pNtk) );
    Abc_NtkForEachNode( pNtk, pNode, i )
    {
//        if ( Vec_IntEntry(vCount, i) == 0 ) continue;
        Abc_ObjRemoveFanins( pNode );
        // create fanins
        assert( Vec_IntEntry(vCount, i) > 0 );
        for ( k = 0; k < Vec_IntEntry(vCount, i); k++ )
        {
            vCube = Vec_WecEntry( vCubes, Vec_IntEntry(vFirst, i) + k );
            assert( Vec_IntEntry( vCube, 0 ) == i );
            Vec_IntForEachEntryStart( vCube, Lit, v, 1 )
            {
                pFanin = Abc_NtkObj(pNtk, Abc_Lit2Var(Lit));
                if ( Vec_IntEntry(vPres, Abc_ObjId(pFanin)) >= 0 )
                    continue;
                Vec_IntWriteEntry(vPres, Abc_ObjId(pFanin), Abc_ObjFaninNum(pNode));
                Abc_ObjAddFanin( pNode, pFanin );
            }
        }
        // create SOP
        pSop = pCube = Abc_SopStart( (Mem_Flex_t *)pNtk->pManFunc, Vec_IntEntry(vCount, i), Abc_ObjFaninNum(pNode) );
        for ( k = 0; k < Vec_IntEntry(vCount, i); k++ )
        {
            vCube = Vec_WecEntry( vCubes, Vec_IntEntry(vFirst, i) + k );
            assert( Vec_IntEntry( vCube, 0 ) == i );
            Vec_IntForEachEntryStart( vCube, Lit, v, 1 )
            {
                pFanin = Abc_NtkObj(pNtk, Abc_Lit2Var(Lit));
                iFanin = Vec_IntEntry(vPres, Abc_ObjId(pFanin));
                assert( iFanin >= 0 && iFanin < Abc_ObjFaninNum(pNode) );
                pCube[iFanin] = Abc_LitIsCompl(Lit) ? '0' : '1';
            }
            pCube += Abc_ObjFaninNum(pNode) + 3;
        }
        // complement SOP if the original one was complemented
        if ( pNode->pData && Abc_SopIsComplement((char *)pNode->pData) )
            Abc_SopComplement( pSop );
        pNode->pData = pSop;
        // clean fanins
        Abc_ObjForEachFanin( pNode, pFanin, v )
            Vec_IntWriteEntry( vPres, Abc_ObjId(pFanin), -1 );
    }
    Vec_IntFree( vFirst );
    Vec_IntFree( vCount );
    Vec_IntFree( vPres );
}

/**Function*************************************************************

  Synopsis    [Makes sure the nodes do not have complemented and duplicated fanins.]

  Description []
               
  SideEffects []

  SeeAlso     []

***********************************************************************/
int Abc_NtkFxCheck( Abc_Ntk_t * pNtk )
{
    Abc_Obj_t * pNode;
    int i;
//    Abc_NtkForEachObj( pNtk, pNode, i )
//        Abc_ObjPrint( stdout, pNode );
    Abc_NtkForEachNode( pNtk, pNode, i )
        if ( !Vec_IntCheckUniqueSmall( &pNode->vFanins ) ) {
            printf( "Fanins of node %d: ", i );
            Vec_IntPrint( &pNode->vFanins );
            return 0;
        }
    return 1;
}

/**Function*************************************************************

  Synopsis    []

  Description []
               
  SideEffects []

  SeeAlso     []

***********************************************************************/
int Abc_NtkFxPerform( Abc_Ntk_t * pNtk, int nNewNodesMax, int LitCountMax, int fCanonDivs, int fVerbose, int fVeryVerbose )
{
    extern int Fx_FastExtract( Vec_Wec_t * vCubes, int ObjIdMax, int nNewNodesMax, int LitCountMax, int fCanonDivs, int fVerbose, int fVeryVerbose );
    Vec_Wec_t * vCubes;
    assert( Abc_NtkIsSopLogic(pNtk) );
    // check unique fanins
    if ( !Abc_NtkFxCheck(pNtk) )
    {
        printf( "Abc_NtkFastExtract: Nodes have duplicated fanins. FX is not performed.\n" );
        return 0;
    }
    // collect information about the covers
    vCubes = Abc_NtkFxRetrieve( pNtk );
    // call the fast extract procedure
    if ( Fx_FastExtract( vCubes, Abc_NtkObjNumMax(pNtk), nNewNodesMax, LitCountMax, fCanonDivs, fVerbose, fVeryVerbose ) > 0 )
    {
        // update the network
        Abc_NtkFxInsert( pNtk, vCubes );
        Vec_WecFree( vCubes );
        if ( !Abc_NtkCheck( pNtk ) )
            printf( "Abc_NtkFxPerform: The network check has failed.\n" );
        return 1;
    }
    else
        printf( "Warning: The network has not been changed by \"fx\".\n" );
    Vec_WecFree( vCubes );
    return 0;
}



/**Function*************************************************************

  Synopsis    [Starting the manager.]

  Description []
               
  SideEffects []

  SeeAlso     []

***********************************************************************/
Fx_Man_t * Fx_ManStart( Vec_Wec_t * vCubes )
{
    Fx_Man_t * p;
    p = ABC_CALLOC( Fx_Man_t, 1 );
    p->vCubes   = vCubes;
    // temporary data
    p->vCubesS   = Vec_IntAlloc( 100 );
    p->vCubesD   = Vec_IntAlloc( 100 );
    p->vCompls   = Vec_IntAlloc( 100 );
    p->vCubeFree = Vec_IntAlloc( 100 );
    p->vDiv      = Vec_IntAlloc( 100 );
    p->vSCC      = Vec_IntAlloc( 100 );
    return p;
}
void Fx_ManStop( Fx_Man_t * p )
{
//    Vec_WecFree( p->vCubes );
    Vec_WecFree( p->vLits );
    Vec_IntFree( p->vCounts );
    Hsh_VecManStop( p->pHash );
    Vec_FltFree( p->vWeights );
    Vec_QueFree( p->vPrio );
    Vec_IntFree( p->vVarCube );
    Vec_IntFree( p->vLevels );
    // temporary data
    Vec_IntFree( p->vCubesS );
    Vec_IntFree( p->vCubesD );
    Vec_IntFree( p->vCompls );
    Vec_IntFree( p->vCubeFree );
    Vec_IntFree( p->vDiv );
    Vec_IntFree( p->vSCC );
    ABC_FREE( p );
}

/**Function*************************************************************

  Synopsis    [Compute levels of the nodes.]

  Description []
               
  SideEffects []

  SeeAlso     []

***********************************************************************/
static inline int Fx_ManComputeLevelDiv( Fx_Man_t * p, Vec_Int_t * vCubeFree )
{
    int i, Lit, Level = 0;
    Vec_IntForEachEntry( vCubeFree, Lit, i )
        Level = Abc_MaxInt( Level, Vec_IntEntry(p->vLevels, Abc_Lit2Var(Abc_Lit2Var(Lit))) );
    return Abc_MinInt( Level, 800 );
}
static inline int Fx_ManComputeLevelCube( Fx_Man_t * p, Vec_Int_t * vCube )
{
    int k, Lit, Level = 0;
    Vec_IntForEachEntryStart( vCube, Lit, k, 1 )
        Level = Abc_MaxInt( Level, Vec_IntEntry(p->vLevels, Abc_Lit2Var(Lit)) );
    return Level;
}
void Fx_ManComputeLevel( Fx_Man_t * p )
{
    Vec_Int_t * vCube;
    int i, iVar, iFirst = 0;
    iVar = Vec_IntEntry( Vec_WecEntry(p->vCubes,0), 0 );
    p->vLevels = Vec_IntStart( p->nVars );
    Vec_WecForEachLevel( p->vCubes, vCube, i )
    {
        if ( iVar != Vec_IntEntry(vCube, 0) )
        {
            // add the number of cubes
            Vec_IntAddToEntry( p->vLevels, iVar, i - iFirst );
            iVar = Vec_IntEntry(vCube, 0);
            iFirst = i;
        }
        Vec_IntUpdateEntry( p->vLevels, iVar, Fx_ManComputeLevelCube(p, vCube) );
    }
}

/**Function*************************************************************

  Synopsis    [Printing procedures.]

  Description []
               
  SideEffects []

  SeeAlso     []

***********************************************************************/
static inline void Fx_PrintDivArray( Vec_Int_t * vDiv )
{
    int i, Lit;
    Vec_IntForEachEntry( vDiv, Lit, i )
        if ( !Abc_LitIsCompl(Lit) )
            printf( "%d(1)", Abc_Lit2Var(Lit) );
    printf( " + " );
    Vec_IntForEachEntry( vDiv, Lit, i )
        if ( Abc_LitIsCompl(Lit) )
            printf( "%d(2)", Abc_Lit2Var(Lit) );
}
static inline void Fx_PrintDiv( Fx_Man_t * p, int iDiv )
{
    int i;
    printf( "%4d : ", p->nDivs );
    printf( "Div %7d : ", iDiv );
    printf( "Weight %12.5f  ", Vec_FltEntry(p->vWeights, iDiv) );
    Fx_PrintDivArray( Hsh_VecReadEntry(p->pHash, iDiv) );
    for ( i = Vec_IntSize(Hsh_VecReadEntry(p->pHash, iDiv)) + 3; i < 16; i++ )
        printf( " " );
    printf( "Lits =%7d  ", p->nLits );
    printf( "Divs =%8d  ", Hsh_VecSize(p->pHash) );
    Abc_PrintTime( 1, "Time", Abc_Clock() - p->timeStart );
}
static void Fx_PrintDivisors( Fx_Man_t * p )
{
    int iDiv;
    for ( iDiv = 0; iDiv < Vec_FltSize(p->vWeights); iDiv++ )
        Fx_PrintDiv( p, iDiv );
}
static void Fx_PrintStats( Fx_Man_t * p, abctime clk )
{
    printf( "Cubes =%8d  ", Vec_WecSizeUsed(p->vCubes) );
    printf( "Lits  =%8d  ", Vec_WecSizeUsed(p->vLits) );
    printf( "Divs  =%8d  ", Hsh_VecSize(p->pHash) );
    printf( "Divs+ =%8d  ", Vec_QueSize(p->vPrio) );
    printf( "Compl =%8d  ", p->nDivMux[1] );
    printf( "Extr  =%7d  ", p->nDivs );
    printf(" nLits =%d   ", p->nLits);
    Abc_PrintTime( 1, "Time", clk );
}

/**Function*************************************************************

  Synopsis    [Returns 1 if the divisor should be complemented.]

  Description [Normalizes the divisor by putting, first, positive control 
  literal first and, second, positive data1 literal. As the result, 
  a MUX divisor is (ab + !ac) and an XOR divisor is (ab + !a!b).]
               
  SideEffects []

  SeeAlso     []

***********************************************************************/
static int Fx_ManDivNormalize( Vec_Int_t * vCubeFree ) // return 1 if complemented
{
    int * L = Vec_IntArray(vCubeFree);
    int RetValue = 0, LitA0 = -1, LitB0 = -1, LitA1 = -1, LitB1 = -1;
    assert( Vec_IntSize(vCubeFree) == 4 );
    if ( Abc_LitIsCompl(L[0]) != Abc_LitIsCompl(L[1]) && (L[0] >> 2) == (L[1] >> 2) ) // diff cubes, same vars
    {
        if ( Abc_LitIsCompl(L[2]) == Abc_LitIsCompl(L[3]) )
            return -1;
        LitA0 = Abc_Lit2Var(L[0]), LitB0 = Abc_Lit2Var(L[1]);
        if ( Abc_LitIsCompl(L[0]) == Abc_LitIsCompl(L[2]) )
        {
            assert( Abc_LitIsCompl(L[1]) == Abc_LitIsCompl(L[3]) );
            LitA1 = Abc_Lit2Var(L[2]), LitB1 = Abc_Lit2Var(L[3]);
        }
        else
        {
            assert( Abc_LitIsCompl(L[0]) == Abc_LitIsCompl(L[3]) );
            assert( Abc_LitIsCompl(L[1]) == Abc_LitIsCompl(L[2]) );
            LitA1 = Abc_Lit2Var(L[3]), LitB1 = Abc_Lit2Var(L[2]);
        }
    }
    else if ( Abc_LitIsCompl(L[1]) != Abc_LitIsCompl(L[2]) && (L[1] >> 2) == (L[2] >> 2) )
    {
        if ( Abc_LitIsCompl(L[0]) == Abc_LitIsCompl(L[3]) )
            return -1;
        LitA0 = Abc_Lit2Var(L[1]), LitB0 = Abc_Lit2Var(L[2]);
        if ( Abc_LitIsCompl(L[1]) == Abc_LitIsCompl(L[0]) )
            LitA1 = Abc_Lit2Var(L[0]), LitB1 = Abc_Lit2Var(L[3]);
        else
            LitA1 = Abc_Lit2Var(L[3]), LitB1 = Abc_Lit2Var(L[0]);
    }
    else if ( Abc_LitIsCompl(L[2]) != Abc_LitIsCompl(L[3]) && (L[2] >> 2) == (L[3] >> 2) )
    {
        if ( Abc_LitIsCompl(L[0]) == Abc_LitIsCompl(L[1]) )
            return -1;
        LitA0 = Abc_Lit2Var(L[2]), LitB0 = Abc_Lit2Var(L[3]);
        if ( Abc_LitIsCompl(L[2]) == Abc_LitIsCompl(L[0]) )
            LitA1 = Abc_Lit2Var(L[0]), LitB1 = Abc_Lit2Var(L[1]);
        else
            LitA1 = Abc_Lit2Var(L[1]), LitB1 = Abc_Lit2Var(L[0]);
    }
    else 
        return -1;
    assert( LitA0 == Abc_LitNot(LitB0) );
    if ( Abc_LitIsCompl(LitA0) )
    {
        ABC_SWAP( int, LitA0, LitB0 );
        ABC_SWAP( int, LitA1, LitB1 );
    }
    assert( !Abc_LitIsCompl(LitA0) );
    if ( Abc_LitIsCompl(LitA1) )
    {
        LitA1 = Abc_LitNot(LitA1);
        LitB1 = Abc_LitNot(LitB1);
        RetValue = 1;
    }
    assert( !Abc_LitIsCompl(LitA1) );
    // arrange literals in such as a way that
    // - the first two literals are control literals from different cubes
    // - the third literal is non-complented data input
    // - the forth literal is possibly complemented data input
    L[0] = Abc_Var2Lit( LitA0, 0 );
    L[1] = Abc_Var2Lit( LitB0, 1 );
    L[2] = Abc_Var2Lit( LitA1, 0 );
    L[3] = Abc_Var2Lit( LitB1, 1 );
    return RetValue;
}

/**Function*************************************************************

  Synopsis    [Find a cube-free divisor of the two cubes.]

  Description []
               
  SideEffects []

  SeeAlso     []

***********************************************************************/
int Fx_ManDivFindCubeFree( Vec_Int_t * vArr1, Vec_Int_t * vArr2, Vec_Int_t * vCubeFree, int * fWarning )
{
    int * pBeg1 = Vec_IntArray( vArr1 ) + 1;  // skip variable ID
    int * pBeg2 = Vec_IntArray( vArr2 ) + 1;  // skip variable ID
    int * pEnd1 = Vec_IntLimit( vArr1 );
    int * pEnd2 = Vec_IntLimit( vArr2 );
    int Counter = 0, fAttr0 = 0, fAttr1 = 1;
    Vec_IntClear( vCubeFree );
    while ( pBeg1 < pEnd1 && pBeg2 < pEnd2 )
    {
        if ( *pBeg1 == *pBeg2 )
            pBeg1++, pBeg2++, Counter++;
        else if ( *pBeg1 < *pBeg2 )
            Vec_IntPush( vCubeFree, Abc_Var2Lit(*pBeg1++, fAttr0) );
        else  
        {
            if ( Vec_IntSize(vCubeFree) == 0 )
                fAttr0 = 1, fAttr1 = 0;
            Vec_IntPush( vCubeFree, Abc_Var2Lit(*pBeg2++, fAttr1) );
        }
    }
    while ( pBeg1 < pEnd1 )
        Vec_IntPush( vCubeFree, Abc_Var2Lit(*pBeg1++, fAttr0) );
    while ( pBeg2 < pEnd2 )
        Vec_IntPush( vCubeFree, Abc_Var2Lit(*pBeg2++, fAttr1) );
    if ( Vec_IntSize(vCubeFree) == 0 )
        printf( "The SOP has duplicated cubes.\n" );
    else if ( Vec_IntSize(vCubeFree) == 1 )
        return -1;
    else if ( Vec_IntSize( vCubeFree ) == 3 )
    {
        int * pArray = Vec_IntArray( vCubeFree ); 

        if ( Abc_Lit2Var( pArray[0] ) == Abc_LitNot( Abc_Lit2Var( pArray[1] ) ) )
        {
            if ( Abc_LitIsCompl( pArray[0] ) == Abc_LitIsCompl( pArray[2] ) )
                Vec_IntDrop( vCubeFree, 0 );
            else
                Vec_IntDrop( vCubeFree, 1 );
        }
        else if ( Abc_Lit2Var( pArray[1] ) == Abc_LitNot( Abc_Lit2Var( pArray[2] ) ) )
        {
            if ( Abc_LitIsCompl( pArray[1] ) == Abc_LitIsCompl( pArray[0] ) )
                Vec_IntDrop( vCubeFree, 1 );
            else
                Vec_IntDrop( vCubeFree, 2 );
        }

        if ( Vec_IntSize( vCubeFree ) == 2 )
        {
            int Lit0 = Abc_Lit2Var( pArray[0] ),
                Lit1 = Abc_Lit2Var( pArray[1] );

            if ( Lit0 > Lit1 )
                ABC_SWAP( int, Lit0, Lit1 );

            Vec_IntWriteEntry( vCubeFree, 0, Abc_Var2Lit( Lit0, 0 ) );
            Vec_IntWriteEntry( vCubeFree, 1, Abc_Var2Lit( Lit1, 1 ) );
        }
    }
    assert( !Abc_LitIsCompl(Vec_IntEntry(vCubeFree, 0)) );
    return Counter;
}

/**Function*************************************************************

  Synopsis    [Procedures operating on a two-cube divisor.]

  Description []
               
  SideEffects []

  SeeAlso     []

***********************************************************************/
static inline void Fx_ManDivFindPivots( Vec_Int_t * vDiv, int * pLit0, int * pLit1 )
{
    int i, Lit;
    *pLit0 = -1;
    *pLit1 = -1;
    Vec_IntForEachEntry( vDiv, Lit, i )
    {
        if ( Abc_LitIsCompl(Lit) )
        {
            if ( *pLit1 == -1 )
                *pLit1 = Abc_Lit2Var(Lit);
        }
        else
        {
            if ( *pLit0 == -1 )
                *pLit0 = Abc_Lit2Var(Lit);
        }
        if ( *pLit0 >= 0 && *pLit1 >= 0 )
            return;
    }
}
static inline int Fx_ManDivRemoveLits( Vec_Int_t * vCube, Vec_Int_t * vDiv, int fCompl )
{
    int i, Lit, Count = 0;
    assert( !fCompl || Vec_IntSize(vDiv) == 4 );
    Vec_IntForEachEntry( vDiv, Lit, i )
    {
        Count += Vec_IntRemove1( vCube, Abc_Lit2Var(Lit) ^ (fCompl && i > 1) );  // the last two lits can be complemented
        if ( Vec_IntSize( vDiv ) == 2 )
            Count += Vec_IntRemove1( vCube, Abc_LitNot( Abc_Lit2Var(Lit) ) );
    }
    return Count;
}
static inline void Fx_ManDivAddLits( Vec_Int_t * vCube, Vec_Int_t * vCube2, Vec_Int_t * vDiv )
{
    int i, Lit, * pArray;
//    Vec_IntClear( vCube );
//    Vec_IntClear( vCube2 );
    Vec_IntForEachEntry( vDiv, Lit, i )
        if ( Abc_LitIsCompl(Lit) )
            Vec_IntPush( vCube2, Abc_Lit2Var(Lit) );
        else
            Vec_IntPush( vCube, Abc_Lit2Var(Lit) );
    if ( Vec_IntSize(vDiv) == 4 && Vec_IntSize(vCube) == 3 )
    {
        assert( Vec_IntSize(vCube2) == 3 );
        pArray = Vec_IntArray(vCube);
        if ( pArray[1] > pArray[2] )
            ABC_SWAP( int, pArray[1], pArray[2] );
        pArray = Vec_IntArray(vCube2);
        if ( pArray[1] > pArray[2] )
            ABC_SWAP( int, pArray[1], pArray[2] );
    }
}

/**Function*************************************************************

  Synopsis    [Setting up the data-structure.]

  Description []
               
  SideEffects []

  SeeAlso     []

***********************************************************************/
void Fx_ManCreateLiterals( Fx_Man_t * p, int nVars )
{
    Vec_Int_t * vCube;
    int i, k, Lit, Count;
    // find the number of variables
    p->nVars = p->nLits = 0;
    Vec_WecForEachLevel( p->vCubes, vCube, i )
    {
        assert( Vec_IntSize(vCube) > 0 );
        p->nVars = Abc_MaxInt( p->nVars, Vec_IntEntry(vCube, 0) );
        p->nLits += Vec_IntSize(vCube) - 1;
        Vec_IntForEachEntryStart( vCube, Lit, k, 1 )
            p->nVars = Abc_MaxInt( p->nVars, Abc_Lit2Var(Lit) );
    }
//    p->nVars++;
    assert( p->nVars < nVars );
    p->nVars = nVars;
    // count literals
    p->vCounts = Vec_IntStart( 2*p->nVars );
    Vec_WecForEachLevel( p->vCubes, vCube, i )
        Vec_IntForEachEntryStart( vCube, Lit, k, 1 )
            Vec_IntAddToEntry( p->vCounts, Lit, 1 );
    // start literals
    p->vLits = Vec_WecStart( 2*p->nVars );
    Vec_IntForEachEntry( p->vCounts, Count, Lit )
        Vec_IntGrow( Vec_WecEntry(p->vLits, Lit), Count );
    // fill out literals
    Vec_WecForEachLevel( p->vCubes, vCube, i )
        Vec_IntForEachEntryStart( vCube, Lit, k, 1 )
            Vec_WecPush( p->vLits, Lit, i );
    // create mapping of variable into the first cube
    p->vVarCube = Vec_IntStartFull( p->nVars );
    Vec_WecForEachLevel( p->vCubes, vCube, i )
        if ( Vec_IntEntry(p->vVarCube, Vec_IntEntry(vCube, 0)) == -1 )
            Vec_IntWriteEntry( p->vVarCube, Vec_IntEntry(vCube, 0), i );
}
int Fx_ManCubeSingleCubeDivisors( Fx_Man_t * p, Vec_Int_t * vPivot, int fRemove, int fUpdate )
{
    int k, n, Lit, Lit2, iDiv;
    if ( Vec_IntSize(vPivot) < 2 )
        return 0;
    Vec_IntForEachEntryStart( vPivot, Lit, k, 1 )
    Vec_IntForEachEntryStart( vPivot, Lit2, n, k+1 )
    {
        assert( Lit < Lit2 );
        Vec_IntClear( p->vCubeFree );
        Vec_IntPush( p->vCubeFree, Abc_Var2Lit(Abc_LitNot(Lit), 0) );
        Vec_IntPush( p->vCubeFree, Abc_Var2Lit(Abc_LitNot(Lit2), 1) );
        iDiv = Hsh_VecManAdd( p->pHash, p->vCubeFree );
        if ( !fRemove )
        {
            if ( Vec_FltSize(p->vWeights) == iDiv )
            {
                Vec_FltPush(p->vWeights, -2 + 0.9 - 0.001 * Fx_ManComputeLevelDiv(p, p->vCubeFree));
                p->nDivsS++;
            }
            assert( iDiv < Vec_FltSize(p->vWeights) );
            Vec_FltAddToEntry( p->vWeights, iDiv, 1 );
            p->nPairsS++;
        }
        else
        {
            assert( iDiv < Vec_FltSize(p->vWeights) );
            Vec_FltAddToEntry( p->vWeights, iDiv, -1 );
            p->nPairsS--;
        }
        if ( fUpdate )
        {
            if ( Vec_QueIsMember(p->vPrio, iDiv) )
                Vec_QueUpdate( p->vPrio, iDiv );
            else if ( !fRemove )
                Vec_QuePush( p->vPrio, iDiv );
        }
    }
    return Vec_IntSize(vPivot) * (Vec_IntSize(vPivot) - 1) / 2;
}
void Fx_ManCubeDoubleCubeDivisors( Fx_Man_t * p, int iFirst, Vec_Int_t * vPivot, int fRemove, int fUpdate, int * fWarning )
{
    Vec_Int_t * vCube;
    int i, iDiv, Base;
    Vec_WecForEachLevelStart( p->vCubes, vCube, i, iFirst )
    {
        if ( Vec_IntSize(vCube) == 0 || vCube == vPivot )
            continue;
        if ( Vec_WecIntHasMark(vCube) && Vec_WecIntHasMark(vPivot) && vCube > vPivot )
            continue;
        if ( Vec_IntEntry(vCube, 0) != Vec_IntEntry(vPivot, 0) )
            break;
        Base = Fx_ManDivFindCubeFree( vCube, vPivot, p->vCubeFree, fWarning );
        if ( Base == -1 ) 
        {
            if ( fRemove == 0 )
            {
                if ( Vec_IntSize( vCube ) > Vec_IntSize( vPivot ) )
                    Vec_IntPush( p->vSCC, Vec_WecLevelId( p->vCubes, vCube ) );
                else
                    Vec_IntPush( p->vSCC, Vec_WecLevelId( p->vCubes, vPivot ) );
            }
            continue;
        }
        if ( Vec_IntSize(p->vCubeFree) == 4 )
        { 
            int Value = Fx_ManDivNormalize( p->vCubeFree );
            if ( Value == 0 )
                p->nDivMux[0]++;
            else if ( Value == 1 )
                p->nDivMux[1]++;
            else
                p->nDivMux[2]++;
            if ( p->fCanonDivs && Value < 0 )
                continue;
        }
        if ( p->LitCountMax && p->LitCountMax < Vec_IntSize(p->vCubeFree) )
            continue;
        if ( p->fCanonDivs && Vec_IntSize(p->vCubeFree) == 3 )
            continue;
        iDiv = Hsh_VecManAdd( p->pHash, p->vCubeFree );
        if ( !fRemove )
        {
            if ( iDiv == Vec_FltSize(p->vWeights) )
                Vec_FltPush(p->vWeights, -Vec_IntSize(p->vCubeFree) + 0.9 - 0.0009 * Fx_ManComputeLevelDiv(p, p->vCubeFree));
            assert( iDiv < Vec_FltSize(p->vWeights) );
            Vec_FltAddToEntry( p->vWeights, iDiv, Base + Vec_IntSize(p->vCubeFree) - 1 );
            p->nPairsD++;
        }
        else
        {
            assert( iDiv < Vec_FltSize(p->vWeights) );
            Vec_FltAddToEntry( p->vWeights, iDiv, -(Base + Vec_IntSize(p->vCubeFree) - 1) );
            p->nPairsD--;
        }
        if ( fUpdate )
        {
            if ( Vec_QueIsMember(p->vPrio, iDiv) )
                Vec_QueUpdate( p->vPrio, iDiv );
            else if ( !fRemove )
                Vec_QuePush( p->vPrio, iDiv );
        }
    } 
}
void Fx_ManCreateDivisors( Fx_Man_t * p )
{
    Vec_Int_t * vCube;
    float Weight;
    int i, fWarning = 0;
    // alloc hash table
    assert( p->pHash == NULL );
    p->pHash = Hsh_VecManStart( 1000 );
    p->vWeights = Vec_FltAlloc( 1000 );
    // create single-cube two-literal divisors
    Vec_WecForEachLevel( p->vCubes, vCube, i )
        Fx_ManCubeSingleCubeDivisors( p, vCube, 0, 0 ); // add - no update
    assert( p->nDivsS == Vec_FltSize(p->vWeights) );
    // create two-cube divisors
    Vec_WecForEachLevel( p->vCubes, vCube, i )
        Fx_ManCubeDoubleCubeDivisors( p, i+1, vCube, 0, 0, &fWarning ); // add - no update
    // create queue with all divisors
    p->vPrio = Vec_QueAlloc( Vec_FltSize(p->vWeights) );
    Vec_QueSetPriority( p->vPrio, Vec_FltArrayP(p->vWeights) );
    Vec_FltForEachEntry( p->vWeights, Weight, i )
        if ( Weight > 0.0 )
            Vec_QuePush( p->vPrio, i );
}


/**Function*************************************************************

  Synopsis    [Compress the cubes by removing unused ones.]

  Description []
               
  SideEffects []

  SeeAlso     []

***********************************************************************/
static inline void Fx_ManCompressCubes( Vec_Wec_t * vCubes, Vec_Int_t * vLit2Cube )
{
    int i, CubeId, k = 0;
    Vec_IntForEachEntry( vLit2Cube, CubeId, i )
        if ( Vec_IntSize(Vec_WecEntry(vCubes, CubeId)) > 0 )
            Vec_IntWriteEntry( vLit2Cube, k++, CubeId );
    Vec_IntShrink( vLit2Cube, k );
}


/**Function*************************************************************

  Synopsis    [Find command cube pairs.]

  Description []
               
  SideEffects []

  SeeAlso     []

***********************************************************************/
static inline int Fx_ManGetCubeVar( Vec_Wec_t * vCubes, int iCube )  { return Vec_IntEntry( Vec_WecEntry(vCubes, iCube), 0 );      }
void Fx_ManFindCommonPairs( Vec_Wec_t * vCubes, Vec_Int_t * vPart0, Vec_Int_t * vPart1, Vec_Int_t * vPairs, Vec_Int_t * vCompls, Vec_Int_t * vDiv, Vec_Int_t * vCubeFree, int * fWarning )
{
    int * pBeg1 = vPart0->pArray;
    int * pBeg2 = vPart1->pArray;
    int * pEnd1 = vPart0->pArray + vPart0->nSize;
    int * pEnd2 = vPart1->pArray + vPart1->nSize;
    int i, k, i_, k_, fCompl, CubeId1, CubeId2;
    Vec_IntClear( vPairs );
    Vec_IntClear( vCompls );
    while ( pBeg1 < pEnd1 && pBeg2 < pEnd2 )
    {
        CubeId1 = Fx_ManGetCubeVar(vCubes, *pBeg1);
        CubeId2 = Fx_ManGetCubeVar(vCubes, *pBeg2);
        if ( CubeId1 == CubeId2 )
        {
            for ( i = 1; pBeg1+i < pEnd1; i++ )
                if ( CubeId1 != Fx_ManGetCubeVar(vCubes, pBeg1[i]) )
                    break;
            for ( k = 1; pBeg2+k < pEnd2; k++ )
                if ( CubeId1 != Fx_ManGetCubeVar(vCubes, pBeg2[k]) )
                    break;
            for ( i_ = 0; i_ < i; i_++ )
            for ( k_ = 0; k_ < k; k_++ )
            {
                if ( pBeg1[i_] == pBeg2[k_] )
                    continue;
                Fx_ManDivFindCubeFree( Vec_WecEntry(vCubes, pBeg1[i_]), Vec_WecEntry(vCubes, pBeg2[k_]), vCubeFree, fWarning );
                fCompl = (Vec_IntSize(vCubeFree) == 4 && Fx_ManDivNormalize(vCubeFree) == 1);
                if ( !Vec_IntEqual( vDiv, vCubeFree ) )
                    continue;
                Vec_IntPush( vPairs, pBeg1[i_] );
                Vec_IntPush( vPairs, pBeg2[k_] );
                Vec_IntPush( vCompls, fCompl );
            }
            pBeg1 += i;
            pBeg2 += k;
        }
        else if ( CubeId1 < CubeId2 )
            pBeg1++;
        else 
            pBeg2++;
    }
}

/**Function*************************************************************

  Synopsis    [Updates the data-structure when one divisor is selected.]

  Description []
               
  SideEffects []

  SeeAlso     []

***********************************************************************/
void Fx_ManUpdate( Fx_Man_t * p, int iDiv, int * fWarning )
{
    Vec_Int_t * vCube, * vCube2, * vLitP = NULL, * vLitN = NULL;
    Vec_Int_t * vDiv = p->vDiv;
    int i, k, Lit0, Lit1, iVarNew = 0, RetValue, Level;
    float Diff = Vec_FltEntry(p->vWeights, iDiv) - (float)((int)Vec_FltEntry(p->vWeights, iDiv));
    assert( Diff > 0.0 && Diff < 1.0 );

    // get the divisor and select pivot variables
    p->nDivs++;
    Vec_IntClear( vDiv );
    Vec_IntAppend( vDiv, Hsh_VecReadEntry(p->pHash, iDiv) );
    Fx_ManDivFindPivots( vDiv, &Lit0, &Lit1 );
    assert( Lit0 >= 0 && Lit1 >= 0 );


    // collect single-cube-divisor cubes
    Vec_IntClear( p->vCubesS );
    if ( Vec_IntSize(vDiv) == 2 )
    {
        Fx_ManCompressCubes( p->vCubes, Vec_WecEntry(p->vLits, Abc_LitNot(Lit0)) );
        Fx_ManCompressCubes( p->vCubes, Vec_WecEntry(p->vLits, Abc_LitNot(Lit1)) );
        Vec_IntTwoRemoveCommon( Vec_WecEntry(p->vLits, Abc_LitNot(Lit0)), Vec_WecEntry(p->vLits, Abc_LitNot(Lit1)), p->vCubesS );
    }

    // collect double-cube-divisor cube pairs
    Fx_ManCompressCubes( p->vCubes, Vec_WecEntry(p->vLits, Lit0) );
    Fx_ManCompressCubes( p->vCubes, Vec_WecEntry(p->vLits, Lit1) );
    Fx_ManFindCommonPairs( p->vCubes, Vec_WecEntry(p->vLits, Lit0), Vec_WecEntry(p->vLits, Lit1), p->vCubesD, p->vCompls, vDiv, p->vCubeFree, fWarning );

    // subtract cost of single-cube divisors
    Fx_ManForEachCubeVec( p->vCubesS, p->vCubes, vCube, i )
        Fx_ManCubeSingleCubeDivisors( p, vCube, 1, 1 );  // remove - update
    Fx_ManForEachCubeVec( p->vCubesD, p->vCubes, vCube, i )
        Fx_ManCubeSingleCubeDivisors( p, vCube, 1, 1 );  // remove - update

    // mark the cubes to be removed
    Vec_WecMarkLevels( p->vCubes, p->vCubesS );
    Vec_WecMarkLevels( p->vCubes, p->vCubesD );

    // subtract cost of double-cube divisors
    Fx_ManForEachCubeVec( p->vCubesS, p->vCubes, vCube, i )
        Fx_ManCubeDoubleCubeDivisors( p, Fx_ManGetFirstVarCube(p, vCube), vCube, 1, 1, fWarning );  // remove - update
    Fx_ManForEachCubeVec( p->vCubesD, p->vCubes, vCube, i )
        Fx_ManCubeDoubleCubeDivisors( p, Fx_ManGetFirstVarCube(p, vCube), vCube, 1, 1, fWarning );  // remove - update

    // unmark the cubes to be removed
    Vec_WecUnmarkLevels( p->vCubes, p->vCubesS );
    Vec_WecUnmarkLevels( p->vCubes, p->vCubesD );

    if ( Abc_Lit2Var(Lit0) == Abc_Lit2Var(Lit1) && Vec_IntSize(Hsh_VecReadEntry(p->pHash, iDiv)) == 2 )
        goto ExtractFromPairs;

    // create new divisor
    iVarNew = Vec_WecSize( p->vLits ) / 2;
    assert( Vec_IntSize(p->vVarCube) == iVarNew );
    Vec_IntPush( p->vVarCube, Vec_WecSize(p->vCubes) );
    vCube = Vec_WecPushLevel( p->vCubes );
    Vec_IntPush( vCube, iVarNew );
    if ( Vec_IntSize(vDiv) == 2 )
    {
        Vec_IntPush( vCube, Abc_LitNot(Lit0) );
        Vec_IntPush( vCube, Abc_LitNot(Lit1) );
        Level = 1 + Fx_ManComputeLevelCube( p, vCube );
    }
    else
    {
        vCube2 = Vec_WecPushLevel( p->vCubes );
        vCube = Vec_WecEntry( p->vCubes, Vec_WecSize(p->vCubes) - 2 );
        Vec_IntPush( vCube2, iVarNew );
        Fx_ManDivAddLits( vCube, vCube2, vDiv );
        Level = 2 + Abc_MaxInt( Fx_ManComputeLevelCube(p, vCube), Fx_ManComputeLevelCube(p, vCube2) );
    }
    assert( Vec_IntSize(p->vLevels) == iVarNew );
    Vec_IntPush( p->vLevels, Level );
    // do not add new cubes to the matrix 
    p->nLits += Vec_IntSize( vDiv );
    // create new literals
    vLitP = Vec_WecPushLevel( p->vLits );
    vLitN = Vec_WecPushLevel( p->vLits );
    vLitP = Vec_WecEntry( p->vLits, Vec_WecSize(p->vLits) - 2 );
    // create updated single-cube divisor cubes
    Fx_ManForEachCubeVec( p->vCubesS, p->vCubes, vCube, i )
    {
        RetValue  = Vec_IntRemove1( vCube, Abc_LitNot(Lit0) );
        RetValue += Vec_IntRemove1( vCube, Abc_LitNot(Lit1) );
        assert( RetValue == 2 );
        Vec_IntPush( vCube, Abc_Var2Lit(iVarNew, 0) );
        Vec_IntPush( vLitP, Vec_WecLevelId(p->vCubes, vCube) );
        p->nLits--;
    }
    // create updated double-cube divisor cube pairs
ExtractFromPairs:
    k = 0;
    p->nCompls = 0;
    assert( Vec_IntSize(p->vCubesD) % 2 == 0 );
    assert( Vec_IntSize(p->vCubesD) == 2 * Vec_IntSize(p->vCompls) );
    for ( i = 0; i < Vec_IntSize(p->vCubesD); i += 2 )
    {
        int fCompl = Vec_IntEntry(p->vCompls, i/2);
        p->nCompls += fCompl;
        vCube  = Vec_WecEntry( p->vCubes, Vec_IntEntry(p->vCubesD, i) );
        vCube2 = Vec_WecEntry( p->vCubes, Vec_IntEntry(p->vCubesD, i+1) );
        RetValue  = Fx_ManDivRemoveLits( vCube, vDiv, fCompl );  // cube 2*i
        RetValue += Fx_ManDivRemoveLits( vCube2, vDiv, fCompl ); // cube 2*i+1
        assert( RetValue == Vec_IntSize(vDiv) || RetValue == Vec_IntSize( vDiv ) + 1);
        if ( iVarNew > 0 )
        {
            if ( Vec_IntSize(vDiv) == 2 || fCompl )
            {
                Vec_IntPush( vCube, Abc_Var2Lit(iVarNew, 1) );
                Vec_IntPush( vLitN, Vec_WecLevelId(p->vCubes, vCube) ); // MAKE SURE vCube IS SORTED BY ID
            }
            else 
            {
                Vec_IntPush( vCube, Abc_Var2Lit(iVarNew, 0) );
                Vec_IntPush( vLitP, Vec_WecLevelId(p->vCubes, vCube) ); // MAKE SURE vCube IS SORTED BY ID
            }
        }
        p->nLits -= Vec_IntSize(vDiv) + Vec_IntSize(vCube2) - 2;

        // remove second cube
        Vec_IntWriteEntry( p->vCubesD, k++, Vec_WecLevelId(p->vCubes, vCube) );
        Vec_IntClear( vCube2 ); 
    }
    assert( k == Vec_IntSize(p->vCubesD) / 2 );
    Vec_IntShrink( p->vCubesD, k );
    Vec_IntSort( p->vCubesD, 0 );
    //Vec_IntSort( vLitN, 0 );
    //Vec_IntSort( vLitP, 0 );

    // add cost of single-cube divisors
    Fx_ManForEachCubeVec( p->vCubesS, p->vCubes, vCube, i )
        Fx_ManCubeSingleCubeDivisors( p, vCube, 0, 1 );  // add - update
    Fx_ManForEachCubeVec( p->vCubesD, p->vCubes, vCube, i )
        Fx_ManCubeSingleCubeDivisors( p, vCube, 0, 1 );  // add - update

    // mark the cubes to be removed
    Vec_WecMarkLevels( p->vCubes, p->vCubesS );
    Vec_WecMarkLevels( p->vCubes, p->vCubesD );

    // add cost of double-cube divisors
    Fx_ManForEachCubeVec( p->vCubesS, p->vCubes, vCube, i )
        Fx_ManCubeDoubleCubeDivisors( p, Fx_ManGetFirstVarCube(p, vCube), vCube, 0, 1, fWarning );  // add - update
    Fx_ManForEachCubeVec( p->vCubesD, p->vCubes, vCube, i )
        Fx_ManCubeDoubleCubeDivisors( p, Fx_ManGetFirstVarCube(p, vCube), vCube, 0, 1, fWarning );  // add - update

    // unmark the cubes to be removed
    Vec_WecUnmarkLevels( p->vCubes, p->vCubesS );
    Vec_WecUnmarkLevels( p->vCubes, p->vCubesD );

    // Deal with SCC
    if ( Vec_IntSize( p->vSCC ) )
    {
        Vec_IntUniqify( p->vSCC );
        Fx_ManForEachCubeVec( p->vSCC, p->vCubes, vCube, i )
        {
            Fx_ManCubeDoubleCubeDivisors( p, Fx_ManGetFirstVarCube(p, vCube), vCube, 1, 1, fWarning );  // remove - update
            Vec_IntClear( vCube );
        }
        Vec_IntClear( p->vSCC );
    }
    // add cost of the new divisor
    if ( Vec_IntSize(vDiv) > 2 )
    {
        vCube  = Vec_WecEntry( p->vCubes, Vec_WecSize(p->vCubes) - 2 );
        vCube2 = Vec_WecEntry( p->vCubes, Vec_WecSize(p->vCubes) - 1 );
        Fx_ManCubeSingleCubeDivisors( p, vCube,  0, 1 );  // add - update
        Fx_ManCubeSingleCubeDivisors( p, vCube2, 0, 1 );  // add - update
        Vec_IntForEachEntryStart( vCube, Lit0, i, 1 )
            Vec_WecPush( p->vLits, Lit0, Vec_WecLevelId(p->vCubes, vCube) );
        Vec_IntForEachEntryStart( vCube2, Lit0, i, 1 )
            Vec_WecPush( p->vLits, Lit0, Vec_WecLevelId(p->vCubes, vCube2) );
    }

    // remove these cubes from the lit array of the divisor
    Vec_IntForEachEntry( vDiv, Lit0, i )
    {
        Vec_IntTwoRemove( Vec_WecEntry(p->vLits, Abc_Lit2Var(Lit0)), p->vCubesD );
        if ( (p->nCompls && i > 1) || Vec_IntSize( vDiv ) == 2 ) // the last two lits are possibly complemented
            Vec_IntTwoRemove( Vec_WecEntry(p->vLits, Abc_LitNot(Abc_Lit2Var(Lit0))), p->vCubesD );
    }

}

#ifdef ABC_USE_OMP
/**Function*************************************************************

  Synopsis    [Thread context allocation and deallocation.]

***********************************************************************/
static Fx_ThreadCtx_t * Fx_ThreadCtxAlloc( int nThreads )
{
    Fx_ThreadCtx_t * pCtxs = ABC_CALLOC( Fx_ThreadCtx_t, nThreads );
    int t;
    for ( t = 0; t < nThreads; t++ )
    {
        pCtxs[t].vCubeFree = Vec_IntAlloc( 100 );
        pCtxs[t].vDeltaDiv = Vec_IntAlloc( 1000 );
        pCtxs[t].vDeltaWt  = Vec_FltAlloc( 1000 );
        pCtxs[t].vSCC      = Vec_IntAlloc( 100 );
    }
    return pCtxs;
}
static void Fx_ThreadCtxFree( Fx_ThreadCtx_t * pCtxs, int nThreads )
{
    int t;
    for ( t = 0; t < nThreads; t++ )
    {
        Vec_IntFree( pCtxs[t].vCubeFree );
        Vec_IntFree( pCtxs[t].vDeltaDiv );
        Vec_FltFree( pCtxs[t].vDeltaWt );
        Vec_IntFree( pCtxs[t].vSCC );
    }
    ABC_FREE( pCtxs );
}
static void Fx_ThreadCtxClear( Fx_ThreadCtx_t * pCtx )
{
    Vec_IntClear( pCtx->vDeltaDiv );
    Vec_FltClear( pCtx->vDeltaWt );
    Vec_IntClear( pCtx->vSCC );
    pCtx->nPairsS = 0;
    pCtx->nPairsD = 0;
    pCtx->fWarning = 0;
}

/**Function*************************************************************

  Synopsis    [Candidate allocation and deallocation.]

***********************************************************************/
static Fx_Candidate_t * Fx_CandAlloc( int K )
{
    Fx_Candidate_t * pCands = ABC_CALLOC( Fx_Candidate_t, K );
    int c;
    for ( c = 0; c < K; c++ )
    {
        pCands[c].vCubesS = Vec_IntAlloc( 100 );
        pCands[c].vCubesD = Vec_IntAlloc( 100 );
        pCands[c].vCompls = Vec_IntAlloc( 100 );
        pCands[c].vDiv    = Vec_IntAlloc( 100 );
    }
    return pCands;
}
static void Fx_CandFree( Fx_Candidate_t * pCands, int K )
{
    int c;
    for ( c = 0; c < K; c++ )
    {
        Vec_IntFree( pCands[c].vCubesS );
        Vec_IntFree( pCands[c].vCubesD );
        Vec_IntFree( pCands[c].vCompls );
        Vec_IntFree( pCands[c].vDiv );
    }
    ABC_FREE( pCands );
}

/* OMP_PLACEHOLDER_1 */

/**Function*************************************************************

  Synopsis    [Gathers affected cubes for a candidate divisor (sequential).]

***********************************************************************/
static void Fx_ManUpdateGather( Fx_Man_t * p, Fx_Candidate_t * pCand )
{
    Vec_IntClear( pCand->vDiv );
    Vec_IntAppend( pCand->vDiv, Hsh_VecReadEntry(p->pHash, pCand->iDiv) );
    Fx_ManDivFindPivots( pCand->vDiv, &pCand->Lit0, &pCand->Lit1 );
    // collect single-cube-divisor cubes
    Vec_IntClear( pCand->vCubesS );
    if ( Vec_IntSize(pCand->vDiv) == 2 )
    {
        Fx_ManCompressCubes( p->vCubes, Vec_WecEntry(p->vLits, Abc_LitNot(pCand->Lit0)) );
        Fx_ManCompressCubes( p->vCubes, Vec_WecEntry(p->vLits, Abc_LitNot(pCand->Lit1)) );
        Vec_IntTwoRemoveCommon( Vec_WecEntry(p->vLits, Abc_LitNot(pCand->Lit0)), Vec_WecEntry(p->vLits, Abc_LitNot(pCand->Lit1)), pCand->vCubesS );
    }
    // collect double-cube-divisor cube pairs
    Fx_ManCompressCubes( p->vCubes, Vec_WecEntry(p->vLits, pCand->Lit0) );
    Fx_ManCompressCubes( p->vCubes, Vec_WecEntry(p->vLits, pCand->Lit1) );
    Vec_IntClear( pCand->vCubesD );
    Vec_IntClear( pCand->vCompls );
    pCand->fWarning = 0;
    Fx_ManFindCommonPairs( p->vCubes, Vec_WecEntry(p->vLits, pCand->Lit0), Vec_WecEntry(p->vLits, pCand->Lit1),
        pCand->vCubesD, pCand->vCompls, pCand->vDiv, p->vCubeFree, &pCand->fWarning );
}

/**Function*************************************************************

  Synopsis    [Thread-safe SCD cost with delta accumulation.]

***********************************************************************/
static int Fx_ManCubeSCD_Delta( Fx_Man_t * p, Vec_Int_t * vPivot, int fRemove, Fx_ThreadCtx_t * pCtx )
{
    int k, n, Lit, Lit2, iDiv;
    if ( Vec_IntSize(vPivot) < 2 )
        return 0;
    Vec_IntForEachEntryStart( vPivot, Lit, k, 1 )
    Vec_IntForEachEntryStart( vPivot, Lit2, n, k+1 )
    {
        assert( Lit < Lit2 );
        Vec_IntClear( pCtx->vCubeFree );
        Vec_IntPush( pCtx->vCubeFree, Abc_Var2Lit(Abc_LitNot(Lit), 0) );
        Vec_IntPush( pCtx->vCubeFree, Abc_Var2Lit(Abc_LitNot(Lit2), 1) );
        iDiv = Hsh_VecManLookup( p->pHash, pCtx->vCubeFree );
        if ( iDiv < 0 ) continue;
        Vec_IntPush( pCtx->vDeltaDiv, iDiv );
        Vec_FltPush( pCtx->vDeltaWt, fRemove ? -1.0f : 1.0f );
        if ( fRemove ) pCtx->nPairsS--;
        else           pCtx->nPairsS++;
    }
    return Vec_IntSize(vPivot) * (Vec_IntSize(vPivot) - 1) / 2;
}

/**Function*************************************************************

  Synopsis    [Thread-safe DCD cost with delta accumulation.]

***********************************************************************/
static void Fx_ManCubeDCD_Delta( Fx_Man_t * p, int iFirst, Vec_Int_t * vPivot, int fRemove, Fx_ThreadCtx_t * pCtx )
{
    Vec_Int_t * vCube;
    int i, iDiv, Base;
    Vec_WecForEachLevelStart( p->vCubes, vCube, i, iFirst )
    {
        if ( Vec_IntSize(vCube) == 0 || vCube == vPivot )
            continue;
        if ( Vec_WecIntHasMark(vCube) && Vec_WecIntHasMark(vPivot) && vCube > vPivot )
            continue;
        if ( Vec_IntEntry(vCube, 0) != Vec_IntEntry(vPivot, 0) )
            break;
        Base = Fx_ManDivFindCubeFree( vCube, vPivot, pCtx->vCubeFree, &pCtx->fWarning );
        if ( Base == -1 )
        {
            if ( fRemove == 0 )
            {
                if ( Vec_IntSize(vCube) > Vec_IntSize(vPivot) )
                    Vec_IntPush( pCtx->vSCC, Vec_WecLevelId(p->vCubes, vCube) );
                else
                    Vec_IntPush( pCtx->vSCC, Vec_WecLevelId(p->vCubes, vPivot) );
            }
            continue;
        }
        if ( Vec_IntSize(pCtx->vCubeFree) == 4 )
        {
            int Value = Fx_ManDivNormalize( pCtx->vCubeFree );
            if ( p->fCanonDivs && Value < 0 )
                continue;
        }
        if ( p->LitCountMax && p->LitCountMax < Vec_IntSize(pCtx->vCubeFree) )
            continue;
        if ( p->fCanonDivs && Vec_IntSize(pCtx->vCubeFree) == 3 )
            continue;
        iDiv = Hsh_VecManLookup( p->pHash, pCtx->vCubeFree );
        if ( iDiv < 0 ) continue;
        {
            float delta = (float)(Base + Vec_IntSize(pCtx->vCubeFree) - 1);
            Vec_IntPush( pCtx->vDeltaDiv, iDiv );
            Vec_FltPush( pCtx->vDeltaWt, fRemove ? -delta : delta );
        }
        if ( fRemove ) pCtx->nPairsD--;
        else           pCtx->nPairsD++;
    }
}

/* OMP_PLACEHOLDER_2 */

/**Function*************************************************************

  Synopsis    [Detects conflicts among candidates (shared cubes).]

***********************************************************************/
static void Fx_ManDetectConflicts( Fx_Man_t * p, Fx_Candidate_t * pCands, int nCands,
    Vec_Int_t * vIndep, Vec_Int_t * vConflict )
{
    Vec_Int_t * vSeen = Vec_IntStart( Vec_WecSize(p->vCubes) );
    int c, i, CubeId, fConflict;
    Vec_IntClear( vIndep );
    Vec_IntClear( vConflict );
    for ( c = 0; c < nCands; c++ )
    {
        fConflict = 0;
        Vec_IntForEachEntry( pCands[c].vCubesS, CubeId, i )
            if ( Vec_IntEntry(vSeen, CubeId) ) { fConflict = 1; break; }
        if ( !fConflict )
        {
            for ( i = 0; i < Vec_IntSize(pCands[c].vCubesD); i++ )
            {
                CubeId = Vec_IntEntry(pCands[c].vCubesD, i);
                if ( Vec_IntEntry(vSeen, CubeId) ) { fConflict = 1; break; }
            }
        }
        if ( fConflict )
        {
            Vec_IntPush( vConflict, c );
            continue;
        }
        Vec_IntPush( vIndep, c );
        Vec_IntForEachEntry( pCands[c].vCubesS, CubeId, i )
            Vec_IntWriteEntry( vSeen, CubeId, 1 );
        for ( i = 0; i < Vec_IntSize(pCands[c].vCubesD); i++ )
            Vec_IntWriteEntry( vSeen, Vec_IntEntry(pCands[c].vCubesD, i), 1 );
    }
    Vec_IntFree( vSeen );
}

/**Function*************************************************************

  Synopsis    [Subtract cost + structural mutation (runs in parallel).]

***********************************************************************/
static void Fx_ManUpdateSubtractAndMutate( Fx_Man_t * p, Fx_Candidate_t * pCand, Fx_ThreadCtx_t * pCtx )
{
    Vec_Int_t * vCube, * vCube2, * vLitP, * vLitN;
    int i, k, Lit0, Lit1, RetValue, Level;
    int iVarNew = pCand->iVarNew;
    Lit0 = pCand->Lit0;
    Lit1 = pCand->Lit1;

    // subtract cost of single-cube divisors (delta accumulation)
    Fx_ManForEachCubeVec( pCand->vCubesS, p->vCubes, vCube, i )
        Fx_ManCubeSCD_Delta( p, vCube, 1, pCtx );
    Fx_ManForEachCubeVec( pCand->vCubesD, p->vCubes, vCube, i )
        Fx_ManCubeSCD_Delta( p, vCube, 1, pCtx );

    // mark cubes, subtract cost of double-cube divisors
    Vec_WecMarkLevels( p->vCubes, pCand->vCubesS );
    Vec_WecMarkLevels( p->vCubes, pCand->vCubesD );
    Fx_ManForEachCubeVec( pCand->vCubesS, p->vCubes, vCube, i )
        Fx_ManCubeDCD_Delta( p, Fx_ManGetFirstVarCube(p, vCube), vCube, 1, pCtx );
    Fx_ManForEachCubeVec( pCand->vCubesD, p->vCubes, vCube, i )
        Fx_ManCubeDCD_Delta( p, Fx_ManGetFirstVarCube(p, vCube), vCube, 1, pCtx );
    Vec_WecUnmarkLevels( p->vCubes, pCand->vCubesS );
    Vec_WecUnmarkLevels( p->vCubes, pCand->vCubesD );

    // structural mutation: create new divisor node using pre-allocated slots
    if ( !(Abc_Lit2Var(Lit0) == Abc_Lit2Var(Lit1) && Vec_IntSize(pCand->vDiv) == 2) )
    {
        vCube = Vec_WecEntry( p->vCubes, pCand->iNewCubeBase );
        Vec_IntPush( vCube, iVarNew );
        if ( Vec_IntSize(pCand->vDiv) == 2 )
        {
            Vec_IntPush( vCube, Abc_LitNot(Lit0) );
            Vec_IntPush( vCube, Abc_LitNot(Lit1) );
            Level = 1 + Fx_ManComputeLevelCube( p, vCube );
        }
        else
        {
            vCube2 = Vec_WecEntry( p->vCubes, pCand->iNewCubeBase + 1 );
            Vec_IntPush( vCube2, iVarNew );
            Fx_ManDivAddLits( vCube, vCube2, pCand->vDiv );
            Level = 2 + Abc_MaxInt( Fx_ManComputeLevelCube(p, vCube), Fx_ManComputeLevelCube(p, vCube2) );
        }
        Vec_IntWriteEntry( p->vLevels, iVarNew, Level );

        // get pre-allocated literal arrays
        vLitP = Vec_WecEntry( p->vLits, 2 * iVarNew );
        vLitN = Vec_WecEntry( p->vLits, 2 * iVarNew + 1 );

        // update single-cube divisor cubes
        Fx_ManForEachCubeVec( pCand->vCubesS, p->vCubes, vCube, i )
        {
            RetValue  = Vec_IntRemove1( vCube, Abc_LitNot(Lit0) );
            RetValue += Vec_IntRemove1( vCube, Abc_LitNot(Lit1) );
            assert( RetValue == 2 );
            Vec_IntPush( vCube, Abc_Var2Lit(iVarNew, 0) );
            Vec_IntPush( vLitP, Vec_WecLevelId(p->vCubes, vCube) );
        }
    }
    else
    {
        vLitP = NULL;
        vLitN = NULL;
    }

    // extract from pairs
    k = 0;
    assert( Vec_IntSize(pCand->vCubesD) % 2 == 0 );
    for ( i = 0; i < Vec_IntSize(pCand->vCubesD); i += 2 )
    {
        int fCompl = Vec_IntEntry(pCand->vCompls, i/2);
        vCube  = Vec_WecEntry( p->vCubes, Vec_IntEntry(pCand->vCubesD, i) );
        vCube2 = Vec_WecEntry( p->vCubes, Vec_IntEntry(pCand->vCubesD, i+1) );
        RetValue  = Fx_ManDivRemoveLits( vCube, pCand->vDiv, fCompl );
        RetValue += Fx_ManDivRemoveLits( vCube2, pCand->vDiv, fCompl );
        assert( RetValue == Vec_IntSize(pCand->vDiv) || RetValue == Vec_IntSize(pCand->vDiv) + 1 );
        if ( iVarNew > 0 && vLitP != NULL )
        {
            if ( Vec_IntSize(pCand->vDiv) == 2 || fCompl )
            {
                Vec_IntPush( vCube, Abc_Var2Lit(iVarNew, 1) );
                Vec_IntPush( vLitN, Vec_WecLevelId(p->vCubes, vCube) );
            }
            else
            {
                Vec_IntPush( vCube, Abc_Var2Lit(iVarNew, 0) );
                Vec_IntPush( vLitP, Vec_WecLevelId(p->vCubes, vCube) );
            }
        }
        Vec_IntWriteEntry( pCand->vCubesD, k++, Vec_WecLevelId(p->vCubes, vCube) );
        Vec_IntClear( vCube2 );
    }
    Vec_IntShrink( pCand->vCubesD, k );
    Vec_IntSort( pCand->vCubesD, 0 );
}

/* OMP_PLACEHOLDER_3 */

/**Function*************************************************************

  Synopsis    [Merge deltas + add cost (sequential per divisor).]

***********************************************************************/
static void Fx_ManUpdateMergeAndAddCost( Fx_Man_t * p, Fx_Candidate_t * pCand, Fx_ThreadCtx_t * pCtx )
{
    Vec_Int_t * vCube, * vCube2;
    int i, iDiv, Lit0, fWarning = pCand->fWarning;
    int iVarNew = pCand->iVarNew;

    // apply accumulated weight deltas
    for ( i = 0; i < Vec_IntSize(pCtx->vDeltaDiv); i++ )
    {
        iDiv = Vec_IntEntry( pCtx->vDeltaDiv, i );
        if ( iDiv >= 0 && iDiv < Vec_FltSize(p->vWeights) )
        {
            Vec_FltAddToEntry( p->vWeights, iDiv, Vec_FltEntry(pCtx->vDeltaWt, i) );
            if ( Vec_QueIsMember(p->vPrio, iDiv) )
                Vec_QueUpdate( p->vPrio, iDiv );
        }
    }
    p->nPairsS += pCtx->nPairsS;
    p->nPairsD += pCtx->nPairsD;
    p->nDivs++;

    // add cost of single-cube divisors (sequential, can insert new divisors)
    Fx_ManForEachCubeVec( pCand->vCubesS, p->vCubes, vCube, i )
        Fx_ManCubeSingleCubeDivisors( p, vCube, 0, 1 );
    Fx_ManForEachCubeVec( pCand->vCubesD, p->vCubes, vCube, i )
        Fx_ManCubeSingleCubeDivisors( p, vCube, 0, 1 );

    // mark cubes, add cost of double-cube divisors
    Vec_WecMarkLevels( p->vCubes, pCand->vCubesS );
    Vec_WecMarkLevels( p->vCubes, pCand->vCubesD );
    Fx_ManForEachCubeVec( pCand->vCubesS, p->vCubes, vCube, i )
        Fx_ManCubeDoubleCubeDivisors( p, Fx_ManGetFirstVarCube(p, vCube), vCube, 0, 1, &fWarning );
    Fx_ManForEachCubeVec( pCand->vCubesD, p->vCubes, vCube, i )
        Fx_ManCubeDoubleCubeDivisors( p, Fx_ManGetFirstVarCube(p, vCube), vCube, 0, 1, &fWarning );
    Vec_WecUnmarkLevels( p->vCubes, pCand->vCubesS );
    Vec_WecUnmarkLevels( p->vCubes, pCand->vCubesD );

    // handle SCC from thread context
    if ( Vec_IntSize(pCtx->vSCC) )
    {
        Vec_IntUniqify( pCtx->vSCC );
        Fx_ManForEachCubeVec( pCtx->vSCC, p->vCubes, vCube, i )
        {
            Fx_ManCubeDoubleCubeDivisors( p, Fx_ManGetFirstVarCube(p, vCube), vCube, 1, 1, &fWarning );
            Vec_IntClear( vCube );
        }
    }

    // add cost of the new divisor
    if ( Vec_IntSize(pCand->vDiv) > 2 && !(Abc_Lit2Var(pCand->Lit0) == Abc_Lit2Var(pCand->Lit1) && Vec_IntSize(pCand->vDiv) == 2) )
    {
        vCube  = Vec_WecEntry( p->vCubes, pCand->iNewCubeBase );
        vCube2 = Vec_WecEntry( p->vCubes, pCand->iNewCubeBase + 1 );
        Fx_ManCubeSingleCubeDivisors( p, vCube,  0, 1 );
        Fx_ManCubeSingleCubeDivisors( p, vCube2, 0, 1 );
        Vec_IntForEachEntryStart( vCube, Lit0, i, 1 )
            Vec_WecPush( p->vLits, Lit0, Vec_WecLevelId(p->vCubes, vCube) );
        Vec_IntForEachEntryStart( vCube2, Lit0, i, 1 )
            Vec_WecPush( p->vLits, Lit0, Vec_WecLevelId(p->vCubes, vCube2) );
    }

    // remove cubes from lit arrays of the divisor
    Vec_IntForEachEntry( pCand->vDiv, Lit0, i )
    {
        Vec_IntTwoRemove( Vec_WecEntry(p->vLits, Abc_Lit2Var(Lit0)), pCand->vCubesD );
        if ( Vec_IntSize(pCand->vDiv) == 2 )
            Vec_IntTwoRemove( Vec_WecEntry(p->vLits, Abc_LitNot(Abc_Lit2Var(Lit0))), pCand->vCubesD );
    }
}
#endif /* ABC_USE_OMP */

/**Function*************************************************************

  Synopsis    [Implements the traditional fast_extract algorithm.]

  Description [J. Rajski and J. Vasudevamurthi, "The testability-
  preserving concurrent decomposition and factorization of Boolean
  expressions", IEEE TCAD, Vol. 11, No. 6, June 1992, pp. 778-793.]
               
  SideEffects []

  SeeAlso     []

***********************************************************************/
int Fx_FastExtract( Vec_Wec_t * vCubes, int ObjIdMax, int nNewNodesMax, int LitCountMax, int fCanonDivs, int fVerbose, int fVeryVerbose )
{
    int fVeryVeryVerbose = 0;
    int i, iDiv, fWarning = 0;
    Fx_Man_t * p;
    abctime clk = Abc_Clock();
    // initialize the data-structure
    p = Fx_ManStart( vCubes );
    p->LitCountMax = LitCountMax;
    p->fCanonDivs = fCanonDivs;
    Fx_ManCreateLiterals( p, ObjIdMax );
    Fx_ManComputeLevel( p );
    Fx_ManCreateDivisors( p );
    if ( fVeryVerbose )
        Fx_PrintDivisors( p );
    if ( fVerbose )
        Fx_PrintStats( p, Abc_Clock() - clk );
    // perform extraction
    p->timeStart = Abc_Clock();
#ifdef ABC_USE_OMP
    {
        int nThreads = omp_get_max_threads();
        if ( nThreads > 1 )
        {
            int K = nThreads * 2;
            int c, j, nPopped, nIndep;
            Fx_Candidate_t * pCands = Fx_CandAlloc( K );
            Fx_ThreadCtx_t * pCtxs  = Fx_ThreadCtxAlloc( nThreads );
            Vec_Int_t * vIndep   = Vec_IntAlloc( K );
            Vec_Int_t * vConflict = Vec_IntAlloc( K );

            for ( i = 0; i < nNewNodesMax && Vec_QueTopPriority(p->vPrio) > 0.0; )
            {
                // Phase A: Pop up to K candidates
                nPopped = 0;
                while ( nPopped < K && i + nPopped < nNewNodesMax && Vec_QueTopPriority(p->vPrio) > 0.0 )
                {
                    pCands[nPopped].iDiv = Vec_QuePop( p->vPrio );
                    nPopped++;
                }
                if ( nPopped == 0 ) break;

                // Phase B: Gather affected cubes (sequential)
                for ( c = 0; c < nPopped; c++ )
                    Fx_ManUpdateGather( p, &pCands[c] );

                // Phase C: Detect conflicts
                Fx_ManDetectConflicts( p, pCands, nPopped, vIndep, vConflict );
                nIndep = Vec_IntSize( vIndep );
                if ( nIndep == 0 )
                {
                    // all conflict - re-insert all and fall back to serial for one
                    for ( j = 1; j < nPopped; j++ )
                        Vec_QuePush( p->vPrio, pCands[j].iDiv );
                    Fx_ManUpdate( p, pCands[0].iDiv, &fWarning );
                    i++;
                    continue;
                }

                // Phase D: Pre-allocate cube/lit slots for independent divisors
                for ( j = 0; j < nIndep; j++ )
                {
                    int idx = Vec_IntEntry( vIndep, j );
                    Fx_Candidate_t * pC = &pCands[idx];
                    int fSkipNode = (Abc_Lit2Var(pC->Lit0) == Abc_Lit2Var(pC->Lit1) && Vec_IntSize(pC->vDiv) == 2);
                    if ( !fSkipNode )
                    {
                        int nNewCubes = (Vec_IntSize(pC->vDiv) == 2) ? 1 : 2;
                        pC->iVarNew = Vec_WecSize( p->vLits ) / 2;
                        pC->iNewCubeBase = Vec_WecSize( p->vCubes );
                        Vec_IntPush( p->vVarCube, pC->iNewCubeBase );
                        for ( c = 0; c < nNewCubes; c++ )
                            Vec_WecPushLevel( p->vCubes );
                        Vec_IntPush( p->vLevels, 0 );
                        p->nLits += Vec_IntSize( pC->vDiv );
                        Vec_WecPushLevel( p->vLits ); // vLitP
                        Vec_WecPushLevel( p->vLits ); // vLitN
                    }
                    else
                    {
                        pC->iVarNew = 0;
                        pC->iNewCubeBase = -1;
                    }
                }

                // Phase E: Subtract cost + structural mutation (PARALLEL)
                for ( j = 0; j < nThreads; j++ )
                    Fx_ThreadCtxClear( &pCtxs[j] );
                #pragma omp parallel for schedule(dynamic) num_threads(nThreads)
                for ( j = 0; j < nIndep; j++ )
                {
                    int tid = omp_get_thread_num();
                    int idx = Vec_IntEntry( vIndep, j );
                    Fx_ManUpdateSubtractAndMutate( p, &pCands[idx], &pCtxs[tid] );
                }

                // Phase F: Merge deltas + add cost (sequential)
                for ( j = 0; j < nIndep; j++ )
                {
                    int idx = Vec_IntEntry( vIndep, j );
                    // find which thread handled this - use round-robin approximation
                    // Actually, we need per-candidate ctx tracking. Simpler: merge all ctxs first.
                }
                // Merge all thread contexts
                for ( j = 0; j < nThreads; j++ )
                {
                    int d;
                    for ( d = 0; d < Vec_IntSize(pCtxs[j].vDeltaDiv); d++ )
                    {
                        int dDiv = Vec_IntEntry( pCtxs[j].vDeltaDiv, d );
                        if ( dDiv >= 0 && dDiv < Vec_FltSize(p->vWeights) )
                        {
                            Vec_FltAddToEntry( p->vWeights, dDiv, Vec_FltEntry(pCtxs[j].vDeltaWt, d) );
                            if ( Vec_QueIsMember(p->vPrio, dDiv) )
                                Vec_QueUpdate( p->vPrio, dDiv );
                        }
                    }
                    p->nPairsS += pCtxs[j].nPairsS;
                    p->nPairsD += pCtxs[j].nPairsD;
                }
                // Add cost + SCC handling per independent divisor (sequential)
                for ( j = 0; j < nIndep; j++ )
                {
                    int idx = Vec_IntEntry( vIndep, j );
                    // We need per-candidate SCC. Store SCC in candidate during subtract.
                    // For now, process all thread SCCs together after merge.
                    p->nDivs++;
                    // add cost of single-cube divisors
                    {
                        Vec_Int_t * vCube;
                        int ii, Lit0x;
                        Fx_ManForEachCubeVec( pCands[idx].vCubesS, p->vCubes, vCube, ii )
                            Fx_ManCubeSingleCubeDivisors( p, vCube, 0, 1 );
                        Fx_ManForEachCubeVec( pCands[idx].vCubesD, p->vCubes, vCube, ii )
                            Fx_ManCubeSingleCubeDivisors( p, vCube, 0, 1 );
                        // mark + add DCD cost
                        Vec_WecMarkLevels( p->vCubes, pCands[idx].vCubesS );
                        Vec_WecMarkLevels( p->vCubes, pCands[idx].vCubesD );
                        Fx_ManForEachCubeVec( pCands[idx].vCubesS, p->vCubes, vCube, ii )
                            Fx_ManCubeDoubleCubeDivisors( p, Fx_ManGetFirstVarCube(p, vCube), vCube, 0, 1, &fWarning );
                        Fx_ManForEachCubeVec( pCands[idx].vCubesD, p->vCubes, vCube, ii )
                            Fx_ManCubeDoubleCubeDivisors( p, Fx_ManGetFirstVarCube(p, vCube), vCube, 0, 1, &fWarning );
                        Vec_WecUnmarkLevels( p->vCubes, pCands[idx].vCubesS );
                        Vec_WecUnmarkLevels( p->vCubes, pCands[idx].vCubesD );
                        // add cost of new divisor cubes
                        if ( Vec_IntSize(pCands[idx].vDiv) > 2 && pCands[idx].iNewCubeBase >= 0 )
                        {
                            Vec_Int_t * vC1 = Vec_WecEntry( p->vCubes, pCands[idx].iNewCubeBase );
                            Vec_Int_t * vC2 = Vec_WecEntry( p->vCubes, pCands[idx].iNewCubeBase + 1 );
                            Fx_ManCubeSingleCubeDivisors( p, vC1, 0, 1 );
                            Fx_ManCubeSingleCubeDivisors( p, vC2, 0, 1 );
                            Vec_IntForEachEntryStart( vC1, Lit0x, ii, 1 )
                                Vec_WecPush( p->vLits, Lit0x, Vec_WecLevelId(p->vCubes, vC1) );
                            Vec_IntForEachEntryStart( vC2, Lit0x, ii, 1 )
                                Vec_WecPush( p->vLits, Lit0x, Vec_WecLevelId(p->vCubes, vC2) );
                        }
                        // remove from lit arrays
                        Vec_IntForEachEntry( pCands[idx].vDiv, Lit0x, ii )
                        {
                            Vec_IntTwoRemove( Vec_WecEntry(p->vLits, Abc_Lit2Var(Lit0x)), pCands[idx].vCubesD );
                            if ( Vec_IntSize(pCands[idx].vDiv) == 2 )
                                Vec_IntTwoRemove( Vec_WecEntry(p->vLits, Abc_LitNot(Abc_Lit2Var(Lit0x))), pCands[idx].vCubesD );
                        }
                    }
                }
                // Handle all thread SCCs
                for ( j = 0; j < nThreads; j++ )
                {
                    if ( Vec_IntSize(pCtxs[j].vSCC) )
                    {
                        Vec_Int_t * vCube;
                        int ii;
                        Vec_IntUniqify( pCtxs[j].vSCC );
                        Fx_ManForEachCubeVec( pCtxs[j].vSCC, p->vCubes, vCube, ii )
                        {
                            Fx_ManCubeDoubleCubeDivisors( p, Fx_ManGetFirstVarCube(p, vCube), vCube, 1, 1, &fWarning );
                            Vec_IntClear( vCube );
                        }
                    }
                }

                i += nIndep;

                // Phase G: Re-insert conflicting divisors
                for ( j = 0; j < Vec_IntSize(vConflict); j++ )
                {
                    int idx = Vec_IntEntry( vConflict, j );
                    Vec_QuePush( p->vPrio, pCands[idx].iDiv );
                }
            }
            Fx_CandFree( pCands, K );
            Fx_ThreadCtxFree( pCtxs, nThreads );
            Vec_IntFree( vIndep );
            Vec_IntFree( vConflict );
        }
        else
        {
            // single thread - use original serial loop
            for ( i = 0; i < nNewNodesMax && Vec_QueTopPriority(p->vPrio) > 0.0; i++ )
            {
                iDiv = Vec_QuePop(p->vPrio);
                if ( fVeryVerbose )
                    Fx_PrintDiv( p, iDiv );
                Fx_ManUpdate( p, iDiv, &fWarning );
                if ( fVeryVeryVerbose )
                    Fx_PrintDivisors( p );
            }
        }
    }
#else
    for ( i = 0; i < nNewNodesMax && Vec_QueTopPriority(p->vPrio) > 0.0; i++ )
    {
        iDiv = Vec_QuePop(p->vPrio);
        if ( fVeryVerbose )
            Fx_PrintDiv( p, iDiv );
        Fx_ManUpdate( p, iDiv, &fWarning );
        if ( fVeryVeryVerbose )
            Fx_PrintDivisors( p );
    }
#endif
    if ( fVerbose )
        Fx_PrintStats( p, Abc_Clock() - clk );
    Fx_ManStop( p );
    // return the result
    Vec_WecRemoveEmpty( vCubes );
    return 1;
}



////////////////////////////////////////////////////////////////////////
///                       END OF FILE                                ///
////////////////////////////////////////////////////////////////////////


ABC_NAMESPACE_IMPL_END

