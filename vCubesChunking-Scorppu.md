# Scorppu's two approaches to vCubes chunking

## Two approaches to this parallel solution at the moment:

1. tracking the max change in objID for each partition and using it as an offset for its successor
2. maintaining a separate list of newly added nodes from extraction, outside of each partition's original vector of nodes.

## Approach 1 (track max objID offset):

Although we can make sure that each partition only serves a certain objID range, during extraction, we cannot stop each partition from creating new nodes with significantly larger objIDs.

when it comes to merging the vectors into a singular vCubes, the objIDs will inevitably be out of order and will cause assertion failures in Abc_NtkFxInsert().

Therefore, we want to track the maximum change in objID, where 

`maxChange = highest newObjID - ObjIDrange[1]`.

```python
class partition:
    self.id
    self.vCubes
    self.maxChange

currentOffset = 0
for partition in partitions: # partitions must be ordered
    if partition.id == 0:
        vCubes += partition.vCubes
        currentOffset = partition.maxChange
    else:
        for node in partition.vCubes:
            node[0] += currentOffset # first item in each vector of vCubes is the objID
        vCubes += partition.vCubes
        currentOffset += partition.maxChange
```

### Possible points of failure for this approach:

1. if cec checks for logical equivalence by objID, the offset added to each partition will cause original nodeIDs to be shifted as well. This means that the logical circuit will be considered inequivalent to the original.

## Approach 2 (separate vector for new nodes):

This approach aims to solve the point of failure for approach 1

Instead of adding an offset to each node's objID in vCubes, we allow the original cubes in vCubes to maintain their objID. We then have a new list for each partition, that contains any new nodes created by `Fx_ManUpdate()`

When mergings, the original cubes can be merged without sorting, and the new cubes can be sorted and inserted into the list by order.

### Caveats for approach 2

- `Fx_ManUpdate` and the `vPrio` will have to read from two separate lists
- We can't guarantee insertion of the new Cubes into the merged list is fast, even if we maintain the start/end index of each node

#### it is possible for us to have a combination of both approaches, and make sure all newly created nodes are sorted and exceed the max objID range of the original cubes. Since the new objIDs are not refered to by the original network, it is possible for this to be the best solution.


# Final Approach (combination of approach 1 and 2)

## Overview

The traditional FX algorithm runs serially over the entire `vCubes` array.
Since two-cube divisors are already node-local (enforced by the early break in
`Fx_ManCubeDoubleCubeDivisors`), we can partition `vCubes` into independent
chunks and run FX on each in parallel. This trades global optimality for
a near-linear speedup.

**Accepted tradeoff:** divisors whose constituent cubes span a partition
boundary are never found. In practice this is acceptable if partitions are
chosen with locality awareness.

**Important:** The original `vCubes` from `Abc_NtkFxRetrieve` is freed
immediately after partitioning. We work entirely on deep-copied per-partition
`Vec_Wec_t*` instances. The final output to `Abc_NtkFxInsert` is a
freshly-built merged `Vec_Wec_t*` — the original is never mutated.

---

## Phase 1 — Partitioning vCubes  [single thread, O(n)]

Split `vCubes` into P chunks at **node boundaries** — never split cubes of the
same node across partitions, as the entire algorithm assumes all cubes of one
node form a contiguous block.

Each partition is a **deep copy** of its row range from the original `vCubes`.
`Fx_FastExtract` mutates cubes in-place and appends new rows via
`Vec_WecPushLevel` — with each partition having private storage, there is zero
cross-thread aliasing.

The original `vCubes` is freed after all partition copies are made.

For each partition, record:
- `partition.vCubes`         — deep-copied `Vec_Wec_t*` for this partition
- `partition.local_ObjIdMax` — the highest original ObjId in this partition

Also compute:
- `global_ObjIdMax = max(local_ObjIdMax[0..P-1])` — the highest ObjId across
  **all** original cubes in the entire network (before any extraction). All
  remapped new-node IDs will be placed strictly above this ceiling.

---

## Phase 2 — Independent FX Extraction  [P threads, O(n²/P) each]

Each partition runs a full, unmodified `Fx_FastExtract` call in isolation:

```

Fx_ManStart()
Fx_ManCreateLiterals()
Fx_ManComputeLevel()
Fx_ManCreateDivisors()
main extraction loop → Fx_ManUpdate() ...

```

Each partition has its own `vLits`, `vWeights`, `pHash`, `vPrio`, and
`vVarCube`. There is zero shared mutable state between threads.

New nodes created within a partition get IDs assigned locally starting from
that partition's `local_ObjIdMax + 1`. These IDs are temporary and will be
globally remapped in Phase 4.

> **Key invariant:** `Fx_ManUpdate` only ever rewrites cubes within the
> current partition. New node literals are pushed into old cubes via
> `Vec_IntPush(vCube, Abc_Var2Lit(iVarNew, 0))`, but `vCube` always belongs
> to the same partition. New node IDs therefore never escape their partition.

**Barrier** — wait for all partitions to finish.

---

## Phase 3 — Count New Nodes Per Partition  [P threads, O(n/P) each]

Each thread scans its own partition's `vCubes`. A cube belongs to a new node if:

```

cube[0] > partition.local_ObjIdMax

```

Count these per partition into a shared array `counts[P]`.

**Barrier** — then compute prefix sums on a single thread (O(P), negligible):

```

// global_ObjIdMax = max ObjId across ALL partitions' original nodes
// All offsets are strictly above this ceiling → no overlap with any original ObjId

offsets[0] = global_ObjIdMax + 1
offsets[i] = offsets[i-1] + counts[i-1]   for i >= 1

```

**Critical:** `offsets[0]` is anchored to `global_ObjIdMax + 1`, not any
local partition's max. For example, if partitions have original ranges
[1–100], [101–200], [401–500] and `global_ObjIdMax = 500`:

```

offsets[0] = 501                         → partition 0 new nodes: [501, 501+counts[0])
offsets[1] = 501 + counts[0]             → partition 1 new nodes: never overlap [401–500]
offsets[2] = 501 + counts[0] + counts[1]

```

No remapped ID can land inside any original partition's ObjId range.

---

## Phase 4 — ID Remapping  [P threads, O(n/P) each]

Each thread remaps its own partition's cubes only. No shared writes.

For every cube in partition `j`:
- `cube[0]` (node owner): if `> local_ObjIdMax[j]`, remap:
  ```
  newId = offsets[j] + (cube[0] - (local_ObjIdMax[j] + 1))
  cube[0] = newId
  ```
- `cube[k]` for `k >= 1` (literals = `2*ObjId + compl_bit`): if
  `Abc_Lit2Var(lit) > local_ObjIdMax[j]`, remap the embedded ObjId:
  ```
  newVar = offsets[j] + (Abc_Lit2Var(lit) - (local_ObjIdMax[j] + 1))
  cube[k] = Abc_Var2Lit(newVar, Abc_LitIsCompl(lit))
  ```

Original node IDs (`<= local_ObjIdMax[j]`) are **never touched**, preserving
correctness for `Abc_NtkFxInsert`'s sorted-order assertion and for CEC.

**Barrier**

---

## Phase 5 — Merging  [single thread, O(n)]

Build a fresh `Vec_Wec_t * vMerged`. Concatenate in two passes:

**Pass 1 — original cubes (all partitions, in order):**
```

[partition 0 original cubes]   ← cube[0] <= local_ObjIdMax[0]
[partition 1 original cubes]   ← cube[0] <= local_ObjIdMax[1], all > partition 0 range
[partition 2 original cubes]
...

```

**Pass 2 — new node cubes (all partitions, in order):**
```

[partition 0 new node cubes]   ← remapped IDs starting at offsets[0]
[partition 1 new node cubes]   ← remapped IDs starting at offsets[1] > offsets[0]
[partition 2 new node cubes]
...

```

Since original ObjIds are untouched and partitions are ordered, Pass 1 is
already globally sorted. Pass 2 IDs are all `> global_ObjIdMax` and increase
by partition order. The combined result satisfies `Abc_NtkFxInsert`'s
`Lit <= vCube[0]` ordering assertion at line 195.

Finally, call `Abc_NtkFxInsert` on `vMerged`.

---

## Phase Summary

| Phase | Parallel? | Cost |
|---|---|---|
| Partition vCubes (deep copy) | No | O(n) |
| FX Extraction | **Yes** | O(n²/P) per thread — dominant |
| Count new nodes | **Yes** | O(n/P) per thread |
| Prefix sum (offset calc) | No | O(P), negligible |
| ID Remap | **Yes** | O(n/P) per thread |
| Merge | No | O(n) |

---

## Known Limitations

- Divisors whose cubes span a partition boundary are never found. Quality of
  extraction depends on how well partitions are chosen — grouping nodes that
  share fanins into the same partition minimises missed divisors.
- The new node budget per partition is unknown before extraction runs, making
  static pre-allocation impossible. This is why counting and prefix-sum is
  done as a separate post-extraction phase rather than pre-allocated ranges.
- The offset anchor (`global_ObjIdMax + 1`) must account for the **entire
  network's** max ObjId, not any single partition's local max. Using a local
  max would allow remapped IDs from one partition to collide with original
  ObjIds in a later partition, violating `Abc_NtkFxInsert`'s sort assertion.
