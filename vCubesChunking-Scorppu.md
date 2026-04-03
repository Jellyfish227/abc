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




