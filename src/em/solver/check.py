from generaterelation import checkset


# nsetA = {0, 1, 2, 3, 4, 5, 8, 9, 11, 13, 14, 16, 24, 25, 27}
nsetB = {0, 1, 2, 3, 4, 5, 8, 9, 11, 13, 14, 16, 24, 25, 27}
print(checkset('data/get_hypergraph_violence.json', 'data/newproblemoperations_violence.json', 'data/newproblemgroups_violence.json',
               ('SegmentEqualSegment', 'B', 'E', 'D', 'C'), nsetA=False, nsetB=nsetB, nsetO=False))

