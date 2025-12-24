from generaterelation import checkset


nsetA = {0, 1, 2, 3, 4, 5, 8, 9, 11, 13, 14, 16, 24, 25, 27}
print(checkset('data/get_hypergraph_num.json', 'data/newproblemoperations_num.json', 'data/newproblemgroups_num.json',
               ('SegmentEqualSegment', 'B', 'E', 'D', 'C'), nsetA=nsetA, nsetB=False, nsetO=False))

