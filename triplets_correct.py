from skbio import TreeNode
import numpy as np 

"""
To compare the similarity of simulated trees to reconstructed trees, 
we take an approach which compares the
sub-trees formed between triplets of the terminal states
across the two trees. To do this, we sample ∼ 10, 000
triplets from our simulated tree and compare the relative
orderings of each triplet to the reconstructed tree. We say
a triplet is “correct” if the orderings of the three terminal 
states are conserved across both trees. This approach
is different from other tree comparison statistics, such
as Robinson-Foulds [34], which measures the number of
edges that are similar between two trees.
To mitigate the effect of disproportionately sampling
triplets relatively close to the root of the tree, we calculate
the percentage of triplets correct across each depth within
the tree independently (depth measured by the distance
from the root to the latest common ancestor (LCA) of
the triplet). We then take the average of the percentage
triplets correct across all depths. To further reduce the
bias towards the few triplets that are sampled at levels
of the tree with very few cells (i.e., few possible triplets),
we modify this statistic to only take into account depths
where there at least 20 cells. We report these statistics
without this depth threshold in Additional file 1: Fig S8
           
Ordering -> Cells share common ancestors in the correct order:

{{AB}C} vs {A{BC}}

"""

def nj_tree(feature_matrix):
    from skbio import DistanceMatrix
    from skbio.tree import nj
    import sklearn
    import time
    t = time.time()
    
    data = sklearn.metrics.pairwise_distances(feature_matrix.values, metric='hamming')
    print(time.time()-t)
    t = time.time()
    
    dm = DistanceMatrix(data)
    
    print('distance matrix', time.time()-t)
    t = time.time()
    
    tree = nj(dm)
    
    print('tree build', time.time()-t)
    
    return tree

def upgma_tree(feature_matrix):
    from scipy.cluster.hierarchy import linkage
    # Average in SciPy's cluster.hierarchy.linkage is UPGMA
    linkage_matrix = linkage(feature_matrix, method='average')
    tree = TreeNode.from_linkage_matrix(linkage_matrix,feature_matrix.index.values)
    
    return tree

def get_ordering(tree, n1, n2, n3):
    """
    Return the node that is furthest phylogenetically from the others.
    e.g. If (a,b),c is the true ordering, return c
    
    The lca of (a,c) and (b,c) will the same.
    """
    if tree.lca([n1, n2]) == tree.lca([n1, n3]):
        return n1
    if tree.lca([n1, n2]) == tree.lca([n2, n3]):
        return n2
    if tree.lca([n1, n3]) == tree.lca([n2, n3]):
        return n3

def triplets_correct(simulation, inferred_trees= ['nj', 'upgma'], n_samples=10):
    
    feature_matrix = simulation.get_feature_matrix()
    print('Got feature matrix')
    
    if isinstance(inferred_trees, dict):
        trees = inferred_trees
        n_correct = {}
        for k in trees:
            n_correct[k] = 0
    else:
        trees = {}
        n_correct = {}
        if 'nj' in inferred_trees:
            trees['nj'] = nj_tree(feature_matrix)
            n_correct['nj'] = 0
        if 'upgma' in inferred_trees:
            trees['upgma'] = upgma_tree(feature_matrix)
            n_correct['upgma'] = 0
    
    simulation.trees = trees
    for i in range(n_samples):
        # Sample a triplet of cells 
        ix = np.random.choice(feature_matrix.index.values, 3, replace=False)
        n1, n2, n3 = ix

        true = get_ordering(simulation.true_tree, str(n1), str(n2), str(n3))
        for t in trees:
            if t in ['nj']:
                furthest = get_ordering(trees[t], str(n1), str(n2), str(n3))
            else:
                furthest = get_ordering(trees[t], n1, n2, n3)
            if furthest == true:
                n_correct[t] += 1/n_samples

    return trees, n_correct

## -------------------------------------------------- Deprecated ----------------------------------------

def compute_triplets_correct(final_cells_labeled, subsampled_ix, sample_size = 1000):
    def get_ordering(a,b,c):
        """
        Return which cell (A,B or C) was least related of the triplet.
        e.g.
             ___A    
         ____|
        |    |___B
        |    
        |________C

        returns C
        """

        # The last (least significant) digit which agree across cells is the depth of their most recent common ancestor

        max_depth = {}
        for i in range(min(len(a),len(b))):
            if a[i] != b[i]:
                max_depth['ab'] = i
                break

        for i in range(min(len(a),len(c))):
            if a[i] != c[i]:
                max_depth['ac'] = i
                break


        for i in range(min(len(c),len(b))):
            if c[i] != b[i]:
                max_depth['bc'] = i
                break

        vals = list(max_depth.values())
        max_val = max(vals)
        if len(set(vals)) == 1:
            # All are at same depth
            return '='
        elif max_depth['ac'] == max_val:
            return 'B'
        elif max_depth['bc'] == max_val:
            return 'A'
        if max_depth['ab'] == max_val:
            return 'C'

    def cell_ix_to_label(i):
        return (format(i-1, '#022b')[2:])

    num_correct = 0
    for _ in range(sample_size):
        # Randomly sample triplets (a,b,c) and determine the relative ordering of ancestry.
        # How many are consistent with the truth?
        ix = np.random.choice(np.arange(subsampled_ix.shape[0]), 3, replace=False)
        a,b,c = (cell_ix_to_label(i) for i in subsampled_ix[ix])
        truth = get_ordering(a,b,c)

        # Grab the corresponding cells from our final array
        a,b,c = final_cells_labeled['Label'].iloc[ix]
        inferred = get_ordering(a,b,c)

        if truth == inferred:
            num_correct += 1

    print('Proportion correct = ', num_correct/sample_size)
    
    return num_correct/sample_size