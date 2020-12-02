import utilities
from modified_nj import mod_nj
from Node import CellGroupNode

import numpy as np
import pandas as pd 

def greedy_probabilistic(simulation,
                        impute_missing_data=False,
                        max_cells_per_greedy_node = 2,
                        reassign = True,
                        consensus_assignment = False):
    
    print('Reconstructing greedy tree')
    if impute_missing_data:
        simulation.imputation_on()
    else:
        simulation.imputation_off()

    true_tree = simulation.get_cleaned_tree()

    print('Got true_tree')
    character_matrix = simulation.get_final_cells()
    character_matrix.drop_duplicates()
    RAND_MAX = 100000 #HACKY


    character_matrix = character_matrix.drop_duplicates()
    character_matrix.index = utilities.character_matrix_to_labels(character_matrix)

    # Convert character matrix to binary feature matrix for neighbour joining 
    feature_matrix = utilities.binarize_character_matrix(character_matrix)

    dels_probs = simulation.get_edit_probs()
    prob_features = dels_probs[feature_matrix.columns].values


    # Split by most frequently occuring character until cell groups are small enough to do neighbour joining efficiently. 
    def recursive_split(fm, cm):
        if fm.shape[0] <= max_cells_per_greedy_node:

            # Create a node containing all the cells in this subtree
            leaf = CellGroupNode(f'lf-{np.random.randint(RAND_MAX)}') 
            leaf.add_feature_matrix(fm)
            leaf.add_character_matrix(cm)
            return leaf

        # Determine the mutation which appears in the largest number of cells (but not all the cells) 
        proportion = fm.sum(axis=0)/fm.shape[0]
        proportion[proportion==1] = 0

        split_on = fm.columns[np.argmax(proportion)]


        left_child = fm[fm[split_on]==1]
        right_child = fm[fm[split_on]!=1]

        if len(left_child)==0 or len(right_child)==0:
            # Return here, no further splitting is possible
            leaf = CellGroupNode(f'lf-{np.random.randint(RAND_MAX)}') 
            leaf.add_feature_matrix(fm)
            leaf.add_character_matrix(cm)

            return leaf

        left_cm = cm.loc[left_child.index]
        right_cm = cm.loc[right_child.index]

        node = CellGroupNode(f'hdn-{np.random.randint(RAND_MAX)}')

        left_child = recursive_split(left_child, left_cm)        
        left_child.add_defining_muts((split_on, 1))

        right_child = recursive_split(right_child, right_cm)
        right_child.add_defining_muts((split_on, 0))
        left_child.parent = node
        right_child.parent = node 

        node.children = [left_child, right_child]
        return node  

    greedy_tree = recursive_split(feature_matrix, character_matrix)

    # Perform probabilistic re-assignment of leaves 
    # For each leaf node, determine the profile of mutations that characterize the tip.

    # Option 1: TODO: Use all the defining mutations accumulated along this lineage
    # Option 2: Use the consensus over all cells 

    star_tree = greedy_tree.copy()

    if greedy_tree.is_tip():
        print('Greedy tree contains only one cell. No need to reassign cells.')
        reassign = False
        leaves = [star_tree]


    print('Computing score before reassignment.')
    leaves = [x for x in star_tree.tips()]

    muts_in_leaf = []
    for i, leaf in enumerate(leaves):
        # Determine consensus of leaf, aka which mutations are present in all cells in the group.
        consensus_muts = (leaf.feature_matrix.sum(0)/leaf.feature_matrix.shape[0]) == 1
        consensus_muts = consensus_muts.astype(int).values
        muts_in_leaf.append(consensus_muts)

    scores = {}
    # Compute triplets correct before reassignment

    for leaf in leaves:
        # Create a star tree for the unresolved cells at the tips
        for name in leaf.char_matrix.index:
            node = CellGroupNode(str(name), parent=leaf)
            node.add_character_matrix(leaf.char_matrix.loc[name])        
            node.add_feature_matrix(leaf.feature_matrix.loc[name])

            leaf.children.append(node)
    star_tree.condense_duplicate_leaves()
    score = utilities.triplets_correct(true_tree, star_tree)
    print('Score before reassignment: ', score)
    scores['before'] = score

    if reassign:
        # Compute the probabilistic assignment of cells to cellgroup leaves
        fm = []
        cm = []
        original_leaf = []

        for i, leaf in enumerate(leaves):
            fm.append(leaf.feature_matrix)
            cm.append(leaf.get_character_matrix())
            if cm[-1].shape[1] != 30:
                assert False

            original_leaf += [i]*len(fm[-1])


        fm = pd.concat(fm, axis=0)
        cm = pd.concat(cm, axis=0)



        original_leaf = np.array(original_leaf)
        muts_in_leaf = np.vstack(muts_in_leaf).T

        if consensus_assignment:
            # Assign cells based on the number of shared mutations with each leaf
            leaf_agreement = np.dot(fm[leaf.feature_matrix.columns].values, muts_in_leaf)
            new_assignments = leaf_agreement.argmax(1)

        else:
            # For each leaf, look at all the mutations it contains. 
            # Compute the probability of a cell belonging there = prob(contains all these muts) 
            prob_existing_edits = fm[leaf.feature_matrix.columns].values * prob_features
            new_assignments = np.argmax(np.dot(np.nan_to_num(prob_existing_edits),muts_in_leaf), axis=1)

        changed_assignments = (new_assignments != original_leaf).sum()
        print(f'Reassigned {changed_assignments} cells out of a total of {len(new_assignments)} cells.')

        fm['Updated_CellGroup'] = new_assignments
        cm['Updated_CellGroup'] = new_assignments

        # Reallocate the leaf indices of all cells into the original tree structure.
        for i, leaf in enumerate(leaves):
            leaf.add_feature_matrix(fm[fm["Updated_CellGroup"] == i].drop('Updated_CellGroup', axis=1))
            leaf.add_character_matrix(cm[cm["Updated_CellGroup"] == i].drop('Updated_CellGroup', axis=1))

        # If any leaf no longer has any cells, remove from the tree. 
        n_removed = 0
        for i, leaf in enumerate(leaves):
            if len(leaf.feature_matrix) == 0:
                n_removed += 1
                leaf.parent.children.remove(leaf)
                while len(leaf.parent.children) == 0:
                    leaf = leaf.parent
                    leaf.parent.children.remove(leaf)

        print(f'Removed {n_removed} leaves after reassigning cells.')

        # Compute the triplets correct after reassigning cells
        star_tree = greedy_tree.copy()
        for leaf in star_tree.tips():
            if leaf.name[:3] == 'hdn':
                # Internal nodes should node be considered since they have no associated cells.
                continue

            # Create a star tree for the unresolved cells at the tips
            for name in leaf.char_matrix.index:
                node = CellGroupNode(str(name), parent=leaf)
                node.add_character_matrix(leaf.char_matrix.loc[name])        
                node.add_feature_matrix(leaf.feature_matrix.loc[name])

                leaf.children.append(node)
        star_tree.condense_duplicate_leaves()
        score = utilities.triplets_correct(true_tree, star_tree)

        print('Score after reassignment: ', score)
        scores['after'] = score

    # Finally, perform neighbor joining in each leaf to complete the tree.

    leaves = [x for x in greedy_tree.tips()]
    if len(leaves) == 0:

        leaf = greedy_tree
        fm = leaf.feature_matrix 
        if len(fm) < 2:
            name=fm.index[0]
            root = CellGroupNode(name)
            root.add_character_matrix(leaf.get_character_matrix().loc[name])
            root.add_feature_matrix(leaf.get_feature_matrix().loc[name])
        else:
            root, steps = mod_nj(fm, leaf.get_character_matrix(), prob_features)
        greedy_tree = root

    else:
        for leaf in leaves:
            fm = leaf.feature_matrix 
            if len(fm) < 2:
                name=fm.index[0]
                root = CellGroupNode(name)
                root.add_character_matrix(leaf.get_character_matrix().loc[name])
                root.add_feature_matrix(leaf.get_feature_matrix().loc[name])
            else:
                root, steps = mod_nj(fm, leaf.get_character_matrix(), prob_features)

            root.parent = leaf.parent  
            # Replace leaf with new root 
            leaf.parent.children.remove(leaf)
            leaf.parent.children.append(root)

    greedy_tree.condense_duplicate_leaves()
    score = utilities.triplets_correct(true_tree, greedy_tree)
    print('Score after neighbor joining: ', score)
    scores['nj'] = score
    return greedy_tree, scores