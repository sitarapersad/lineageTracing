import numpy as np
import pandas as pd
import networkx as nx
import time 

import utilities

from cassiopeia.TreeSolver.Cassiopeia_Tree import Cassiopeia_Tree
from cassiopeia.TreeSolver.score_triplets import score_triplets
from cassiopeia.TreeSolver.Node import Node 
from cassiopeia.TreeSolver.lineage_solver.lineage_solver import solve_lineage_instance

def cassiopeia_reconstruct(simulation):
    print('Reconstructing Cassiopeia Tree')
    
    priors = None 
    character_matrix = simulation.get_final_cells()

    # Cassiopeia takes a string dataframe
    cm = character_matrix.replace(np.nan, -1)
    cm = cm.astype(int).astype(str).replace('-1','-')
    cm_uniq = cm.drop_duplicates(inplace=False)
    target_nodes = cm_uniq.values.tolist()
    target_nodes = list(map(lambda x, n: Node(n,x), target_nodes, cm_uniq.index))


    t = time.time()
    reconstructed_network_greedy = solve_lineage_instance(target_nodes, 
                                                          method="greedy", 
                                                          prior_probabilities=priors)
    cass_time = time.time()-t
    cass_network = reconstructed_network_greedy[0]
    true_tree = simulation.get_cleaned_tree()
    true_network = utilities.convert_tree_to_nx(true_tree) 
    
    cass_score = score_triplets(true_network, cass_network, number_of_trials = 5000, min_size_depth = 20)

    # # Why is the score_triplets not 1??
    base_score = score_triplets(true_network, true_network, number_of_trials = 5000, min_size_depth = 20)

    cass_tree, duplicates = utilities.convert_nx_to_tree(cass_network.network)
    
    our_score = utilities.triplets_correct(true_tree, cass_tree)

    print('Baseline:', base_score, 'Cassiopeia:', our_score)

    return cass_tree, {'cass_score':cass_score, 'base_score':base_score, 'our_score':our_score}
