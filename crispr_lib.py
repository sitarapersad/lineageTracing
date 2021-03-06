import numpy as np
import pandas as pd
import torch
import os
import time 
import copy
import random
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set(font_scale=1)

from skbio import TreeNode
from SimulationResult import SimulationResult
from identify_recurrent import identify_recurrent

import networkx as nx
from Node import CellGroupNode 
from cassiopeia.TreeSolver.Node import Node

def performCRISPR(cell_recorder, 
                  num_edit_sites, 
                  mutation_probs,
                  deletions_probs,
                  ):
    """
    Perform one generation of crispr editing before cells replicate.
    """
    num_cells = cell_recorder.shape[0]

    # For each target, mutate with a fixed probability
    # This probability empirically estimated from MSKCC data with 
    # the assumption of fixed reproduction time.

    # An edit occurs if: 
    # (1) the probability drawn is sufficient and 
    mutating = np.random.random((num_cells, num_edit_sites)) <= mutation_probs

    # (2) the site is not already edited.
    not_edited = cell_recorder == 0 

    num_available_sites = not_edited.sum()

    edit_here = mutating & not_edited
    num_edits_made = edit_here.sum()

    novel_edits = edit_here
    
    if edit_here.sum() > 0:
        # For the cells and targets which are mutating, now draw a deletion 
        # from the distribution specific to that. Deletions are indexed starting at 1
        deletion_choice = 1 + torch.multinomial(torch.DoubleTensor(deletions_probs), num_cells, replacement=True).numpy().T
        deletion_choice[~edit_here] = 0
        novel_edits = deletion_choice 
        cell_recorder += deletion_choice 

    # New cell_recorder has double the entries after replication
    # TODO: Only allow some cells to replicate    
    cell_recorder = np.repeat(cell_recorder, 2, axis=0)
    novel_edits = np.repeat(novel_edits, 2, axis=0)
    
    return cell_recorder, novel_edits, num_available_sites, num_edits_made

def get_new_ix(y, parent_reix, tree_depth):
    to_keep = [np.unique(y)]
    for i in range(tree_depth):
        y = y//2
        to_keep.append(np.unique(y))
    to_keep = list(reversed(to_keep))

    new_ix = []
    for i in range(0, tree_depth+1):
        mp = np.vectorize(lambda x: parent_reix.get(x//2, None))

        re_mapped = mp(to_keep[i])
        new_ix.append(2*re_mapped + np.array(to_keep[i])%2)

        array = np.arange(2**i)
        parent_reix = dict(zip(to_keep[i], np.arange(len(to_keep[i]))))

    return to_keep, new_ix

def simulate_imbalanced_tree(num_init_cells, init_death_prob = 0.1, init_repr_prob = 0.75, cancer_prob = 1e-3, tree_depth=15):
    num_cells = num_init_cells
    death_probs = [init_death_prob]*num_cells
    repr_probs = [init_repr_prob]*num_cells


    init_cells = [str(i) for i in np.arange(num_cells)]
    parent_ix = []
    cell_names = [np.array(init_cells)]
    repetition_coefs_list = []

    for i in range(tree_depth):
        dying = np.random.random(num_cells) < death_probs
        mutating = np.random.random(num_cells) < repr_probs

        repetition_coefs = (mutating+1)*(1-dying)

        repetition_coefs_list.append(repetition_coefs)
        next_gen = np.repeat(init_cells, repetition_coefs)
        
        if len(next_gen) == 0:
            raise Exception('No cells left to replicate. Terminate simulation.')
        # Label generation in terms of binary split with parents
        binary_labels = [next_gen[i]+'1' if next_gen[i-1]==next_gen[i] else next_gen[i]+'0' for i in range(1, len(next_gen))]
        binary_labels = [next_gen[0]+'0'] + binary_labels
        cell_names.append(np.array(binary_labels))
        parent_ix.append(np.repeat(np.arange(num_cells), repetition_coefs))

        death_probs = np.repeat(death_probs , repetition_coefs)
        repr_probs = np.repeat(repr_probs , repetition_coefs)

        num_cells = sum(repetition_coefs)
        
        init_cells = binary_labels

        # Introduce cancerous mutations which may increase tumour fitness
        has_cancer = np.random.random(num_cells) < cancer_prob
        death_probs[has_cancer] -= 1e-2
        repr_probs[has_cancer] += 1e-2

    return num_cells, cell_names, parent_ix, repetition_coefs_list
    
def lineageSimulationImbalanced(label, tree_depth, num_sites, deletions_probs, 
                          mutation_probs, edit_probs, compute_tree=False, 
                          init_cells=1, n_subsample=10000, missing_fraction=0, 
                          init_death_prob = 0.1, init_repr_prob = 0.75, cancer_prob = 1e-3, 
                          identify_recurrent_muts=True):
    
    start = time.time()
    simulation = SimulationResult(label, init_cells, tree_depth, num_sites, edit_probs, run=None)
    
    
    num_cells, cell_names, parent_ix, repetition_coefs_list = simulate_imbalanced_tree(num_init_cells=init_cells, 
                                                                init_death_prob = init_death_prob, 
                                                                init_repr_prob = init_repr_prob, 
                                                                cancer_prob = cancer_prob, 
                                                                tree_depth = tree_depth)
    
    
    if n_subsample > num_cells:
        print('There are insufficient cells to subsample. Using full simulation ({0} cells).'.format(num_cells))
        n_subsample = num_cells
    
    # Now we can subsample cells from the final round and determine the lineage
    subsampled_ix = np.sort(np.random.choice(num_cells, n_subsample, replace=False))

    # Subsample the parents to keep in the trees 
    subsampled_parents = [subsampled_ix]
    level_ix = subsampled_ix

    parent_maps= {}
    for i in range(tree_depth-1, -1, -1):
        level_ix = np.unique(parent_ix[i][level_ix])
        subsampled_parents.append(level_ix)

    subsampled_parents = list(reversed(subsampled_parents))

    keep_labels = [np.array(cell_names[i])[subsampled_parents[i]] for i in range(len(subsampled_parents))]
    y = [[i[:-1] for i in keep_labels[level]] for level in range(1,len(keep_labels))]
    keep_ix = []

    init_cells = keep_labels[0]
    
    num_init_cells = len(init_cells)

    parent_ix_levels = []

    for depth, level in enumerate(y):
        nodes, counts = np.unique(level, return_counts=True)
        n_cells = len(nodes)
        options = np.arange(n_cells * 2).reshape(-1,2)

        keep_ix_level = np.sort(np.hstack([np.random.choice(options[i], counts[i], replace=False) for i in range(len(counts))]))
        keep_ix.append(keep_ix_level)

        parent_ix_levels.append(dict(zip(np.arange(sum(counts)), np.repeat(np.arange(len(counts)),counts))))
    
    simulation.add_parent_child_map(parent_ix_levels)
    simulation.add_node_labels(keep_labels)
    
    simulation.add_subsampled_ix(subsampled_ix)
   
    # Build a subsampled tree

    level_ix = subsampled_ix
    
    # Create tips corresponding to each of the sampled cells
    tips = [CellGroupNode(keep_labels[-1][i]) for i in np.arange(len(level_ix))]

    for j in range(tree_depth-1, -1, -1):
        # Map the subsampled cells from the preceding level as parents/children
        parent_ii = parent_ix[j][level_ix]
        parent_dict = {}
        for i, ix in enumerate(parent_ii):
            parent = parent_dict.get(ix, CellGroupNode())
            parent.children.append(tips[i])
            tips[i].parent = parent
            parent_dict[ix] = parent

        # These are the new base layer, and we continue to build upwards
        level_ix = pd.unique(parent_ii)
        tips = [parent_dict[ix] for ix in level_ix]

    root = CellGroupNode()
    root.children = tips
    for child in tips:
        child.parent = root

    true_tree = root
    
    simulation.add_sampled_tree(true_tree)
    
    # Simulating a tree: we start out with a single cell which is unmutated
    cell_recorder = np.zeros((num_init_cells, num_sites))
    min_depth = np.nan
    record = [cell_recorder]

    available_sites = []
    edits_made = []
    edit_record = []
    
    # Not needed??
    edits_occurred = set([])
    convergence = {}

    for i in range(tree_depth):
        # One round of CRISPR may happen between this generation's division and the previous generation
        print('Generation: {0}'.format(i))
        cell_recorder, novel_edits, num_available_sites, num_edits_made = performCRISPR(cell_recorder, 
                                                                                      num_sites, 
                                                                                      mutation_probs,
                                                                                      deletions_probs,
                                                                                      )
        
        
        cell_recorder = cell_recorder[keep_ix[i]]
        novel_edits = novel_edits[keep_ix[i]]
        
        record.append(cell_recorder)
        edits_made.append(novel_edits)
    
    if missing_fraction > 0:
        # Randomly drop mutations to introduce missingness in data 
        prop = int(record[-1].size * missing_fraction)
        
        print('drop', prop, record[-1].shape)
        #Randomly choose indices of the numpy array:
        i = [np.random.choice(range(record[-1].shape[0])) for _ in range(prop)]
        j = [np.random.choice(range(record[-1].shape[1])) for _ in range(prop)]
        
        record[-1][i,j] = -1

    simulation.add_cell_record(record)
    simulation.add_edit_record(edits_made)
    
    simulation.subsample_time = time.time()-start 
    
    simulation.add_sampled_network()
                              
    # Plot the number of mutations that occurred per generation
    simulation.plot_edits_made()
    
    # Plot the character matrix of the final 
    simulation.plot_character_matrix()
    
    # Plot various dynamics of subsampled tree
    simulation.plot_subsampled_growth_curve()
    simulation.plot_unique_cells()
    
    # Plot the number of available sites per generation 
    simulation.plot_open_sites()
    
    final_cells = simulation.get_final_cells()
    
    subsampled_edits = edits_made
    
    # Investigating number of singletons, recurrent mutations and 'good' mutations 
    recurring_edits = {} # Mapping (site, deletion) to a list of generations where the mutation reoccured. 
    gen_occurred = {}
    for gen in range(tree_depth):
        rows, cols = np.where(subsampled_edits[gen]!=0)
        edits = subsampled_edits[gen][np.where(subsampled_edits[gen]!=0)]
        for col, edit in zip(cols, edits):
            # Track the generation that each mutation occurred
            gen_occurred[(col,edit)] = gen_occurred.get((col, edit), []) + [gen]
            # Track only recurrent mutations
            if ((record[gen - 1 ][:, col] == edit).sum()) > 0:
                # Add this (position, deletion) pair to the list of recurring muts
                recurring_edits[(col, edit)] = recurring_edits.get((col, edit), []) + [gen]
                
                
    plt.figure()
    plt.suptitle('Subsampled - Number of Occurrences for Repeated Mutations')
    total = 0
    for x in gen_occurred:
        if len(gen_occurred[x])==1:
            total+=1
    plt.title('Number of Unique Muts: {0}'.format(total))
    repeated = [len(x) for x in recurring_edits.values()]
    if len(repeated) > 0:
        plt.hist(repeated, bins=len(repeated))
        plt.ylabel('Count')
        plt.xlabel('Number of Occurrences')
        plt.show()
    else:
        print('No repeated mutations')
    
    if identify_recurrent_muts:
        try:
            # Plot the tests for identifying recurrent mutations 
            result = identify_recurrent(record, recurring_edits, gen_occurred)
            runtime, all_conflicting, conflict_matrix, truth_tape, prevalence_tape, first_gen_tape, num_recur_tape = result 

            simulation.add_conflict_matrix(conflict_matrix)
            simulation.add_truth_tape(truth_tape)
            simulation.add_prevalance_tape(prevalence_tape)
            simulation.add_first_gen_tape(first_gen_tape)
            simulation.add_num_recur_tape(num_recur_tape)
            simulation.add_conflicting_muts(all_conflicting)
        except:
            print('identify_recurrent failed')
        
    simulation.runtime = time.time()-start
    
    return simulation


def lineageSimulationFast(label, tree_depth, num_sites, deletions_probs, 
                          mutation_probs, edit_probs, compute_tree=False, 
                          init_cells=1, n_subsample=10000, missing_fraction=0):
    
    start = time.time()
    simulation = SimulationResult(label, init_cells, tree_depth, num_sites, edit_probs, run=None)
    
    reproduction_prob = 1.0 
    
    # Subsample 10,000 cells from the final expected number of cells 
    # (initial_cells x 2^tree_depth). Warning - this only works when we 
    # have uniform cell doubling! 

    # This tells us which cells we can throw away at each generation
    # We need to convert the subsampled index to the index given that some 
    # cells are thrown away. 

    parent_reix = dict(zip(np.arange(init_cells), np.arange(init_cells)))

    subsampled_ix = np.sort(np.random.choice(2**tree_depth, n_subsample, replace=False))
    
    simulation.add_subsampled_ix(subsampled_ix)
    
    original_ix, keep_ix = get_new_ix(subsampled_ix, parent_reix, tree_depth)
    
    # Build a subsampled tree
    from skbio import TreeNode

    level_ix = simulation.subsampled_ix
    
    
    # Create tips corresponding to each of the sampled cells
    tips = [TreeNode(str(i)) for i in np.arange(len(level_ix))]
    
    for j in enumerate(range(tree_depth-1, -1, -1)):
        # Map the subsampled cells from the preceding level as parents/children
        parent_ix = level_ix//2
        parent_dict = {}
        for i, ix in enumerate(parent_ix):
            parent = parent_dict.get(ix, TreeNode())
            parent.children.append(tips[i])
            tips[i].parent = parent
            parent_dict[ix] = parent

        # These are the new base layer, and we continue to build upwards
        level_ix = pd.unique(parent_ix)
        tips = [parent_dict[ix] for ix in level_ix]

    true_tree = tips[0]
    simulation.add_sampled_tree(true_tree)

    # Simulating a tree: we start out with a single cell which is unmutated
    cell_recorder = np.zeros((init_cells, num_sites))
    min_depth = np.nan
    record = [cell_recorder]

    available_sites = []
    edits_made = []
    edit_record = []
    
    # Not needed??
    edits_occurred = set([])
    convergence = {}
    
    keep_ix = keep_ix[1:]

    for i in range(tree_depth):
        # One round of CRISPR may happen between this generation's division and the previous generation
        print('Generation: {0}'.format(i))
        cell_recorder, novel_edits, num_available_sites, num_edits_made = performCRISPR(cell_recorder, 
                                                                                      num_sites, 
                                                                                      mutation_probs,
                                                                                      deletions_probs,
                                                                                      )
        
        
        cell_recorder = cell_recorder[keep_ix[i]]
        novel_edits = novel_edits[keep_ix[i]]
        
        record.append(cell_recorder)
        edits_made.append(novel_edits)
    
    if missing_fraction > 0:
        # Randomly drop mutations to introduce missingness in data 
        prop = int(cell_r[-1].size * missing_fraction)
        #Randomly choose indices of the numpy array:
        i = [random.choice(range(cell_r[-1].shape[0])) for _ in range(prop)]
        j = [random.choice(range(cell_r[-1].shape[1])) for _ in range(prop)]

        cell_r[-1][i,j] = np.NaN

    simulation.add_cell_record(record)
    simulation.add_edit_record(edits_made)
    
    simulation.subsample_time = time.time()-start 
    
    simulation.add_sampled_network()
                              
    # Plot the number of mutations that occurred per generation
    simulation.plot_edits_made()
    
    # Plot the character matrix of the final 
    simulation.plot_character_matrix()
    
    # Plot various dynamics of subsampled tree
    simulation.plot_subsampled_growth_curve()
    simulation.plot_unique_cells()
    
    # Plot the number of available sites per generation 
    simulation.plot_open_sites()
    final_cells = simulation.compute_final_cells()
    
    subsampled_edits = edits_made
    
    # Investigating number of singletons, recurrent mutations and 'good' mutations 
    recurring_edits = {} # Mapping (site, deletion) to a list of generations where the mutation reoccured. 
    gen_occurred = {}
    for gen in range(tree_depth):
        rows, cols = np.where(subsampled_edits[gen]!=0)
        edits = subsampled_edits[gen][np.where(subsampled_edits[gen]!=0)]
        for col, edit in zip(cols, edits):
            # Track the generation that each mutation occurred
            gen_occurred[(col,edit)] = gen_occurred.get((col, edit), []) + [gen]
            # Track only recurrent mutations
            if ((record[gen - 1 ][:, col] == edit).sum()) > 0:
                # Add this (position, deletion) pair to the list of recurring muts
                recurring_edits[(col, edit)] = recurring_edits.get((col, edit), []) + [gen]
                
                
    plt.figure()
    plt.suptitle('Subsampled - Number of Occurrences for Repeated Mutations')
    total = 0
    for x in gen_occurred:
        if len(gen_occurred[x])==1:
            total+=1
    plt.title('Number of Unique Muts: {0}'.format(total))
    repeated = [len(x) for x in recurring_edits.values()]
    if len(repeated) > 0:
        plt.hist(repeated, bins=len(repeated))
        plt.ylabel('Count')
        plt.xlabel('Number of Occurrences')
        plt.show()
    else:
        print('No repeated mutations')
    
    try:
        # Plot the tests for identifying recurrent mutations 
        result = identify_recurrent(record, recurring_edits, gen_occurred)
        runtime, all_conflicting, conflict_matrix, truth_tape, prevalence_tape, first_gen_tape, num_recur_tape = result 

        simulation.add_conflict_matrix(conflict_matrix)
        simulation.add_truth_tape(truth_tape)
        simulation.add_prevalance_tape(prevalence_tape)
        simulation.add_first_gen_tape(first_gen_tape)
        simulation.add_num_recur_tape(num_recur_tape)
        simulation.add_conflicting_muts(all_conflicting)
    except:
        print('identify_recurrent failed')
        
    simulation.runtime = time.time()-start
    
    
    return simulation


def lineageSimulation(label, tree_depth, num_sites, deletions_probs, mutation_probs, compute_tree=False, init_cells=1, n_subsample=10000):
    """
    Simulate a lineage tracing experiment using a specific guide with given edit probabilities and distributions.
    """
    sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5, 'figure.figsize':(20, 10)})
    
    start = time.time()
    simulation = SimulationResult(label, init_cells, tree_depth, num_sites, run=None)
    
    # TODO: Fix this reproduction probability
    reproduction_prob = 1.0 
    
    # Simulating a tree: we start out with a single cell which is unmutated
    cell_recorder = np.zeros((init_cells, num_sites))
    min_depth = np.nan
    record = [cell_recorder]

    available_sites = []
    edits_made = []
    edit_record = []
    
    # Not needed??
    edits_occurred = set([])
    convergence = {}
    
    for i in range(tree_depth):
        # One round of CRISPR may happen between this generation's division and the previous generation
        print('Generation: {0}'.format(i))
        cell_recorder, novel_edits, num_available_sites, num_edits_made = performCRISPR(cell_recorder, 
                                                                                      num_sites,
                                                                                      mutation_probs,
                                                                                      deletions_probs,
                                                                                      )

        available_sites.append(num_available_sites)
        edits_made.append(num_edits_made)
        
        record.append(cell_recorder)
        edit_record.append(novel_edits)
        
    simulation.add_full_cell_record(record)
    simulation.add_full_edit_record(edit_record)
    
    # Subsample cells 
    subsampled_ix = np.random.choice(cell_recorder.shape[0], n_subsample, replace=False)
    
    # Build a subsampled tree
    
    level_ix = subsampled_ix
    # Create tips corresponding to each of the sampled cells
    tips = [TreeNode(str(i)) for i in range(len(level_ix))]

    for j in enumerate(range(len(record)-1, -1, -1)):
        # Map the subsampled cells from the preceding level as parents/children
        parent_ix = level_ix//2
        parent_dict = {}
        for i, ix in enumerate(parent_ix):
            parent = parent_dict.get(ix, TreeNode())
            parent.children.append(tips[i])
            tips[i].parent = parent
            parent_dict[ix] = parent

        # These are the new base layer, and we continue to build upwards
        level_ix = pd.unique(parent_ix)
        tips = [parent_dict[ix] for ix in level_ix]

    true_tree = tips[0]
    simulation.add_sampled_tree(true_tree)

    # We only care about the ancestors of these subsampled cells
    subsampled_record = []
    level_ix = subsampled_ix
    for i, ix in enumerate(range(len(record)-1, -1, -1)):
        rec = record[ix]
        subsampled_record.append(rec[level_ix, :])
        level_ix = pd.unique(level_ix//2)
    subsampled_record.reverse()
    
    
    simulation.add_subsampled_ix(subsampled_ix)
    simulation.add_cell_record(subsampled_record)
    
    # We also only care about edits related to these subsampled cells
    # The edit at index i corresponds to the parents of the cells at 2i, 2i+1 in the level below
    level_ix = subsampled_ix
    subsampled_edits = []
    for rec in reversed(edit_record):
        level_ix = pd.unique(level_ix//2)
        subsampled_edits.append(rec[level_ix, :])
    subsampled_edits.reverse()
    
    simulation.add_edit_record(subsampled_edits)
    
    simulation.subsample_time = time.time()-start 
    
    # Plot the number of mutations that occurred per generation
    simulation.plot_edits_made()
    
    # Plot the character matrix of the final 
    simulation.plot_character_matrix()
    
    # Plot various dynamics of subsampled tree
    simulation.plot_subsampled_growth_curve()
    simulation.plot_unique_cells()
    
    # Plot the number of available sites per generation 
    simulation.plot_open_sites()
    final_cells = simulation.compute_final_cells()
    
    
    # Investigating number of singletons, recurrent mutations and 'good' mutations 
    recurring_edits = {} # Mapping (site, deletion) to a list of generations where the mutation reoccured. 
    gen_occurred = {}
    for gen in range(tree_depth):
        rows, cols = np.where(subsampled_edits[gen]!=0)
        edits = subsampled_edits[gen][np.where(subsampled_edits[gen]!=0)]
        for col, edit in zip(cols, edits):
            # Track the generation that each mutation occurred
            gen_occurred[(col,edit)] = gen_occurred.get((col, edit), []) + [gen]
            # Track only recurrent mutations
            if ((subsampled_record[gen - 1 ][:, col] == edit).sum()) > 0:
                # Add this (position, deletion) pair to the list of recurring muts
                recurring_edits[(col, edit)] = recurring_edits.get((col, edit), []) + [gen]
    simulation.recurring_edits = recurring_edits            
                
    plt.figure()
    plt.suptitle('Subsampled - Number of Occurrences for Repeated Mutations')
    total = 0
    for x in gen_occurred:
        if len(gen_occurred[x])==1:
            total+=1
    plt.title('Number of Unique Muts: {0}'.format(total))
    repeated = [len(x) for x in recurring_edits.values()]
    plt.hist(repeated, bins=len(repeated))
    plt.ylabel('Count')
    plt.xlabel('Number of Occurrences')
    plt.show()
    
    # Plot some dynamics of full simulation

    simulation.add_full_open_sites(available_sites)
    simulation.plot_full_open_sites()
    
    simulation.add_full_edits_made(edits_made)
    simulation.plot_full_edits_made()
    
    
    if compute_tree:
        final_cells = copy.deepcopy(pd.DataFrame(subsampled_record[-1]))
        final_cells['Label'] = '1'
        final_cells_labeled = split_tree(final_cells)
        final_cells = copy.deepcopy(pd.DataFrame(subsampled_record[-1]))
        final_cells['Label'] = '1'
        compute_triplets_correct(final_cells_labeled, subsampled_ix, sample_size = 1000)
    
    try:
        # Plot the tests for identifying recurrent mutations 
        result = identify_recurrent(subsampled_record, recurring_edits, gen_occurred)
        runtime, all_conflicting, conflict_matrix, truth_tape, prevalence_tape, first_gen_tape, num_recur_tape = result 

        simulation.add_conflict_matrix(conflict_matrix)
        simulation.add_truth_tape(truth_tape)
        simulation.add_prevalance_tape(prevalence_tape)
        simulation.add_first_gen_tape(first_gen_tape)
        simulation.add_num_recur_tape(num_recur_tape)
        simulation.add_conflicting_muts(all_conflicting)
    except:
        print('identify_recurrent failed')
        
    simulation.runtime = time.time()-start

    return simulation 


