# Store simulation result

import numpy as np
import pandas as pd
import torch
import os
import time 
import copy
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set(font_scale=1)

import networkx as nx
from cassiopeia.TreeSolver.Node import Node
        
def load_simulation(path):
    simulation = SimulationResult

class SimulationResult():
    def __init__(self, label, init_cells, tree_depth, num_sites, edit_probs, run=None, missing_fraction=0):
        self.label = label
        self.init_cells = init_cells
        self.run = run 
        self.missing_fraction = missing_fraction
        
        self.tree_depth = tree_depth
        self.num_sites = num_sites
        
        self.edit_probs = edit_probs 
        
        self._edit_prob_df = None
        
        self.unique_cells = None
        self.open_sites = None 
        
        self.feature_matrix = None
        self.random_attr = {}
       
    def get_edit_probs(self):
        def dels_to_df(pad_dels):
            df = pd.DataFrame(pad_dels.reshape(-1)).T
            # Compute correct column names to index mutations 
            t = np.tile(np.arange(pad_dels.shape[1]), pad_dels.shape[0]).reshape(pad_dels.shape[0], -1)

            tt = t+1000*np.arange(t.shape[0]).reshape(-1,1)

            df.columns = tt.reshape(-1)
            return df
        
        if self._edit_prob_df is None:
            self._edit_prob_df = dels_to_df(self.edit_probs)
        return self._edit_prob_df
    
    
        
    def add_full_cell_record(self, cr):
        self.full_cell_record = cr
       
    def get_full_cell_record(self):
        return copy.deepcopy(self.full_cell_record)
    
    def add_sampled_tree(self, true_tree):
        self.true_tree = true_tree
       
    def get_sampled_tree(self):
        return copy.deepcopy(self.true_tree)
    
    def add_sampled_network(self):
        
        def format_char_vec(a):
            nan_a = np.isnan(a)
            a = a.astype(np.int).astype(str)
            a[nan_a] = '-'
            return list(a)

        # Create networkx DiGraph to represent true_tree 
        tree = nx.DiGraph()

        cell_record = self.get_cell_record()
        keep_labels = self.get_node_labels()
        parent_ix_levels = self.get_parent_child_map()

        # Create nodes representing the leaves

        record = cell_record[0]
        prev_level = [Node(label, format_char_vec(record[i])) for i, label in enumerate(keep_labels[0])]

        for level, mapping in enumerate(parent_ix_levels):
            # Construct edges from this level to the next level
            level_labels = keep_labels[level+1]
            record = cell_record[level+1]
            current_level = []
            for child in mapping:
                # Create a child node
                child_node = Node(level_labels[child], format_char_vec(record[child]))
                parent = prev_level[mapping[child]]
                tree.add_edges_from([(parent, child_node)])
                current_level.append(child_node)
            # Current level finished adding to tree, move on to lower level
            prev_level = current_level

        self.true_network = tree

    def get_sampled_network(self):
        return copy.deepcopy(self.true_network)
        
    def add_full_edit_record(self, er):
        self.full_edit_record = er
        
    def add_full_open_sites(self, ops):
        self.full_open_sites = ops
        
    def plot_full_open_sites(self, save_as=None, display_plot=True):
        plt.figure()
        plt.title('Available Sites in Full Simulation')
        plt.plot(self.full_open_sites)
        plt.xlabel('Generation')
        plt.ylabel('Number of Sites')
        if save_as is not None:
            plt.savefig(save_as)
        if display_plot:
            plt.show()
        plt.close()
        
    def add_full_edits_made(self, em):
        self.full_edits_made = em
    
    def get_full_edits_made(self):
        return copy.deepcopy(self.full_edits_made)
    
    def plot_full_edits_made(self, save_as=None, display_plot=True):
        plt.figure()
        plt.suptitle('Edits Made per Generation in Full Simulation')
        plt.title('Total Edits: {0}'.format(sum(self.full_edits_made)))
        plt.plot(self.full_edits_made)
        plt.xlabel('Generation')
        plt.ylabel('Edits Made')
        if save_as is not None:
            plt.savefig(save_as)
        if display_plot:
            plt.show()
        plt.close()
        
    def add_conflict_matrix(self, cm):
        """

        """
        self.conflict_matrix = cm 

    def get_conflict_matrix(self):
        """

        """
        return copy.deepcopy(self.conflict_matrix)

    def add_truth_tape(self, tt):
        """

        """
        self.truth_tape = tt

    def get_truth_tape(self):
        """

        """
        return copy.deepcopy(self.truth_tape)
        
    def add_prevalance_tape(self, pt):
        """

        """
        self.prevalence_tape = pt

    def get_prevalance_tape(self):
        """

        """
        return copy.deepcopy(self.prevalence_tape)

    def add_first_gen_tape(self, fgt):
        """

        """
        self.first_gen_tape = fgt

    def get_first_gen_tape(self):
        """

        """
        return copy.deepcopy(self.first_gen_tape)
    
    def add_num_recur_tape(self, nrt):
        """

        """
        self.num_recur_tape = nrt

    def get_num_recur_tape(self):
        """

        """
        return copy.deepcopy(self.num_recur_tape)

    def add_conflicting_muts(self, cm):
        """

        """
        self.conflicting_muts = cm

    def get_conflicting_muts(self):
        """

        """
        return copy.deepcopy(self.conflicting_muts)

    def save(self, path):
        
        self.full_cell_record.to_csv()
        self.full_edit_record.to_csv()
        
        self.full_open_sites
        self.true_tree
        self.conflict_matrix
        
        self.plot_full_edits_made(save_as=None)
        
        self.truth_tape
        self.prevalence_tape
       
        self.first_gen_tape
        self.num_recur_tape
        
        self.conflicting_muts
        
        fc = self.get_final_cells()
        fc.to_csv() 
        
        fm = self.get_feature_matrix()
        fm.to_csv()
        
    def get_final_cells(self):
        return pd.copy(self.final_cells.astype(int), deep=True)
    
    def get_feature_matrix(self):
        """ 
        Convert the final cells matrix to a character matrix
        Convert each unique (col, non-zero entry) into a character
        """
        
        if self.feature_matrix is None:
            f = self.get_final_cells()
            
            from utilities import binarize_character_matrix, character_matrix_to_labels
            
            str_labels = character_matrix_to_labels(f)
            self.cell_id_to_char_vec = dict(zip(f.index, str_labels))
            
            inv_map = {}
            for k, v in self.cell_id_to_char_vec.items():
                inv_map[v] = inv_map.get(v, []) + [k]
            self.char_vec_to_cell_id = inv_map
            
            f.index = str_labels
            
            
            self.feature_matrix = binarize_character_matrix(f)
            
        return copy.deepcopy(self.feature_matrix)



    def add_subsampled_ix(self, ix):
        self.subsampled_ix = ix
        
    def add_node_labels(self, labels):
        self.node_labels = labels
       
    def get_node_labels(self):
        return copy.deepcopy(self.node_labels)
        
    def add_parent_child_map(self, parent_child_map):
        self.parent_child_map = parent_child_map
        
    def get_parent_child_map(self):
        '''
        Stores a list where the ith element maps the indices of children on level i+1 to the indices of their parents 
        on level i
        
        e.g [{0:0,1:0}, {0:0, 1:1, 2:1}] would be the parent/child map for the tree:
           __0_____0
       0 _/      __1
          \__1__/
                \__2
        '''
        try:
            return copy.deepcopy(self.parent_child_map)
        except:
            raise ValueError('No parent/child map stored') 
        
    def add_cell_record(self, cr):
        """
        Stores a subsampled cell record as a list of arrays, where the array at index g corresponds to generation g.
        arr[g] (num_sites x num_cells) contains the record of the mutation state of each cell in generation g.
        @param: cr - cell record, where cr[i] has shape (num_sites x num_cells) 
        """
        self.cell_record = cr 
        
    def get_cell_record(self):
        """
        Returns a copy of the subsampled cell record.
        """
        return copy.deepcopy(self.cell_record)
    
    def add_edit_record(self, er):
        """
        Stores a subsampled edit record as alist of arrays, where the array at index g corresponds to generation g.
        arr[g] contains the record of which edits (indel, site paires) were added to which cells in generation g.
        @param: er - edit record, where er[i] has shape (num_sites x num_cells) 
        """
        self.edit_record = er 
        
    def get_edit_record(self):
        """
        Returns a copy of the subsampled edit record.
        """
        return copy.deepcopy(self.edit_record)
    
    def plot_subsampled_growth_curve(self, save_as=None, display_plot=True):
        plt.figure()
        r_shapes = [r.shape[0] for r in self.cell_record]
        plt.plot(r_shapes)
        plt.suptitle('Subsampled - Number of Ancestors per Generation')
        plt.ylabel('Number of Cells')
        plt.xlabel('Generation')
        if save_as is not None:
            plt.savefig(save_as)
        if display_plot:
            plt.show()
        plt.close()
    
    def plot_character_matrix(self, save_as=None, display_plot=True):
        plt.figure()
        sns.heatmap(self.cell_record[-1]!=0, cmap='Paired')
        plt.suptitle('Subsampled - Character Matrix')
        if save_as is not None:
            plt.savefig(save_as)
        if display_plot:
            plt.show()
        plt.close()
        
    def compute_unique_cells(self):
        if self.unique_cells is None:
            unique_cell_per_gen = []
            for gen in range(self.tree_depth):
                final_cells = pd.DataFrame(self.cell_record[gen])
                unique_cells = final_cells.drop_duplicates().shape[0]/final_cells.shape[0]
                unique_cell_per_gen.append(unique_cells)
                
            self.unique_cells = unique_cell_per_gen
            self.final_cells = final_cells
            
        return self.unique_cells
    
    def plot_unique_cells(self, save_as=None, display_plot=True):
        unique_cell_per_gen = self.compute_unique_cells()
        plt.figure()
        plt.plot(unique_cell_per_gen)
        plt.xlabel('Generation')
        plt.ylabel('Unique Cells/ Total Cells')
        plt.title('Subsampled - Proportion Cells Unique')
        if save_as is not None:
            plt.savefig(save_as)
        if display_plot:
            plt.show()
        plt.close()
    
    def plot_edits_made(self, save_as=None, display_plot=True):
        
        plt.figure()
        plt.plot(np.arange(self.tree_depth), [x.sum() for x in self.edit_record])
        plt.xlabel('Generation')
        plt.ylabel('Number of Mutations')
        plt.suptitle('Subsampled - Number of mutations per generation')
        if save_as is not None:
            plt.savefig(save_as)
        if display_plot:
            plt.show()
        plt.close()
        
        
    def compute_final_cells(self):
        final_cells = pd.DataFrame(self.cell_record[-1])
        self.final_cells = final_cells.astype(int)
        
        self.final_cells = self.final_cells.drop_duplicates()
        
        return self.final_cells.copy(deep=True)
    
    def get_final_cells(self):
        return copy.deepcopy(self.final_cells)

    def get_open_sites(self):
        if self.open_sites is None:
            open_sites = []
            for rec in self.cell_record:
                not_edited = rec == 0
                open_sites.append(not_edited.sum())
            self.open_sites = open_sites
        return self.open_sites
        
    def plot_open_sites(self, save_as=None, display_plot=True):
        plt.figure()
        plt.suptitle('Available Sites in Sampled Simulation')
        plt.plot(self.get_open_sites())
        plt.xlabel('Generation')
        plt.ylabel('Number of Sites')
        if save_as is not None:
            plt.savefig(save_as)
        if display_plot:
            plt.show()
        plt.close()