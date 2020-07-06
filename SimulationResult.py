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


class SimulationResult():
    def __init__(self, label, tree_depth, num_sites, run=None):
        self.label = label
        self.run = run 
        
        self.tree_depth = tree_depth
        self.num_sites = num_sites
        
        self.unique_cells = None
        self.open_sites = None 
        
        self.random_attr = {}
        
    def add_full_cell_record(self, cr):
        self.full_cell_record = cr
        
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

    def get_first_gen_tape(self):
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
        
    def add_final_cells(self, final_cells):
        self.final_cells = final_cells

    def get_final_cells(self):
        return copy.deepcopy(self.final_cells)
    
    def add_subsampled_ix(self, ix):
        self.subsampled_ix = ix
        
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
        if self.final_cells is None:
            final_cells = pd.DataFrame(self.cell_record[-1])
            self.final_cells = final_cells
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