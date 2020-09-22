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

from skbio import TreeNode
from SimulationResult import SimulationResult
from crispr_lib import lineageSimulation

from identify_recurrent import identify_recurrent 

## How many cells does the real data end up with?
## Did this seed from one cell? 

import pickle
with open('d21_indel_distributions.p', 'rb') as f:
    indel_distributions = pickle.load(f)

plot_distributions = False 

slow_guides = ['AGCTGCTTAGGGCGCAGCCT', 'CTCCTTGCGTTGACCCGCTT', 'TATTGCCTCTTAATCGTCTT']
medium_guides = ['AATCCCTAGTAGATTAGCCT', 'CACAGAACTTTATGACGATA', 'TTAAGTTTGAGCTCGCGCAA']
fast_guides = ['TAATTCCGGACGAAATCTTG', 'CTTCCACGGCTCTAGTACAT', 'CCTCCCGTAGTGTTGAGTCA']

for li in [slow_guides, medium_guides, fast_guides]:
    for guide in li:
        dist = indel_distributions[guide]
        try:
            del dist['']
        except KeyError:
            pass
        distvals = np.array(sorted(list(dist.values()), reverse=True))
        distvals = distvals/distvals.sum()

        if plot_distributions:
	        plt.figure()
	        sns.barplot(np.arange(len(distvals)), distvals, linewidth=0)
	        plt.title('Density over different edits for guide {0}'.format(guide))
	        plt.show()
	        plt.close()

# Estimate the mutation probability - refer to notebook

# For a given site E_i[%] = E_{i-1}[%] + p(1-E_{i-1}[%])
# E_i = 1-q^i, where q = 1-p 
# Sanity check: 
# E_0 = 1-(1-p)^0 = 0 
# E_1 = 1-(1-p) = p 

# We can choose to use a combination of varying speed guides 

# 4, 7, 14, 21 days -> 4 generations, 7 generations etc (24 hrs for cell cycle?)

# How many generations does 1 day correspond to?
# Slow guides:
slow = [0.09, 0.15, 0.29, 0.4]

# Medium guides
medium = [0.15, 0.35, 0.6, 0.75]

# Fast guides
fast = [0.75, 0.9, 0.95, 0.96]


rate = {}
rate['slow'] = 0.02361
rate['medium'] = 0.05668
rate['fast'] = 0.2269


slow_guides = ['AGCTGCTTAGGGCGCAGCCT', 'CTCCTTGCGTTGACCCGCTT', 'TATTGCCTCTTAATCGTCTT']
medium_guides = ['AATCCCTAGTAGATTAGCCT', 'CACAGAACTTTATGACGATA', 'TTAAGTTTGAGCTCGCGCAA']
fast_guides = ['TAATTCCGGACGAAATCTTG', 'CTTCCACGGCTCTAGTACAT', 'CCTCCCGTAGTGTTGAGTCA']

ssm = ['AGCTGCTTAGGGCGCAGCCT', 'CTCCTTGCGTTGACCCGCTT','AATCCCTAGTAGATTAGCCT']
smm = ['AGCTGCTTAGGGCGCAGCCT', 'AATCCCTAGTAGATTAGCCT', 'CACAGAACTTTATGACGATA']
ssf = ['AGCTGCTTAGGGCGCAGCCT', 'CTCCTTGCGTTGACCCGCTT', 'CCTCCCGTAGTGTTGAGTCA']
smf = ['AGCTGCTTAGGGCGCAGCCT', 'AATCCCTAGTAGATTAGCCT', 'CCTCCCGTAGTGTTGAGTCA']
sff = ['AGCTGCTTAGGGCGCAGCCT', 'CTTCCACGGCTCTAGTACAT', 'CCTCCCGTAGTGTTGAGTCA']
mmf = ['AATCCCTAGTAGATTAGCCT', 'CACAGAACTTTATGACGATA', 'TAATTCCGGACGAAATCTTG']
mff = ['AATCCCTAGTAGATTAGCCT', 'TAATTCCGGACGAAATCTTG', 'CTTCCACGGCTCTAGTACAT']


## Run simulation 

lists_of_guides = [smf]
labels = ['smf']

verbose = False
tree_depth = 15
num_runs = 1
n_subsample = 10000
num_arrays = 10

simulation_list = []
for i, list_of_guides in enumerate(lists_of_guides):
    label = labels[i]
    print('Label:', label)

    # Each array has 3-6 targets, we insert ~10 arrays. This gives us 30-60 sites
    site_ix = 0
    target_distributions = {}
    speed = {}
    for guide in list_of_guides:
        dist = indel_distributions[guide]
        try:
            del dist['']
        except KeyError:
            pass
        distvals = np.array(sorted(list(dist.values()), reverse=True))
        distvals = distvals/distvals.sum()
        target_distributions[site_ix] = distvals
        if guide in slow_guides:
            speed[site_ix] = 'slow'
        elif guide in medium_guides:
            speed[site_ix] = 'medium'
        else:
            speed[site_ix] = 'fast'

        site_ix += 1

    num_targets = site_ix 
    num_edit_sites = num_targets * num_arrays

    import itertools
    targets = list(itertools.product(np.arange(num_arrays), np.arange(num_targets)))
    print('List of targets: ', targets)

    deletions = []
    mutation_probs = []
    for array, target in targets:
        deletions.append(target_distributions[target])
        mutation_probs.append(rate[speed[target]])
    
    deletions_probs = pd.DataFrame(deletions)
    deletions_probs = torch.DoubleTensor(deletions_probs.fillna(0.0).values)
    
    # Each edit site has a different mutation probability 
    mutation_probs= np.array(mutation_probs)


    for run in range(num_runs):
        simulation = lineageSimulation(label, tree_depth, num_edit_sites, deletions_probs, mutation_probs, init_cells=1000)
           
        # Summarize results
        # Plot a regression for the non-recurrent mutations
        X = simulation.conflict_matrix.sum(0)
        Y = simulation.prevalence_tape
        ix = np.array(simulation.truth_tape)==0

#         from sklearn.linear_model import LinearRegression
#         reg = LinearRegression().fit(X[ix].reshape(-1,1), Y[ix].reshape(-1,1))
        
        x = X[ix]
        y = Y[ix]
        
        new_x = np.linspace(min(x), max(x), num=np.size(x))
        coefs = np.polyfit(x,y,2)
        new_line = np.polyval(coefs, new_x)
            
        plt.figure()
        sns.scatterplot(X, Y, hue=simulation.truth_tape, 
                        cmap='Paired', edgecolor=None,)

        plt.scatter(new_x,new_line,c='g', marker='^', s=1)
        plt.xlabel('Co-conflicts')
        plt.ylabel('Prevalence')
        plt.show()
        plt.close()
        
        simulation.random_attr['coef'] = coefs
        
        
        simulation.random_attr['num_conflict'] = simulation.conflict_matrix.shape[0]
        
        x = X
        y = Y
        
        new_x = np.linspace(min(x), max(x), num=np.size(x))
        coefs = np.polyfit(x,y,2)
        new_line = np.polyval(coefs, new_x)
        
        plt.figure()
        sns.scatterplot(X, Y, hue=simulation.truth_tape, 
                        cmap='Paired', edgecolor=None,)

        plt.scatter(new_x,new_line,c='g', marker='^', s=1)
        plt.xlabel('Co-conflicts')
        plt.ylabel('Prevalence')
        plt.show()
        plt.close()
        
        simulation.random_attr['full_reg_coef'] = coefs
        
        simulation_list.append(simulation)

        df = pd.DataFrame(subsampled_results)

