import pandas as pd
import numpy as np 
import time

import matplotlib.pyplot as plt
import seaborn as sns

# Identify mutations that are recurrent according to the four gamete test 

def identify_recurrent(subsampled_record, recurring_edits, gen_occurred):
    # Test every pair of mutations and identify those which are in conflict
    final_cells = pd.DataFrame(subsampled_record[-1])
    print(final_cells.shape)
    x = final_cells.values

    import time
    t = time.time()
    # For each target site, access all the mutations
    muts = {}
    for col in range(x.shape[1]):
        unique_muts  = np.unique(x[:, col][x[:, col]!=0]) 

        m_list = []
        for m in unique_muts:
            count = (subsampled_record[-1][:, col] == m).sum(0)
            if count >= 2:
                m_list.append(m)
        muts[col] = m_list


    print('Time 1:', time.time()-t)
    # Iterate over all pairs of mutations and determine whether they are in conflict)
    conflicts_with = {}

    sites = list(muts.keys())
    n_sites = len(sites)

    t1 = time.time()
    # See 4-gamete test: A pair of mutations are in conflict if we observe A with B, A without B and B without A.
    for i in range(n_sites):
        for m1 in muts[i]:
            yes_cells = final_cells[final_cells[i] == m1]
            no_cells = final_cells[final_cells[i] != m1]

            for j in range(i+1, n_sites):

                x = set(yes_cells[j].drop_duplicates().values)
                y = set(no_cells[j].drop_duplicates().values)

                for m2 in muts[j]:
                    # Find all the cells that contain both A and B. If none exist, we move on.
                    if m2 not in x:
                        continue
                    # Find all the cells that contain only A. If none exist, we move on.
                    if len(x)==1:
                        continue
                    # Find all the cells that contain only B. If none exist, we move on.
                    if m2 not in y:
                        continue

                    A = (i,m1)
                    B = (j,m2)

                    # Mutations A and B are in conflict
                    conflicts_with[A] = conflicts_with.get(A, []) + [B]                
                    conflicts_with[B] = conflicts_with.get(B, []) + [A]

    print('Compute dictionary:', time.time()-t)          
    
    all_conflicting = list(conflicts_with.keys())
    num_conflicting = [len(conflicts_with[a]) for a in all_conflicting]

    mut_to_ix = {}
    ix_to_mut = {}
    i = 0
    for mut in all_conflicting:
        mut_to_ix[mut] = i
        ix_to_mut[i] = mut
        i += 1
    total_muts = i

    conflict_matrix = np.zeros((total_muts, total_muts))

    for m1 in conflicts_with:
        for m2 in conflicts_with[m1]:
            conflict_matrix[mut_to_ix[m1], mut_to_ix[m2]] = 1

    def check_symmetric(a, tol=1e-8):
        return np.all(np.abs(a-a.T) < tol)

    print('Conflict matrix is symmetric? ', check_symmetric(conflict_matrix))
    runtime = time.time()-t
    print('Total time:', runtime) 

    true = set(recurring_edits.keys())
    inferred = set(conflicts_with.keys())
    print('Number of true recurrent muts: ', len(true))
    print('Number of detected recurrent muts: ', len(inferred))
    undetected = true - true.intersection(inferred)
    prevalence = []
    for m in undetected:
        prevalence.append(sum(subsampled_record[-1][:, m[0]]==m[1]))
    print('Number of true recurrent muts not detected:', len(undetected), 'with prevalence', prevalence)

    # For each identified conflicting mutation, get the number of times it recurred
    num_recur_tape = np.zeros(total_muts)
    for mut in all_conflicting:
        num_recur_tape[mut_to_ix[mut]] = len(gen_occurred[mut])

    # For each identified conflicting mutation, get the generation the mutation first existed
    first_gen_tape = np.zeros(total_muts)
    for mut in all_conflicting:
        first_gen_tape[mut_to_ix[mut]] = gen_occurred[mut][0]

    # For each identified conflicting mutation, what is the prevalence in the final generation?
    prevalence_tape = np.zeros(total_muts)
    for mut in all_conflicting:
        prevalence_tape[mut_to_ix[mut]] = sum(subsampled_record[-1][:, mut[0]]==mut[1])

    # Identify which conflicting muts are true 
    truth = []
    for mut in all_conflicting:
        truth.append(int(mut in recurring_edits))

    sns.set(rc={'axes.facecolor':'gainsboro'})

    f, axes = plt.subplots(8, 1, sharex=True, gridspec_kw={'height_ratios': [5, 0.5, 1, 1, 1, 1, 1,1 ]}, figsize=(10,30))
    sns.heatmap(conflict_matrix, cmap='Paired', ax=axes[0], cbar_kws = dict(use_gridspec=False,location="top"))
    axes[0].set_title('Conflict Matrix')
    sns.heatmap([truth], cmap='Paired', ax=axes[1], cbar=False)
    axes[1].set_title('True Mutations')
    sns.heatmap([conflict_matrix.sum(0)], cmap=sns.color_palette("Blues"), ax=axes[2], cbar_kws = dict(use_gridspec=False,location="bottom"))
    axes[2].set_title('Number of Co-conflicting Mutations')

    axes[3].plot(num_recur_tape)
    axes[3].set_title('Number of Recurrences')

    axes[4].plot(first_gen_tape)
    axes[4].set_title('Generation of First Occurrence')

    axes[5].plot(prevalence_tape)
    axes[5].set_title('Prevalence in Final Generation')

    axes[6].plot(conflict_matrix.sum(0))
    axes[6].set_title('Number of Co-conflicts')
    axes[7].plot(truth, '*')
    axes[7].set_title('Truth')

    axes[7].set_xticklabels(axes[4].get_xticklabels(), rotation=75)
    plt.show()
    plt.close()



    df = {'truth':truth, 'truth_label': ['true_recurrent' if i else 'false_recurrent' for i in truth ], 'co-conflicts':conflict_matrix.sum(0), 
          'prevalence': prevalence_tape, 'first_occur': first_gen_tape, 'num_recur': num_recur_tape}
    df = pd.DataFrame(df)

    df['weighted_conflicts'] = (conflict_matrix * conflict_matrix.sum(0)).sum(0)
    pd.melt(df)

    g = sns.PairGrid(df, hue = 'truth_label', palette="Set2")
    g.map_diag(plt.hist)
    g.map_offdiag(plt.scatter)
    g.add_legend()
    g.fig.suptitle("Relationship Between Metrics", y=1) # y= some height>1
    plt.show()
    plt.close()

    return runtime, all_conflicting, conflict_matrix, truth, prevalence_tape, first_gen_tape, num_recur_tape  
    