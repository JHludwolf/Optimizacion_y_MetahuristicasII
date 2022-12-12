import numpy as np

from Dataframes import df_with_mrms_start, df_without_mrms_start, df_permutations

def count_perms_in_substring(substrings, permutations):
    aux_perm = [perm in substrings[0] or perm in substrings[1] or perm in substrings[2] for perm in permutations]
    return np.array(aux_perm)

def count_mrms_in_substring(substrings, permutations):
    aux_perm = [perm in substrings[0] and perm in substrings[1] and perm in substrings[2] for perm in permutations]
    return np.array(aux_perm)

def general_objective_func(df_substrings):
        
    symbols = ['🎅', '🤶', '🦌', '🧝', '🎄', '🎁', '🎀']

    # BOOL ARR OF MRMS IN EVERY WILDCARD VARIATION
    mrms_in_wildcards = np.zeros(df_with_mrms_start.shape[0])
    mrms_in_wildcards.fill(False)

    # BOOL ARR OF PERMS IN EVERY WILDCARD VARIATION
    perm_in_wildcards = np.zeros(df_without_mrms_start.shape[0])
    perm_in_wildcards.fill(False)

    all_wildcards = np.array([[s.replace('🌟', symbol) for s in df_substrings['schedule']] for symbol in symbols])
    wildcard_subs = [''.join(all_wildcards[:,i]) for i in range(3)]

    # CHECK PERMUTATIONS THAT START WITH MR &. MRS CLAUSE
    mrms_in_sub = count_mrms_in_substring(wildcard_subs, df_with_mrms_start['Permutation'])
    mrms_in_wildcards = np.logical_or(mrms_in_wildcards, mrms_in_sub)
        
    # CHECK PERMUTATIONS THAT DON'T START WITH MR &. MRS CLAUSE
    perms_in_sub = count_perms_in_substring(wildcard_subs, df_without_mrms_start['Permutation'])
    perm_in_wildcards = np.logical_or(perm_in_wildcards, perms_in_sub)


    max_lenght = max([len(substring) for substring in df_substrings['schedule']])
    score = df_permutations.shape[0] - (np.sum(mrms_in_wildcards) + np.sum(perm_in_wildcards))

    return True if score == 0 else False # CHECK THAT THERE ARE NO REPEATED VALUES

def delete_nodes_objective_func(df_substrings):
    # BOOL ARR OF MRMS IN EVERY WILDCARD VARIATION
    mrms_in_wildcards = np.zeros(df_with_mrms_start.shape[0])
    mrms_in_wildcards.fill(False)

    # BOOL ARR OF PERMS IN EVERY WILDCARD VARIATION
    perm_in_wildcards = np.zeros(df_without_mrms_start.shape[0])
    perm_in_wildcards.fill(False)

    # CHECK PERMUTATIONS THAT START WITH MR &. MRS CLAUSE
    mrms_in_sub = count_mrms_in_substring(df_substrings['schedule'], df_with_mrms_start['Permutation'])
    mrms_in_wildcards = np.logical_or(mrms_in_wildcards, mrms_in_sub)

    # CHECK PERMUTATIONS THAT DON'T START WITH MR &. MRS CLAUSE
    perms_in_sub = count_perms_in_substring(df_substrings['schedule'], df_without_mrms_start['Permutation'])
    perm_in_wildcards = np.logical_or(perm_in_wildcards, perms_in_sub)

    score = df_permutations.shape[0] - (np.sum(mrms_in_wildcards) + np.sum(perm_in_wildcards))

    return True if score == 0 else False # CHECK THAT THERE ARE NO REPEATED VALUES

def count_nodes_objective_func(df_substrings):

    symbols = ['🎅', '🤶', '🦌', '🧝', '🎄', '🎁', '🎀']
    c=0

    for symbol in symbols:
        
        wildcard_subs = [s.replace('🌟', symbol) for s in df_substrings['schedule']]
        
        for perm in df_permutations['Permutation']:
            for substring in wildcard_subs:
                c += substring.count(perm)
    return c
