import pandas as pd
import numpy as np

df_distance_matrix = pd.read_csv("./santa-2021/distance_matrix.csv")
df_permutations = pd.read_csv("./santa-2021/permutations.csv")
df_sample_submission = pd.read_csv("./santa-2021/sample_submission.csv")
df_wildcards = pd.read_csv("./santa-2021/wildcards.csv")

df_with_mrms_start = df_permutations[df_permutations['Permutation'].str.startswith('🎅🤶')]
df_without_mrms_start = df_permutations[np.logical_not(df_permutations['Permutation'].str.startswith('🎅🤶'))]