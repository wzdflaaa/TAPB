import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('../datasets/bindingdb/random/train_with_id.csv')

def calculate_tendency(df, column):
    tendencies = df.groupby(column)['Y'].mean().round(1)
    return tendencies

smiles_tendencies = calculate_tendency(df, 'SMILES')
protein_tendencies = calculate_tendency(df, 'Protein')

smiles_frequency = smiles_tendencies.value_counts(normalize=True).sort_index()
protein_frequency = protein_tendencies.value_counts(normalize=True).sort_index()

all_tendencies = [round(x, 1) for x in np.arange(0.0, 1.1, 0.1)]

protein_values = protein_frequency.reindex(all_tendencies, fill_value=0).values
smiles_values = smiles_frequency.reindex(all_tendencies, fill_value=0).values

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 18

plt.figure(figsize=(10, 6))

x = np.arange(len(all_tendencies))
width = 0.35

# Target: left, Drug: Right
plt.bar(x - width/2, smiles_values, width, color='#82AAE8', label='Drug', edgecolor='#1A3092', linewidth=1.5 )   # Drug (SMILES)
plt.bar(x + width/2, protein_values, width, color='#E36B8F', label='Target', edgecolor='#9B123C', linewidth=1.5)  # Target (Protein)

plt.xlabel('Prior Tendency $z_i$')
plt.ylabel('Frequency')
plt.legend()

plt.xticks(x, all_tendencies)
plt.xlim(-0.5, len(all_tendencies)-0.5)
plt.ylim(0, max(protein_values.max(), smiles_values.max()) * 1.1)

# plt.grid(axis='y', linestyle='--', alpha=0.7)
ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)


plt.tight_layout()
plt.savefig('./test2.svg', format='svg', bbox_inches='tight')
plt.show()
