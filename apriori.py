import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Load encoded CSV
df = pd.read_csv("parcel_damage_encoded.csv")


# Assume your categorical columns are already one-hot encoded + Damaged
columns_for_apriori = [col for col in df.columns if col not in ['Parcel_ID', 'Packaging_Type']]

df_apriori = df[columns_for_apriori].astype(bool)
frequent_itemsets = apriori(df_apriori, min_support=0.01, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.3)
damage_rules = rules[rules['consequents'].apply(lambda x: 'Damaged' in x)]
damage_rules = damage_rules.sort_values(by='lift', ascending=False)

data = damage_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]

# Convert frozensets to strings for CSV
data['antecedents'] = data['antecedents'].apply(lambda x: ','.join(list(x)))
data['consequents'] = data['consequents'].apply(lambda x: ','.join(list(x)))

# Save to CSV
data.to_csv('apriori_data.csv', index=False)
print("Saved Apriori rules to apriori_data.csv")