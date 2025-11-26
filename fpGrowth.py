import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules

# Load encoded CSV
df = pd.read_csv("parcel_damage_encoded.csv")

for col in ['Parcel_ID', 'Packaging_Type']:
    if col in df.columns:
        df = df.drop(columns=[col])

columns_for_fp = [col for col in df.columns]
df_fp = df[columns_for_fp].astype(bool)
frequent_itemsets = fpgrowth(df_fp, min_support=0.05, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.3)
damage_rules = rules[rules['consequents'].apply(lambda x: 'Damaged' in x)]
damage_rules = damage_rules.sort_values(by='lift', ascending=False)
damage_rules['antecedents'] = damage_rules['antecedents'].apply(lambda x: ','.join(list(x)))
damage_rules['consequents'] = damage_rules['consequents'].apply(lambda x: ','.join(list(x)))

# --- Save rules to CSV ---
damage_rules.to_csv('fp_growth_damage_rules.csv', index=False)

print("FP-Growth damage rules saved to 'fp_growth_damage_rules.csv'.")
print(damage_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
