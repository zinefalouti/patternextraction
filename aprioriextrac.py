import pandas as pd

# Load Apriori rules
df = pd.read_csv("apriori_data.csv")

# Sort by lift (highest first)
df_sorted = df.sort_values(by="lift", ascending=False)

# Option 1: Save the single highest-lift rule
top_rule = df_sorted.head(1)
top_rule.to_csv("apriori_top_lift_single.csv", index=False)

# Option 2: Save all rules with the maximum lift value
max_lift = df["lift"].max()
top_rules = df[df["lift"] == max_lift]
top_rules.to_csv("apriori_top_lift_all.csv", index=False)

# Option 3: Save top N rules (e.g., top 10)
top_n = df_sorted.head(10)
top_n.to_csv("apriori_top_lift_10.csv", index=False)

