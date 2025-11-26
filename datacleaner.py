from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv('parcel_damage_synthetic.csv')

# Label encode Packaging_Type into a single vector
le = LabelEncoder()
df['Packaging_Type_Encoded'] = le.fit_transform(df['Packaging_Type'])

# Categorical columns to one-hot encode (excluding Packaging_Type)
categ_cols = ['Fill_Level', 'Handling_Stage', 'Humidity_Level', 'Stacking']

# Column transformer for one-hot encoding
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categ_cols)
    ],
    remainder='passthrough'
)

# Fit and transform
encoded_data = preprocessor.fit_transform(df)

# Get one-hot feature names
ohe = preprocessor.named_transformers_['cat']
ohe_features = ohe.get_feature_names_out(categ_cols)

# Numeric / passthrough columns
numeric_cols = [col for col in df.columns if col not in categ_cols]

# Rebuild DataFrame
df_encoded = pd.DataFrame(
    encoded_data,
    columns=list(ohe_features) + numeric_cols
)

# Save CSV
df_encoded.to_csv("parcel_damage_encoded.csv", index=False)
print("Saved encoded CSV.")
