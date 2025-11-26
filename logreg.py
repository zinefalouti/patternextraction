import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# --- Load dataset ---
df = pd.read_csv("parcel_damage_synthetic.csv")

if 'Parcel_ID' in df.columns:
    df = df.drop(columns=['Parcel_ID'])

X = df.drop(columns=['Damaged'])
y = df['Damaged']
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)],
    remainder='passthrough'
)
X_encoded = preprocessor.fit_transform(X)

# --- Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# --- Fit logistic regression ---
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)

# --- Predictions ---
y_pred_prob = logreg.predict_proba(X_test)[:, 1]  # Probability of Damaged=1
y_pred = (y_pred_prob >= 0.5).astype(int)         # Binary predictions

# --- Evaluation ---
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# --- Feature importance (coefficients) ---
feature_names = preprocessor.get_feature_names_out()
coeff_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': logreg.coef_[0]
}).sort_values(by='Coefficient', ascending=False)

print("\nTop features increasing damage risk:")
print(coeff_df.head(10))
