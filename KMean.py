import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load encoded CSV
df_encoded = pd.read_csv("parcel_damage_encoded.csv")

# Drop Parcel_ID and original Packaging_Type if they exist
for col in ['Parcel_ID', 'Packaging_Type']:
    if col in df_encoded.columns:
        df_encoded = df_encoded.drop(columns=[col])

X = df_encoded.drop(columns=['Damaged'], errors='ignore')
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
if 'Packaging_Type_Encoded' in df_encoded.columns:
    n_clusters = df_encoded['Packaging_Type_Encoded'].nunique()
else:
    n_clusters = 3  # fallback

print(f"Number of clusters (K) = {n_clusters}")
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
df_encoded['Cluster'] = clusters
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8,6))
for c in range(n_clusters):
    plt.scatter(X_pca[clusters == c, 0], X_pca[clusters == c, 1], label=f'Cluster {c}', alpha=0.7)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('K-Means Clusters (PCA 2D)')
plt.legend()
plt.show()

# --- Plot damage distribution ---
if 'Damaged' in df_encoded.columns:
    plt.figure(figsize=(8,6))
    plt.scatter(X_pca[:,0], X_pca[:,1], c=df_encoded['Damaged'], cmap='coolwarm', alpha=0.7)
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('Parcel Damage Visualization (color = Damaged)')
    plt.colorbar(label='Damaged (0=No, 1=Yes)')
    plt.show()

# --- Damage summary per cluster ---
if 'Damaged' in df_encoded.columns:
    damage_summary = df_encoded.groupby('Cluster')['Damaged'].agg(['count', 'sum'])
    damage_summary['Damage_Proportion'] = damage_summary['sum'] / damage_summary['count']
    print("\nDamage summary per cluster:")
    print(damage_summary)
