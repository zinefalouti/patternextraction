import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load encoded CSV
df_encoded = pd.read_csv("parcel_damage_encoded.csv")

#Main for local execution
def main():
    extracted = split(df_encoded)
    OcMatrix = occur(extracted)
    Eigvals, Eigvecs = eigen(OcMatrix)
    print(f"Eigen Values: {Eigvals} / EigenVectors: {Eigvecs}")
    NorMatrix = normalize(OcMatrix)
    plot(NorMatrix)

#Splitting the Dataset between Damaged and Not Damaged
def split(data):
    damaged_data = data[data['Damaged'] == 1].copy()
    damaged_data = damaged_data.drop(columns=['Packaging_Type','Damaged','Parcel_ID'])
    return damaged_data

#Occurence Matrix Multip
def occur(data):
    X = data.to_numpy()
    Xt = X.T
    C = np.dot(Xt,X)
    return C

#Normalize Occurence Matrix
def normalize(data):
    diag = np.sqrt(np.diag(data))
    C_norm = data / diag[:, None] / diag[None, :]
    return C_norm

def eigen(data):
    eigvals, eigvecs = np.linalg.eig(data)
    return eigvals, eigvecs

#Plot Matrix
def plot(matrix, labels=None, title="Matrix Heatmap"):
    plt.figure(figsize=(8, 6))
    if labels is not None:
        sns.heatmap(matrix, annot=True, fmt=".2f", cmap="Blues", 
                    xticklabels=labels, yticklabels=labels)
    else:
        sns.heatmap(matrix, annot=True, fmt=".2f", cmap="Blues")
    
    plt.title(title)
    plt.xlabel("Features")
    plt.ylabel("Features")
    plt.show()

if __name__ == '__main__':
    main()

