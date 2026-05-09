# =============================================================
# STEP 2: CLUSTERING MODELS
# Real Estate Buyer Segmentation Project
# =============================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os, pickle

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

os.makedirs("charts", exist_ok=True)
os.makedirs("models", exist_ok=True)

print("=" * 60)
print("   REAL ESTATE BUYER SEGMENTATION - CLUSTERING")
print("=" * 60)

# ─────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────
df = pd.read_csv("processed_clients.csv")
print(f"\n✅ Data loaded: {df.shape}")

# ─────────────────────────────────────────
# 2. SELECT CLUSTERING FEATURES
# ─────────────────────────────────────────
cluster_features = [
    'Age', 'satisfaction_score', 'total_properties',
    'avg_sale_price', 'total_investment', 'avg_floor_area',
    'is_investor', 'has_loan', 'is_corporate',
    'price_per_sqft', 'gender_enc', 'referral_enc'
]

X = df[cluster_features].fillna(0)
print(f"✅ Clustering features: {cluster_features}")

# ─────────────────────────────────────────
# 3. SCALE FEATURES
# ─────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
print("✅ Scaler saved")

# ─────────────────────────────────────────
# 4. ELBOW METHOD
# ─────────────────────────────────────────
print("\n📊 Running Elbow Method...")
inertias = []
K_range = range(2, 11)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

plt.figure(figsize=(8,5))
plt.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
plt.axvline(x=4, color='red', linestyle='--', label='Optimal K=4')
plt.title('Elbow Method - Optimal Number of Clusters', fontsize=14, fontweight='bold')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.legend()
plt.tight_layout()
plt.savefig("charts/07_elbow_method.png", dpi=150)
plt.close()
print("✅ Chart saved: charts/07_elbow_method.png")

# ─────────────────────────────────────────
# 5. SILHOUETTE SCORES
# ─────────────────────────────────────────
print("\n📊 Calculating Silhouette Scores...")
sil_scores = []

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    sil_scores.append(score)
    print(f"   K={k}: Silhouette Score = {score:.4f}")

plt.figure(figsize=(8,5))
plt.plot(K_range, sil_scores, 'ro-', linewidth=2, markersize=8)
plt.axvline(x=4, color='blue', linestyle='--', label='Optimal K=4')
plt.title('Silhouette Scores by Number of Clusters', fontsize=14, fontweight='bold')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')
plt.legend()
plt.tight_layout()
plt.savefig("charts/08_silhouette_scores.png", dpi=150)
plt.close()
print("✅ Chart saved: charts/08_silhouette_scores.png")

# ─────────────────────────────────────────
# 6. FINAL K-MEANS (K=4)
# ─────────────────────────────────────────
print("\n🤖 Training Final K-Means (K=4)...")
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

with open("models/kmeans_model.pkl", "wb") as f:
    pickle.dump(kmeans, f)

with open("models/cluster_features.pkl", "wb") as f:
    pickle.dump(cluster_features, f)

print("✅ K-Means model saved")

# ─────────────────────────────────────────
# 7. CLUSTER NAMING
# ─────────────────────────────────────────
cluster_summary = df.groupby('Cluster').agg({
    'Age':               'mean',
    'satisfaction_score':'mean',
    'total_investment':  'mean',
    'avg_sale_price':    'mean',
    'is_investor':       'mean',
    'has_loan':          'mean',
    'is_corporate':      'mean',
    'total_properties':  'mean'
}).round(2)

print("\n📊 Cluster Summary:")
print(cluster_summary)

# Auto name clusters
cluster_names = {}
for cluster in range(4):
    row = cluster_summary.loc[cluster]
    if row['is_corporate'] > 0.3:
        cluster_names[cluster] = "C3 - Corporate Buyers"
    elif row['is_investor'] > 0.5 and row['avg_sale_price'] > cluster_summary['avg_sale_price'].mean():
        cluster_names[cluster] = "C1 - Global Investors"
    elif row['has_loan'] > 0.5 and row['Age'] < cluster_summary['Age'].mean():
        cluster_names[cluster] = "C2 - First-Time Buyers"
    else:
        cluster_names[cluster] = "C4 - Luxury Investors"

df['Segment'] = df['Cluster'].map(cluster_names)

print("\n🏷️  Cluster Names:")
for k, v in cluster_names.items():
    print(f"   Cluster {k}: {v}")

with open("models/cluster_names.pkl", "wb") as f:
    pickle.dump(cluster_names, f)

# ─────────────────────────────────────────
# 8. CLUSTER DISTRIBUTION CHART
# ─────────────────────────────────────────
seg_counts = df['Segment'].value_counts()
colors = ['#3498db','#e74c3c','#2ecc71','#f39c12']

plt.figure(figsize=(8,5))
bars = plt.bar(seg_counts.index, seg_counts.values, color=colors, edgecolor='black')
for bar, val in zip(bars, seg_counts.values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height()+5,
             str(val), ha='center', fontweight='bold')
plt.title('Buyer Segment Distribution', fontsize=14, fontweight='bold')
plt.ylabel('Number of Buyers')
plt.xticks(rotation=15, ha='right')
plt.tight_layout()
plt.savefig("charts/09_segment_distribution.png", dpi=150)
plt.close()
print("✅ Chart saved: charts/09_segment_distribution.png")

# ─────────────────────────────────────────
# 9. PCA VISUALIZATION
# ─────────────────────────────────────────
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(9,6))
for i, (cluster, name) in enumerate(cluster_names.items()):
    mask = df['Cluster'] == cluster
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                label=name, alpha=0.6, s=30, color=colors[i])
plt.title('Buyer Segments - PCA Visualization', fontsize=14, fontweight='bold')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend(fontsize=9)
plt.tight_layout()
plt.savefig("charts/10_pca_clusters.png", dpi=150)
plt.close()
print("✅ Chart saved: charts/10_pca_clusters.png")

# ─────────────────────────────────────────
# 10. HIERARCHICAL CLUSTERING DENDROGRAM
# ─────────────────────────────────────────
from scipy.cluster.hierarchy import dendrogram, linkage

print("\n📊 Running Hierarchical Clustering...")
sample_idx = np.random.choice(len(X_scaled), 200, replace=False)
X_sample   = X_scaled[sample_idx]

linked = linkage(X_sample, method='ward')

plt.figure(figsize=(12, 5))
dendrogram(linked, truncate_mode='lastp', p=20,
           leaf_rotation=45, leaf_font_size=10)
plt.title('Hierarchical Clustering Dendrogram', fontsize=14, fontweight='bold')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.tight_layout()
plt.savefig("charts/11_dendrogram.png", dpi=150)
plt.close()
print("✅ Chart saved: charts/11_dendrogram.png")

# ─────────────────────────────────────────
# 11. INVESTMENT BY SEGMENT
# ─────────────────────────────────────────
seg_invest = df.groupby('Segment')['total_investment'].mean().sort_values(ascending=True)

plt.figure(figsize=(8,5))
colors_inv = ['#3498db','#e74c3c','#2ecc71','#f39c12']
bars = plt.barh(seg_invest.index, seg_invest.values, color=colors_inv, edgecolor='black')
for bar, val in zip(bars, seg_invest.values):
    plt.text(bar.get_width()+1000, bar.get_y()+bar.get_height()/2,
             f'${val:,.0f}', va='center', fontweight='bold')
plt.title('Average Investment by Buyer Segment', fontsize=14, fontweight='bold')
plt.xlabel('Average Total Investment ($)')
plt.tight_layout()
plt.savefig("charts/12_investment_by_segment.png", dpi=150)
plt.close()
print("✅ Chart saved: charts/12_investment_by_segment.png")

# ─────────────────────────────────────────
# 12. SAVE FINAL DATA
# ─────────────────────────────────────────
df.to_csv("segmented_clients.csv", index=False)
print("\n💾 Segmented data saved: segmented_clients.csv")

print("\n" + "=" * 60)
print("   ✅ CLUSTERING COMPLETE!")
print(f"\n   Segments Found: {df['Segment'].nunique()}")
for seg, count in df['Segment'].value_counts().items():
    print(f"   {seg}: {count} buyers")
print("\n   ▶️  Now run: streamlit run 3_streamlit_app.py")
print("=" * 60)
