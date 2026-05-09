# =============================================================
# STEP 1: EDA & PREPROCESSING
# Real Estate Buyer Segmentation Project
# =============================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

os.makedirs("charts", exist_ok=True)

print("=" * 60)
print("   REAL ESTATE BUYER SEGMENTATION - EDA & PREPROCESSING")
print("=" * 60)

# ─────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────
clients    = pd.read_csv("clients.csv")
properties = pd.read_csv("properties.csv")

print(f"\n✅ Clients loaded:    {clients.shape[0]} rows, {clients.shape[1]} columns")
print(f"✅ Properties loaded: {properties.shape[0]} rows, {properties.shape[1]} columns")

# ─────────────────────────────────────────
# 2. BASIC INFO
# ─────────────────────────────────────────
print("\n📋 Clients Columns:", clients.columns.tolist())
print("\n📋 Properties Columns:", properties.columns.tolist())
print("\n❓ Missing Values - Clients:")
print(clients.isnull().sum())
print("\n❓ Missing Values - Properties:")
print(properties.isnull().sum())

# ─────────────────────────────────────────
# 3. DROP USELESS COLUMNS
# ─────────────────────────────────────────
clients.drop(columns=['first_name', 'last_name'], inplace=True)
print("\n🗑️  Dropped: first_name, last_name")

# ─────────────────────────────────────────
# 4. AGE CALCULATION
# ─────────────────────────────────────────
clients['date_of_birth'] = pd.to_datetime(clients['date_of_birth'], errors='coerce')
clients['Age'] = (datetime.now() - clients['date_of_birth']).dt.days // 365
clients.drop(columns=['date_of_birth'], inplace=True)
print(f"✅ Age calculated - Mean: {clients['Age'].mean():.1f}, Min: {clients['Age'].min()}, Max: {clients['Age'].max()}")

# ─────────────────────────────────────────
# 5. MERGE WITH PROPERTIES
# ─────────────────────────────────────────
properties['sale_price'] = properties['sale_price'].astype(str).str.replace('$','').str.replace(',','').astype(float)
prop_agg = properties.groupby('client_ref').agg(
    total_properties  = ('listing_id', 'count'),
    avg_sale_price    = ('sale_price', 'mean'),
    total_investment  = ('sale_price', 'sum'),
    avg_floor_area    = ('floor_area_sqft', 'mean')
).reset_index().rename(columns={'client_ref': 'client_id'})

df = clients.merge(prop_agg, on='client_id', how='left')
df['total_properties'] = df['total_properties'].fillna(0)
df['avg_sale_price']   = df['avg_sale_price'].fillna(df['avg_sale_price'].median())
df['total_investment'] = df['total_investment'].fillna(0)
df['avg_floor_area']   = df['avg_floor_area'].fillna(df['avg_floor_area'].median())

print(f"\n✅ Merged dataset shape: {df.shape}")

# ─────────────────────────────────────────
# 6. EDA CHARTS
# ─────────────────────────────────────────

# Chart 1: Client Type
plt.figure(figsize=(6,4))
ct = df['client_type'].value_counts()
plt.bar(ct.index, ct.values, color=['#3498db','#e74c3c'], edgecolor='black')
for i, (idx, val) in enumerate(zip(ct.index, ct.values)):
    plt.text(i, val+10, str(val), ha='center', fontweight='bold')
plt.title('Client Type Distribution', fontsize=14, fontweight='bold')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig("charts/01_client_type.png", dpi=150)
plt.close()
print("\n✅ Chart saved: charts/01_client_type.png")

# Chart 2: Acquisition Purpose
plt.figure(figsize=(6,4))
ap = df['acquisition_purpose'].value_counts()
colors = ['#2ecc71','#f39c12']
plt.bar(ap.index, ap.values, color=colors, edgecolor='black')
for i, (idx, val) in enumerate(zip(ap.index, ap.values)):
    plt.text(i, val+10, str(val), ha='center', fontweight='bold')
plt.title('Acquisition Purpose Distribution', fontsize=14, fontweight='bold')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig("charts/02_acquisition_purpose.png", dpi=150)
plt.close()
print("✅ Chart saved: charts/02_acquisition_purpose.png")

# Chart 3: Loan Applied
plt.figure(figsize=(6,4))
la = df['loan_applied'].value_counts()
plt.bar(la.index, la.values, color=['#9b59b6','#1abc9c'], edgecolor='black')
for i, (idx, val) in enumerate(zip(la.index, la.values)):
    plt.text(i, val+10, str(val), ha='center', fontweight='bold')
plt.title('Loan Applied Distribution', fontsize=14, fontweight='bold')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig("charts/03_loan_applied.png", dpi=150)
plt.close()
print("✅ Chart saved: charts/03_loan_applied.png")

# Chart 4: Top Countries
plt.figure(figsize=(8,4))
top_countries = df['country'].value_counts().head(8)
plt.bar(top_countries.index, top_countries.values, color='#3498db', edgecolor='black')
plt.title('Top Countries by Buyer Count', fontsize=14, fontweight='bold')
plt.ylabel('Count')
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("charts/04_top_countries.png", dpi=150)
plt.close()
print("✅ Chart saved: charts/04_top_countries.png")

# Chart 5: Age Distribution
plt.figure(figsize=(8,4))
plt.hist(df['Age'].dropna(), bins=30, color='#e74c3c', edgecolor='black', alpha=0.7)
plt.title('Age Distribution of Buyers', fontsize=14, fontweight='bold')
plt.xlabel('Age')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig("charts/05_age_distribution.png", dpi=150)
plt.close()
print("✅ Chart saved: charts/05_age_distribution.png")

# Chart 6: Satisfaction Score
plt.figure(figsize=(6,4))
ss = df['satisfaction_score'].value_counts().sort_index()
plt.bar(ss.index, ss.values, color='#f39c12', edgecolor='black')
plt.title('Satisfaction Score Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Score')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig("charts/06_satisfaction_score.png", dpi=150)
plt.close()
print("✅ Chart saved: charts/06_satisfaction_score.png")

# ─────────────────────────────────────────
# 7. FEATURE ENGINEERING
# ─────────────────────────────────────────
print("\n⚙️  Feature Engineering...")
df['is_investor']     = (df['acquisition_purpose'] == 'Investment').astype(int)
df['has_loan']        = (df['loan_applied'] == 'Yes').astype(int)
df['is_corporate']    = (df['client_type'] == 'Company').astype(int)
df['price_per_sqft']  = df['avg_sale_price'] / (df['avg_floor_area'] + 1)

print("✅ New features: is_investor, has_loan, is_corporate, price_per_sqft")

# ─────────────────────────────────────────
# 8. ENCODING
# ─────────────────────────────────────────
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['gender_enc']   = le.fit_transform(df['gender'].fillna('M'))
df['referral_enc'] = le.fit_transform(df['referral_channel'])

df = pd.get_dummies(df, columns=['region'], prefix='region')

print(f"\n✅ After encoding, shape: {df.shape}")

# ─────────────────────────────────────────
# 9. SAVE PROCESSED DATA
# ─────────────────────────────────────────
df.to_csv("processed_clients.csv", index=False)
print("\n💾 Processed data saved: processed_clients.csv")

print("\n" + "=" * 60)
print("   ✅ EDA & PREPROCESSING COMPLETE!")
print("   ▶️  Now run: python 2_clustering.py")
print("=" * 60)
