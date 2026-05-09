# =============================================================
# STREAMLIT APP - Real Estate Buyer Segmentation Dashboard
# Self-contained: trains model on startup
# =============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

st.set_page_config(
    page_title="Real Estate Buyer Intelligence",
    page_icon="🏢",
    layout="wide"
)

st.markdown("""
<style>
    .stApp { background-color: #0a1628; color: white; }
    h1, h2, h3 { color: #f39c12 !important; }
    .stSelectbox label, .stMultiSelect label { color: white !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# TRAIN MODEL ON STARTUP
# ─────────────────────────────────────────
@st.cache_resource
def load_and_train():
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    clients    = pd.read_csv("clients(1).csv")
    properties = pd.read_csv("properties(2) (1).csv")

    # Clean
    clients.drop(columns=['first_name','last_name'], inplace=True)
    clients['date_of_birth'] = pd.to_datetime(clients['date_of_birth'], errors='coerce')
    clients['Age'] = (datetime.now() - clients['date_of_birth']).dt.days // 365
    clients.drop(columns=['date_of_birth'], inplace=True)

    # Properties merge
    properties['sale_price'] = properties['sale_price'].astype(str).str.replace('$','').str.replace(',','').astype(float)
    prop_agg = properties.groupby('client_ref').agg(
        total_properties=('listing_id','count'),
        avg_sale_price  =('sale_price','mean'),
        total_investment=('sale_price','sum'),
        avg_floor_area  =('floor_area_sqft','mean')
    ).reset_index().rename(columns={'client_ref':'client_id'})

    df = clients.merge(prop_agg, on='client_id', how='left')
    df['total_properties'] = df['total_properties'].fillna(0)
    df['avg_sale_price']   = df['avg_sale_price'].fillna(df['avg_sale_price'].median())
    df['total_investment'] = df['total_investment'].fillna(0)
    df['avg_floor_area']   = df['avg_floor_area'].fillna(df['avg_floor_area'].median())

    # Feature Engineering
    df['is_investor']    = (df['acquisition_purpose'] == 'Investment').astype(int)
    df['has_loan']       = (df['loan_applied'] == 'Yes').astype(int)
    df['is_corporate']   = (df['client_type'] == 'Company').astype(int)
    df['price_per_sqft'] = df['avg_sale_price'] / (df['avg_floor_area'] + 1)

    le = LabelEncoder()
    df['gender_enc']   = le.fit_transform(df['gender'].fillna('M'))
    df['referral_enc'] = le.fit_transform(df['referral_channel'])

    cluster_features = [
        'Age','satisfaction_score','total_properties',
        'avg_sale_price','total_investment','avg_floor_area',
        'is_investor','has_loan','is_corporate',
        'price_per_sqft','gender_enc','referral_enc'
    ]

    X = df[cluster_features].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)

    # Auto name
    summary = df.groupby('Cluster').agg({
        'is_corporate':'mean','is_investor':'mean',
        'has_loan':'mean','Age':'mean','avg_sale_price':'mean'
    })
    avg_price = summary['avg_sale_price'].mean()
    avg_age   = summary['Age'].mean()

    cluster_names = {}
    for c in range(4):
        row = summary.loc[c]
        if row['is_corporate'] > 0.3:
            cluster_names[c] = "C3 - Corporate Buyers"
        elif row['is_investor'] > 0.5 and row['avg_sale_price'] > avg_price:
            cluster_names[c] = "C1 - Global Investors"
        elif row['has_loan'] > 0.5 and row['Age'] < avg_age:
            cluster_names[c] = "C2 - First-Time Buyers"
        else:
            cluster_names[c] = "C4 - Luxury Investors"

    df['Segment'] = df['Cluster'].map(cluster_names)

    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    df['PCA1'] = X_pca[:, 0]
    df['PCA2'] = X_pca[:, 1]

    return df, kmeans, scaler, cluster_features, cluster_names

with st.spinner("🔄 Loading intelligence engine..."):
    df, kmeans, scaler, cluster_features, cluster_names = load_and_train()

COLORS = ['#3498db','#e74c3c','#2ecc71','#f39c12']
SEGMENTS = sorted(df['Segment'].unique())

# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/real-estate.png", width=70)
st.sidebar.title("🏢 RE Buyer Intelligence")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigate To:", [
    "🏠 Overview",
    "📊 Buyer Segmentation",
    "💰 Investor Behavior",
    "🌍 Geographic Analysis",
    "🔍 Segment Insights",
])

st.sidebar.markdown("---")
st.sidebar.subheader("🔽 Filters")
sel_country = st.sidebar.multiselect("Country", sorted(df['country'].unique()), default=[])
sel_purpose = st.sidebar.multiselect("Acquisition Purpose", df['acquisition_purpose'].unique(), default=[])
sel_type    = st.sidebar.multiselect("Client Type", df['client_type'].unique(), default=[])

# Apply filters
dff = df.copy()
if sel_country: dff = dff[dff['country'].isin(sel_country)]
if sel_purpose: dff = dff[dff['acquisition_purpose'].isin(sel_purpose)]
if sel_type:    dff = dff[dff['client_type'].isin(sel_type)]

# ─────────────────────────────────────────
# PAGE 1: OVERVIEW
# ─────────────────────────────────────────
if page == "🏠 Overview":
    st.title("🏢 Real Estate Buyer Intelligence Dashboard")
    st.markdown("#### ML-Based Buyer Segmentation & Investment Profiling")
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Clients", f"{len(dff):,}")
    c2.metric("Avg Investment", f"${dff['total_investment'].mean():,.0f}")
    c3.metric("Investor %", f"{dff['is_investor'].mean()*100:.1f}%")
    c4.metric("Loan Applied %", f"{dff['has_loan'].mean()*100:.1f}%")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("👥 Client Type")
        ct = dff['client_type'].value_counts()
        fig, ax = plt.subplots(figsize=(5,4), facecolor='#0a1628')
        ax.set_facecolor('#0a1628')
        ax.pie(ct.values, labels=ct.index, colors=['#3498db','#e74c3c'],
               autopct='%1.1f%%', textprops={'color':'white'})
        ax.set_title('Client Type Split', color='white', fontweight='bold')
        st.pyplot(fig); plt.close()

    with col2:
        st.subheader("🎯 Acquisition Purpose")
        ap = dff['acquisition_purpose'].value_counts()
        fig, ax = plt.subplots(figsize=(5,4), facecolor='#0a1628')
        ax.set_facecolor('#0a1628')
        ax.pie(ap.values, labels=ap.index, colors=['#2ecc71','#f39c12'],
               autopct='%1.1f%%', textprops={'color':'white'})
        ax.set_title('Acquisition Purpose', color='white', fontweight='bold')
        st.pyplot(fig); plt.close()

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("💳 Loan Applied")
        la = dff['loan_applied'].value_counts()
        fig, ax = plt.subplots(figsize=(5,3), facecolor='#0a1628')
        ax.set_facecolor('#0a1628')
        bars = ax.bar(la.index, la.values, color=['#9b59b6','#1abc9c'], edgecolor='white')
        for bar, val in zip(bars, la.values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+5,
                    str(val), ha='center', color='white', fontweight='bold')
        ax.tick_params(colors='white'); ax.spines[:].set_visible(False)
        ax.set_title('Loan Applied', color='white', fontweight='bold')
        st.pyplot(fig); plt.close()

    with col2:
        st.subheader("⭐ Satisfaction Score")
        fig, ax = plt.subplots(figsize=(5,3), facecolor='#0a1628')
        ax.set_facecolor('#0a1628')
        ss = dff['satisfaction_score'].value_counts().sort_index()
        ax.bar(ss.index, ss.values, color='#f39c12', edgecolor='white')
        ax.tick_params(colors='white'); ax.spines[:].set_visible(False)
        ax.set_title('Satisfaction Score', color='white', fontweight='bold')
        st.pyplot(fig); plt.close()

# ─────────────────────────────────────────
# PAGE 2: BUYER SEGMENTATION
# ─────────────────────────────────────────
elif page == "📊 Buyer Segmentation":
    st.title("📊 Buyer Segmentation Overview")
    st.markdown("---")

    seg_counts = dff['Segment'].value_counts()
    c1, c2, c3, c4 = st.columns(4)
    for col, (seg, count) in zip([c1,c2,c3,c4], seg_counts.items()):
        col.metric(seg, f"{count} buyers")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🥧 Segment Distribution")
        fig, ax = plt.subplots(figsize=(6,5), facecolor='#0a1628')
        ax.set_facecolor('#0a1628')
        ax.pie(seg_counts.values, labels=seg_counts.index,
               colors=COLORS, autopct='%1.1f%%', textprops={'color':'white', 'fontsize':9})
        st.pyplot(fig); plt.close()

    with col2:
        st.subheader("📊 Segment Count")
        fig, ax = plt.subplots(figsize=(6,5), facecolor='#0a1628')
        ax.set_facecolor('#0a1628')
        bars = ax.bar(range(len(seg_counts)), seg_counts.values, color=COLORS, edgecolor='white')
        ax.set_xticks(range(len(seg_counts)))
        ax.set_xticklabels([s.split(' - ')[1] if ' - ' in s else s for s in seg_counts.index],
                           rotation=15, ha='right', color='white', fontsize=9)
        for bar, val in zip(bars, seg_counts.values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2,
                    str(val), ha='center', color='white', fontweight='bold')
        ax.tick_params(colors='white'); ax.spines[:].set_visible(False)
        st.pyplot(fig); plt.close()

    st.markdown("---")
    st.subheader("🔵 PCA Cluster Visualization")
    fig, ax = plt.subplots(figsize=(10,5), facecolor='#0a1628')
    ax.set_facecolor('#0a1628')
    for i, seg in enumerate(SEGMENTS):
        mask = dff['Segment'] == seg
        ax.scatter(dff.loc[mask,'PCA1'], dff.loc[mask,'PCA2'],
                   label=seg, alpha=0.6, s=20, color=COLORS[i % len(COLORS)])
    ax.set_xlabel('PCA Component 1', color='white')
    ax.set_ylabel('PCA Component 2', color='white')
    ax.tick_params(colors='white'); ax.spines[:].set_visible(False)
    ax.legend(facecolor='#1a2a3a', labelcolor='white', fontsize=8)
    ax.set_title('Buyer Segments - PCA View', color='white', fontweight='bold')
    st.pyplot(fig); plt.close()

# ─────────────────────────────────────────
# PAGE 3: INVESTOR BEHAVIOR
# ─────────────────────────────────────────
elif page == "💰 Investor Behavior":
    st.title("💰 Investor Behavior Dashboard")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("💵 Avg Investment by Segment")
        seg_inv = dff.groupby('Segment')['total_investment'].mean().sort_values()
        fig, ax = plt.subplots(figsize=(7,5), facecolor='#0a1628')
        ax.set_facecolor('#0a1628')
        ax.barh(range(len(seg_inv)), seg_inv.values, color=COLORS, edgecolor='white')
        ax.set_yticks(range(len(seg_inv)))
        ax.set_yticklabels([s.split(' - ')[1] if ' - ' in s else s for s in seg_inv.index],
                           color='white', fontsize=9)
        for i, val in enumerate(seg_inv.values):
            ax.text(val+1000, i, f'${val:,.0f}', va='center', color='white', fontsize=8)
        ax.tick_params(colors='white'); ax.spines[:].set_visible(False)
        ax.set_title('Avg Total Investment', color='white', fontweight='bold')
        st.pyplot(fig); plt.close()

    with col2:
        st.subheader("🏠 Loan Rate by Segment")
        seg_loan = dff.groupby('Segment')['has_loan'].mean() * 100
        fig, ax = plt.subplots(figsize=(7,5), facecolor='#0a1628')
        ax.set_facecolor('#0a1628')
        bars = ax.bar(range(len(seg_loan)), seg_loan.values, color=COLORS, edgecolor='white')
        ax.set_xticks(range(len(seg_loan)))
        ax.set_xticklabels([s.split(' - ')[1] if ' - ' in s else s for s in seg_loan.index],
                           rotation=15, ha='right', color='white', fontsize=9)
        for bar, val in zip(bars, seg_loan.values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                    f'{val:.1f}%', ha='center', color='white', fontweight='bold')
        ax.tick_params(colors='white'); ax.spines[:].set_visible(False)
        ax.set_title('Loan Application Rate (%)', color='white', fontweight='bold')
        st.pyplot(fig); plt.close()

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📦 Avg Properties by Segment")
        seg_prop = dff.groupby('Segment')['total_properties'].mean().sort_values()
        fig, ax = plt.subplots(figsize=(7,4), facecolor='#0a1628')
        ax.set_facecolor('#0a1628')
        ax.barh(range(len(seg_prop)), seg_prop.values, color=COLORS, edgecolor='white')
        ax.set_yticks(range(len(seg_prop)))
        ax.set_yticklabels([s.split(' - ')[1] if ' - ' in s else s for s in seg_prop.index],
                           color='white', fontsize=9)
        ax.tick_params(colors='white'); ax.spines[:].set_visible(False)
        ax.set_title('Avg Number of Properties', color='white', fontweight='bold')
        st.pyplot(fig); plt.close()

    with col2:
        st.subheader("⭐ Satisfaction by Segment")
        seg_sat = dff.groupby('Segment')['satisfaction_score'].mean().sort_values()
        fig, ax = plt.subplots(figsize=(7,4), facecolor='#0a1628')
        ax.set_facecolor('#0a1628')
        bars = ax.bar(range(len(seg_sat)), seg_sat.values, color=COLORS, edgecolor='white')
        ax.set_xticks(range(len(seg_sat)))
        ax.set_xticklabels([s.split(' - ')[1] if ' - ' in s else s for s in seg_sat.index],
                           rotation=15, ha='right', color='white', fontsize=9)
        for bar, val in zip(bars, seg_sat.values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                    f'{val:.2f}', ha='center', color='white', fontweight='bold')
        ax.tick_params(colors='white'); ax.spines[:].set_visible(False)
        ax.set_ylim(0, 5.5)
        ax.set_title('Avg Satisfaction Score', color='white', fontweight='bold')
        st.pyplot(fig); plt.close()

# ─────────────────────────────────────────
# PAGE 4: GEOGRAPHIC ANALYSIS
# ─────────────────────────────────────────
elif page == "🌍 Geographic Analysis":
    st.title("🌍 Geographic Buyer Analysis")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🌐 Top Countries by Buyers")
        top_c = dff['country'].value_counts().head(8)
        fig, ax = plt.subplots(figsize=(7,5), facecolor='#0a1628')
        ax.set_facecolor('#0a1628')
        ax.barh(range(len(top_c)), top_c.values, color='#3498db', edgecolor='white')
        ax.set_yticks(range(len(top_c)))
        ax.set_yticklabels(top_c.index, color='white')
        ax.tick_params(colors='white'); ax.spines[:].set_visible(False)
        ax.set_title('Top Countries', color='white', fontweight='bold')
        st.pyplot(fig); plt.close()

    with col2:
        st.subheader("📍 Top Regions by Buyers")
        top_r = dff['region'].value_counts().head(8)
        fig, ax = plt.subplots(figsize=(7,5), facecolor='#0a1628')
        ax.set_facecolor('#0a1628')
        ax.barh(range(len(top_r)), top_r.values, color='#e74c3c', edgecolor='white')
        ax.set_yticks(range(len(top_r)))
        ax.set_yticklabels(top_r.index, color='white')
        ax.tick_params(colors='white'); ax.spines[:].set_visible(False)
        ax.set_title('Top Regions', color='white', fontweight='bold')
        st.pyplot(fig); plt.close()

    st.markdown("---")
    st.subheader("🗺️ Segment Distribution by Country (Top 6)")
    top6 = dff['country'].value_counts().head(6).index
    geo_seg = dff[dff['country'].isin(top6)].groupby(['country','Segment']).size().unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=(12,5), facecolor='#0a1628')
    ax.set_facecolor('#0a1628')
    geo_seg.plot(kind='bar', ax=ax, color=COLORS, edgecolor='white')
    ax.set_title('Buyer Segments by Country', color='white', fontweight='bold')
    ax.tick_params(colors='white', axis='both')
    ax.spines[:].set_visible(False)
    ax.legend(facecolor='#1a2a3a', labelcolor='white', fontsize=8)
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    st.pyplot(fig); plt.close()

# ─────────────────────────────────────────
# PAGE 5: SEGMENT INSIGHTS
# ─────────────────────────────────────────
elif page == "🔍 Segment Insights":
    st.title("🔍 Segment Insights Panel")
    st.markdown("---")

    selected_seg = st.selectbox("Select Segment", SEGMENTS)
    seg_df = dff[dff['Segment'] == selected_seg]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Buyers", f"{len(seg_df):,}")
    c2.metric("Avg Age", f"{seg_df['Age'].mean():.0f} yrs")
    c3.metric("Avg Investment", f"${seg_df['total_investment'].mean():,.0f}")
    c4.metric("Avg Satisfaction", f"{seg_df['satisfaction_score'].mean():.2f}/5")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Descriptive Stats")
        stats = seg_df[['Age','satisfaction_score','total_investment',
                         'avg_sale_price','total_properties']].describe().round(2)
        st.dataframe(stats, use_container_width=True)

    with col2:
        st.subheader("🎯 Key Characteristics")
        inv_pct  = seg_df['is_investor'].mean() * 100
        loan_pct = seg_df['has_loan'].mean() * 100
        corp_pct = seg_df['is_corporate'].mean() * 100
        top_country = seg_df['country'].value_counts().index[0]
        top_channel = seg_df['referral_channel'].value_counts().index[0]

        st.info(f"**Investment Purpose:** {inv_pct:.1f}% buyers")
        st.info(f"**Loan Applied:** {loan_pct:.1f}% buyers")
        st.info(f"**Corporate Buyers:** {corp_pct:.1f}%")
        st.info(f"**Top Country:** {top_country}")
        st.info(f"**Top Referral Channel:** {top_channel}")

    st.markdown("---")
    st.subheader("📋 Sample Buyers in This Segment")
    show_cols = ['client_id','client_type','gender','country','region',
                 'acquisition_purpose','loan_applied','satisfaction_score',
                 'total_investment','Age']
    st.dataframe(seg_df[show_cols].head(10).reset_index(drop=True),
                 use_container_width=True)

# FOOTER
st.sidebar.markdown("---")
st.sidebar.markdown("**🏢 Real Estate Buyer Intelligence**")
st.sidebar.markdown("Powered by K-Means Clustering")
st.sidebar.markdown("Parcl Co. x Unified Mentor")
