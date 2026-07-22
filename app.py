import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.impute import KNNImputer


st.set_page_config(page_title="Laptop Analytics Pro", layout="wide")



def smart_cleaning_pipeline(df_input: pd.DataFrame) -> pd.DataFrame:
    """Full cleaning logic from the notebook."""
    df_temp = df_input.copy()

    # 1. Regex Cleaning for Numeric Columns
    cols_to_fix = ['Price', 'Total Sales', 'cpu_speed', 'screen_size',
                   'harddisk', 'ram', 'Available Stock', 'Sale Product Count']

    for col in cols_to_fix:
        if col in df_temp.columns:
            df_temp[col] = df_temp[col].astype(str).str.replace(r'[^\d.]', '', regex=True)
            df_temp[col] = pd.to_numeric(df_temp[col], errors='coerce')

    # 2. Categorical Imputation
    cat_cols = df_temp.select_dtypes(include=['object']).columns
    df_temp[cat_cols] = df_temp[cat_cols].fillna("Unknown")

    # 3. Numeric Imputation Logic (Median / KNN hybrid)
    for col in cols_to_fix:
        if col not in df_temp.columns:
            continue

        missing_pct = df_temp[col].isnull().mean()

        if missing_pct > 0.60:  # Drop if > 60% missing
            df_temp = df_temp.drop(columns=[col])
            continue

        num_only = df_temp.select_dtypes(include=[np.number])
        correlations = num_only.corr().abs()[col].drop(col, errors='ignore')
        max_corr = correlations.max() if not correlations.empty else 0

        if missing_pct < 0.05:
            df_temp[col] = df_temp[col].fillna(df_temp[col].median())
        elif 0.05 <= missing_pct <= 0.30:
            if max_corr > 0.3:
                imputer = KNNImputer(n_neighbors=5)
                df_temp[[col]] = imputer.fit_transform(df_temp[[col]])
            else:
                df_temp[col] = df_temp[col].fillna(df_temp[col].median())
        elif 0.30 < missing_pct <= 0.60:
            df_temp[col] = df_temp[col].fillna(df_temp[col].median())

    # 4. Outlier Handling (Total Sales)
    if 'Total Sales' in df_temp.columns:
        Q1 = df_temp['Total Sales'].quantile(0.25)
        Q3 = df_temp['Total Sales'].quantile(0.75)
        IQR = Q3 - Q1
        upper_limit = Q3 + 5 * IQR
        df_temp = df_temp[df_temp['Total Sales'] <= upper_limit]

    # 5. String Normalization
    for text_col in ['brand', 'model']:
        if text_col in df_temp.columns:
            df_temp[text_col] = df_temp[text_col].astype(str).str.lower().str.strip()

    return df_temp


@st.cache_data(show_spinner=False)
def load_and_clean(file_bytes: bytes) -> pd.DataFrame:
 
    df_raw = pd.read_excel(file_bytes)
    df_cleaned = smart_cleaning_pipeline(df_raw)

  
    cols_to_fix = ['brand', 'OS', 'cpu', 'graphics', 'color', 'special_features']
    for col in cols_to_fix:
        if col in df_cleaned.columns:
            df_cleaned[col] = df_cleaned[col].astype(str).replace('nan', np.nan).replace('Unknown', np.nan)

    return df_cleaned


# ============================================================
# SIDEBAR — UPLOAD
# ============================================================
st.sidebar.header("📁 Charger les données")
uploaded_file = st.sidebar.file_uploader(
    "Dépose ton fichier Excel (.xlsx)",
    type=["xlsx"],
    help="Le fichier sera nettoyé automatiquement (regex, imputation, outliers) avant l'affichage."
)

if uploaded_file is None:
    st.title("💻 Laptop Analytics Pro")
    st.info("👈 Charge un fichier Excel dans la barre latérale pour démarrer l'analyse.")
    st.stop()

# ============================================================
# NETTOYAGE + CHARGEMENT
# ============================================================
with st.spinner("Nettoyage des données en cours..."):
    try:
        df = load_and_clean(uploaded_file)
    except Exception as e:
        st.error("Impossible de lire ou nettoyer ce fichier.")
        st.info(f"Détail technique : {e}")
        st.stop()

st.sidebar.success(f"✅ {len(df)} lignes chargées et nettoyées")

csv_buffer = df.to_csv(index=False).encode("utf-8")
st.sidebar.download_button(
    "⬇️ Télécharger les données nettoyées (CSV)",
    data=csv_buffer,
    file_name="laptops_cleaned.csv",
    mime="text/csv"
)


try:
    st.sidebar.header("🔍 Filtres")

    def get_options(column_name):
        if column_name not in df.columns:
            return []
        return sorted(df[column_name].dropna().unique().tolist())

    brand_filter = st.sidebar.multiselect("Marque", options=get_options('brand'), default=get_options('brand'))
    os_filter = st.sidebar.multiselect("OS", options=get_options('OS'), default=get_options('OS'))
    cpu_filter = st.sidebar.multiselect("CPU", options=get_options('cpu'), default=get_options('cpu'))
    gpu_filter = st.sidebar.multiselect("Carte graphique", options=get_options('graphics'), default=get_options('graphics'))
    color_filter = st.sidebar.multiselect("Couleur", options=get_options('color'), default=get_options('color'))
    feature_filter = st.sidebar.multiselect("Fonctionnalités spéciales", options=get_options('special_features'))

    mask = pd.Series(True, index=df.index)
    for col, sel in [('brand', brand_filter), ('OS', os_filter), ('cpu', cpu_filter),
                      ('graphics', gpu_filter), ('color', color_filter)]:
        if col in df.columns:
            mask &= df[col].isin(sel)

    filtered_df = df[mask].copy()

    if feature_filter and 'special_features' in filtered_df.columns:
        pattern = '|'.join(feature_filter)
        filtered_df = filtered_df[filtered_df['special_features'].str.contains(pattern, na=False)]

    st.title("💻 Laptop Analytics Pro")

  
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Revenu total", f"${filtered_df['Total Sales'].sum():,.2f}")
    with col2:
        st.metric("Prix moyen", f"${filtered_df['Price'].mean():,.2f}")
    with col3:
        st.metric("Unités vendues", f"{int(filtered_df['Sale Product Count'].sum())}")
    with col4:
        st.metric("Stock dispo", f"{int(filtered_df['Available Stock'].sum())}")

    st.markdown("---")

  
    st.subheader("🔍 Analyse des prix par caractéristique")
    potential_x = ['ram', 'harddisk', 'screen_size', 'rating', 'cpu_speed']
    available_x = [col for col in potential_x if col in filtered_df.columns]

    if available_x:
        feature_x = st.selectbox("Comparer le prix avec", options=available_x)
        fig_price = px.scatter(
            filtered_df, x=feature_x, y='Price',
            color='brand', hover_name='model',
            title=f"Prix vs {feature_x.replace('_', ' ').capitalize()}",
            template="plotly_white"
        )
        st.plotly_chart(fig_price, use_container_width=True)
    else:
        st.info("Ajoute plus de colonnes techniques pour débloquer cette analyse.")


    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("🏢 Revenu par marque")
        brand_rev = filtered_df.groupby('brand')['Total Sales'].sum().reset_index()
        fig_brand = px.pie(brand_rev, values='Total Sales', names='brand', hole=0.4)
        st.plotly_chart(fig_brand, use_container_width=True)

    with col_b:
        st.subheader("📦 Revenu par marque & modèle")
        fig_tree = px.treemap(
            filtered_df, path=['brand', 'model'], values='Total Sales',
            color='Total Sales', color_continuous_scale='RdBu'
        )
        st.plotly_chart(fig_tree, use_container_width=True)

 
    st.subheader("🏆 Produits les plus vendus")
    top_products = filtered_df.nlargest(10, 'Sale Product Count')[
        ['brand', 'model', 'color', 'cpu', 'Sale Product Count', 'Total Sales']
    ]
    st.dataframe(top_products, use_container_width=True)

except Exception as e:
    st.error("Erreur lors de l'affichage du dashboard.")
    st.info(f"Détail technique : {e}")