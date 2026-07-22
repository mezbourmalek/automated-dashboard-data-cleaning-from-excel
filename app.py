import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(page_title="Laptop Analytics Pro", layout="wide")


# ============================================================
# PIPELINE DE NETTOYAGE (reprise de cleaner2.py, appliquée
# directement sur le fichier uploadé, sans passer par le disque)
# ============================================================
def smart_cleaning_pipeline(df_input: pd.DataFrame) -> pd.DataFrame:
    """Full cleaning logic from the notebook."""
    df_temp = df_input.copy()

    cols_to_fix = ['Price', 'Total Sales', 'cpu_speed', 'screen_size',
                   'harddisk', 'ram', 'Available Stock', 'Sale Product Count']

    for col in cols_to_fix:
        if col in df_temp.columns:
            df_temp[col] = df_temp[col].astype(str).str.replace(r'[^\d.]', '', regex=True)
            df_temp[col] = pd.to_numeric(df_temp[col], errors='coerce')

    cat_cols = df_temp.select_dtypes(include=['object']).columns
    df_temp[cat_cols] = df_temp[cat_cols].fillna("Unknown")

    for col in cols_to_fix:
        if col not in df_temp.columns:
            continue

        missing_pct = df_temp[col].isnull().mean()

        if missing_pct > 0.60:
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

    if 'Total Sales' in df_temp.columns:
        Q1 = df_temp['Total Sales'].quantile(0.25)
        Q3 = df_temp['Total Sales'].quantile(0.75)
        IQR = Q3 - Q1
        upper_limit = Q3 + 5 * IQR
        df_temp = df_temp[df_temp['Total Sales'] <= upper_limit]

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
# HELPERS ML
# ============================================================
NUMERIC_FEATURES = ['ram', 'harddisk', 'screen_size', 'cpu_speed']
CATEGORICAL_FEATURES = ['brand', 'OS', 'graphics']


def collapse_rare_categories(series: pd.Series, top_n: int = 10) -> pd.Series:
    """Garde les top_n catégories les plus fréquentes, regroupe le reste en 'Other'."""
    top_values = series.value_counts().nlargest(top_n).index
    return series.where(series.isin(top_values), other='Other')


def build_feature_matrix(df: pd.DataFrame, extra_numeric=None):
    """Construit une matrice de features (numériques + one-hot) à partir du df nettoyé."""
    numeric_cols = [c for c in NUMERIC_FEATURES if c in df.columns]
    if extra_numeric:
        numeric_cols = numeric_cols + [c for c in extra_numeric if c in df.columns and c not in numeric_cols]
    categorical_cols = [c for c in CATEGORICAL_FEATURES if c in df.columns]

    work = df[numeric_cols + categorical_cols].copy()

    for col in numeric_cols:
        work[col] = work[col].fillna(work[col].median())

    encoders_info = {}
    for col in categorical_cols:
        work[col] = work[col].fillna('Unknown').astype(str)
        collapsed = collapse_rare_categories(work[col], top_n=10)
        encoders_info[col] = sorted(collapsed.unique().tolist())
        work[col] = collapsed

    work_encoded = pd.get_dummies(work, columns=categorical_cols)
    return work_encoded, numeric_cols, categorical_cols, encoders_info


@st.cache_resource(show_spinner=False)
def train_price_model(df_hash: str, df: pd.DataFrame):
    data = df.dropna(subset=['Price']).copy()
    X, numeric_cols, categorical_cols, enc_info = build_feature_matrix(data)
    y = data.loc[X.index, 'Price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)

    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

    return {
        'model': model,
        'columns': X.columns.tolist(),
        'numeric_cols': numeric_cols,
        'categorical_cols': categorical_cols,
        'enc_info': enc_info,
        'r2': r2,
        'mae': mae,
        'importances': importances,
    }


@st.cache_resource(show_spinner=False)
def train_demand_model(df_hash: str, df: pd.DataFrame):
    data = df.dropna(subset=['Total Sales', 'Price']).copy()
    X, numeric_cols, categorical_cols, enc_info = build_feature_matrix(data, extra_numeric=['Price'])
    y = data.loc[X.index, 'Total Sales']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)

    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

    return {
        'model': model,
        'columns': X.columns.tolist(),
        'numeric_cols': numeric_cols,
        'categorical_cols': categorical_cols,
        'enc_info': enc_info,
        'r2': r2,
        'mae': mae,
        'importances': importances,
    }


def predict_single(model_bundle, input_dict):
    row = pd.DataFrame([input_dict])
    row_encoded = pd.get_dummies(row)
    row_final = row_encoded.reindex(columns=model_bundle['columns'], fill_value=0)
    return float(model_bundle['model'].predict(row_final)[0])


@st.cache_resource(show_spinner=False)
def run_segmentation(df_hash: str, df: pd.DataFrame, k: int):
    numeric_cols = [c for c in ['Price'] + NUMERIC_FEATURES if c in df.columns]
    data = df.dropna(subset=numeric_cols).copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data[numeric_cols])

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    data['cluster'] = clusters.astype(str)

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    data['pca_x'] = coords[:, 0]
    data['pca_y'] = coords[:, 1]

    return data, numeric_cols


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

with st.spinner("Nettoyage des données en cours..."):
    try:
        df = load_and_clean(uploaded_file)
    except Exception as e:
        st.error("Impossible de lire ou nettoyer ce fichier.")
        st.info(f"Détail technique : {e}")
        st.stop()

df_hash = str(pd.util.hash_pandas_object(df).sum())

st.sidebar.success(f"✅ {len(df)} lignes chargées et nettoyées")

csv_buffer = df.to_csv(index=False).encode("utf-8")
st.sidebar.download_button(
    "⬇️ Télécharger les données nettoyées (CSV)",
    data=csv_buffer,
    file_name="laptops_cleaned.csv",
    mime="text/csv"
)

# ============================================================
# DASHBOARD
# ============================================================
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

    # ========================================================
    # 🤖 AI INSIGHTS
    # ========================================================
    st.markdown("---")
    st.header("🤖 AI Insights")
    st.caption(
        "⚠️ La colonne `rating` est exclue des modèles : Excel a converti certaines valeurs "
        "en dates lors de l'import (bug de format), ce qui la rend inutilisable telle quelle."
    )

    tab_price, tab_demand, tab_seg = st.tabs(
        ["💰 Prédiction de prix", "📈 Prédiction de demande", "🧩 Segmentation"]
    )

    # --- TAB 1 : PRICE PREDICTION ---
    with tab_price:
        st.write("Modèle : **Random Forest Regressor** — prédit le prix à partir des specs techniques.")

        with st.spinner("Entraînement du modèle de prix..."):
            price_bundle = train_price_model(df_hash, df)

        m1, m2 = st.columns(2)
        m1.metric("R² (test)", f"{price_bundle['r2']:.3f}")
        m2.metric("Erreur moyenne (MAE)", f"${price_bundle['mae']:,.2f}")

        fig_imp = px.bar(
            price_bundle['importances'].head(12)[::-1],
            orientation='h',
            title="Importance des features",
            labels={'value': 'Importance', 'index': 'Feature'}
        )
        st.plotly_chart(fig_imp, use_container_width=True)

        st.markdown("#### 🔮 Simuler un prix")
        c1, c2, c3 = st.columns(3)
        with c1:
            in_ram = st.number_input("RAM (GB)", min_value=1.0,
                                      value=float(df['ram'].median()) if 'ram' in df.columns else 8.0)
            in_brand = st.selectbox("Marque", options=price_bundle['enc_info'].get('brand', ['Other']))
        with c2:
            in_hdd = st.number_input("Stockage (GB)", min_value=1.0,
                                      value=float(df['harddisk'].median()) if 'harddisk' in df.columns else 256.0)
            in_os = st.selectbox("OS", options=price_bundle['enc_info'].get('OS', ['Other']))
        with c3:
            in_screen = st.number_input("Taille écran (pouces)", min_value=1.0,
                                         value=float(df['screen_size'].median()) if 'screen_size' in df.columns else 15.6)
            in_gpu = st.selectbox("Carte graphique", options=price_bundle['enc_info'].get('graphics', ['Other']))

        if st.button("Prédire le prix", type="primary"):
            input_dict = {
                'ram': in_ram, 'harddisk': in_hdd, 'screen_size': in_screen,
                'brand': in_brand, 'OS': in_os, 'graphics': in_gpu,
            }
            if 'cpu_speed' in price_bundle['numeric_cols']:
                input_dict['cpu_speed'] = float(df['cpu_speed'].median())
            pred_price = predict_single(price_bundle, input_dict)
            st.success(f"💰 Prix estimé : **${pred_price:,.2f}**")

    # --- TAB 2 : DEMAND PREDICTION ---
    with tab_demand:
        st.write(
            "Modèle : **Random Forest Regressor** — prédit les ventes totales (`Total Sales`) "
            "attendues pour un produit selon ses specs et son prix. "
            "⚠️ Pas de colonne date disponible → ceci n'est pas une prévision temporelle, "
            "mais une estimation de la demande basée sur des produits similaires."
        )

        with st.spinner("Entraînement du modèle de demande..."):
            demand_bundle = train_demand_model(df_hash, df)

        m1, m2 = st.columns(2)
        m1.metric("R² (test)", f"{demand_bundle['r2']:.3f}")
        m2.metric("Erreur moyenne (MAE)", f"${demand_bundle['mae']:,.2f}")

        fig_imp2 = px.bar(
            demand_bundle['importances'].head(12)[::-1],
            orientation='h',
            title="Importance des features",
            labels={'value': 'Importance', 'index': 'Feature'}
        )
        st.plotly_chart(fig_imp2, use_container_width=True)

        st.markdown("#### 🔮 Simuler une demande")
        d1, d2, d3 = st.columns(3)
        with d1:
            din_ram = st.number_input("RAM (GB) ", min_value=1.0,
                                       value=float(df['ram'].median()) if 'ram' in df.columns else 8.0, key='d_ram')
            din_brand = st.selectbox("Marque ", options=demand_bundle['enc_info'].get('brand', ['Other']), key='d_brand')
        with d2:
            din_hdd = st.number_input("Stockage (GB) ", min_value=1.0,
                                       value=float(df['harddisk'].median()) if 'harddisk' in df.columns else 256.0, key='d_hdd')
            din_os = st.selectbox("OS ", options=demand_bundle['enc_info'].get('OS', ['Other']), key='d_os')
        with d3:
            din_price = st.number_input("Prix ($)", min_value=1.0,
                                         value=float(df['Price'].median()))
            din_gpu = st.selectbox("Carte graphique ", options=demand_bundle['enc_info'].get('graphics', ['Other']), key='d_gpu')

        if st.button("Prédire la demande", type="primary"):
            input_dict = {
                'ram': din_ram, 'harddisk': din_hdd, 'Price': din_price,
                'brand': din_brand, 'OS': din_os, 'graphics': din_gpu,
            }
            if 'screen_size' in demand_bundle['numeric_cols']:
                input_dict['screen_size'] = float(df['screen_size'].median())
            if 'cpu_speed' in demand_bundle['numeric_cols']:
                input_dict['cpu_speed'] = float(df['cpu_speed'].median())
            pred_demand = predict_single(demand_bundle, input_dict)
            st.success(f"📈 Ventes totales estimées : **${pred_demand:,.2f}**")

    # --- TAB 3 : SEGMENTATION ---
    with tab_seg:
        st.write("Modèle : **K-Means** sur Prix + specs numériques, projeté en 2D via **PCA**.")

        k = st.slider("Nombre de segments (clusters)", min_value=2, max_value=8, value=4)

        with st.spinner("Segmentation en cours..."):
            seg_df, seg_numeric_cols = run_segmentation(df_hash, df, k)

        fig_clusters = px.scatter(
            seg_df, x='pca_x', y='pca_y', color='cluster',
            hover_data=['brand', 'model', 'Price'],
            title="Segments de laptops (projection PCA 2D)",
            template="plotly_white"
        )
        st.plotly_chart(fig_clusters, use_container_width=True)

        st.markdown("#### 📋 Profil des segments")
        profile = seg_df.groupby('cluster')[seg_numeric_cols].mean().round(1)
        profile['Nb produits'] = seg_df.groupby('cluster').size()
        st.dataframe(profile, use_container_width=True)

        st.markdown("#### 🏷️ Marques dominantes par segment")
        chosen_cluster = st.selectbox("Choisir un segment", options=sorted(seg_df['cluster'].unique()))
        top_brands = (
            seg_df[seg_df['cluster'] == chosen_cluster]['brand']
            .value_counts().head(10).reset_index()
        )
        top_brands.columns = ['brand', 'count']
        fig_seg_brand = px.bar(top_brands, x='brand', y='count', title=f"Top marques — Segment {chosen_cluster}")
        st.plotly_chart(fig_seg_brand, use_container_width=True)

except Exception as e:
    st.error("Erreur lors de l'affichage du dashboard.")
    st.info(f"Détail technique : {e}")