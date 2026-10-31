from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
import joblib
import streamlit as st

ROOT = Path(__file__).parent
DATA, ART, RESULTS = ROOT / "data", ROOT / "artifacts", ROOT / "results"

# If the Colab export ZIP is uploaded to the repository root, unpack it on startup.
EXPORT_ZIP = ROOT / "StayWise_Colab_Outputs.zip"
if EXPORT_ZIP.exists() and not (DATA / "hotels_catalog.csv").exists():
    with zipfile.ZipFile(EXPORT_ZIP) as archive:
        archive.extractall(ROOT)

st.set_page_config(page_title="StayWise | Hotel Recommendations", page_icon="🏨", layout="wide")

@st.cache_data
def load_catalog():
    p = DATA / "hotels_catalog.csv"
    if not p.exists():
        return pd.DataFrame(columns=["hotel_id", "name", "location"])
    d = pd.read_csv(p)
    for c in ["hotel_id", "name", "location"]:
        if c not in d: d[c] = ""
    d["hotel_id"] = d["hotel_id"].astype(str)
    d["name"] = d["name"].fillna("Unnamed hotel")
    d["location"] = d["location"].fillna("Unknown")
    return d

@st.cache_resource
def load_artifacts():
    required = ["metadata.joblib", "collaborative_arrays.joblib", "hotel_content_matrix.joblib"]
    missing = [x for x in required if not (ART / x).exists()]
    if missing: return None, None, None, missing
    m = joblib.load(ART / "metadata.joblib")
    c = joblib.load(ART / "collaborative_arrays.joblib")
    x = joblib.load(ART / "hotel_content_matrix.joblib")
    return m, c, x, []

@st.cache_data
def load_csv(path):
    return pd.read_csv(path) if path.exists() else pd.DataFrame()

catalog = load_catalog()
meta, collab, content, missing = load_artifacts()
st.sidebar.title("🏨 StayWise")
page = st.sidebar.radio("Navigate", ["Home", "Find Hotels", "Personalized Recommendations", "Similar Hotels", "Model Performance", "About"])

if page == "Home":
    st.title("StayWise")
    st.subheader("Hotel recommendations powered by machine learning")
    a,b,c = st.columns(3)
    a.metric("Hotels in catalogue", f"{len(catalog):,}")
    b.metric("Recommendation methods", "5")
    c.metric("Dataset", "ViHoRec")
    st.info("Academic research prototype. Recommendations use historical interactions; they do not represent live room availability, prices, or booking inventory.")

elif page == "Find Hotels":
    st.title("Find Hotels")
    q = st.text_input("Hotel name contains")
    locations = ["All locations"] + sorted(catalog.location.dropna().astype(str).unique().tolist())
    loc = st.selectbox("Location", locations)
    view = catalog.copy()
    if q: view = view[view.name.str.contains(q, case=False, na=False)]
    if loc != "All locations": view = view[view.location == loc]
    st.caption(f"{len(view):,} hotels")
    st.dataframe(view[["name","location","hotel_id"]], use_container_width=True, hide_index=True)

elif page == "Personalized Recommendations":
    st.title("Personalized Recommendations")
    if missing:
        st.warning("Model files are not yet in the repository: " + ", ".join(missing))
    else:
        users = [str(x) for x in meta["users"]]
        uid = st.selectbox("Select anonymized dataset user", users)
        method = st.selectbox("Algorithm", ["Hybrid", "User-based CF", "Matrix factorization", "Content TF-IDF", "Popularity"])
        top_n = st.slider("Number of recommendations", 5, 20, 10)
        if st.button("Recommend", type="primary"):
            items = [str(x) for x in meta["items"]]
            ui = {str(k): int(v) for k,v in meta["user_idx"].items()}
            ii = {str(k): int(v) for k,v in meta["item_idx"].items()}
            seen = {str(k): set(map(str,v)) for k,v in meta["seen"].items()}
            R = collab["rating_matrix"]
            u = ui[uid]
            def scale(z):
                z = np.asarray(z, dtype=float)
                lo, hi = np.nanmin(z), np.nanmax(z)
                return (z-lo)/(hi-lo) if hi > lo else np.zeros_like(z)
            cf = np.asarray(collab["user_similarity"][u]).copy()
            cf[u] = 0
            den = np.asarray((R != 0).T.dot(np.abs(cf))).ravel()
            num = np.asarray(R.T.dot(cf)).ravel()
            cf_scores = np.divide(num, den, out=np.zeros_like(num,dtype=float), where=den>1e-12)
            mf_scores = np.asarray(collab["item_factors"]) @ np.asarray(collab["user_factors"])[u]
            row = R.getrow(u)
            profile = row.dot(content).toarray().ravel()
            norm = np.linalg.norm(profile)
            if norm: profile /= norm
            cb_scores = np.asarray(content @ profile).ravel()
            pop_scores = np.asarray(R.sum(axis=0)).ravel()
            if method == "User-based CF": scores = cf_scores
            elif method == "Matrix factorization": scores = mf_scores
            elif method == "Content TF-IDF": scores = cb_scores
            elif method == "Popularity": scores = pop_scores
            else:
                alpha = float(meta.get("best_alpha", 0.25))
                scores = alpha * scale(cf_scores) + (1-alpha) * scale(cb_scores)
            scores = np.asarray(scores,dtype=float).copy()
            for hid in seen.get(uid,set()):
                if hid in ii: scores[ii[hid]] = -np.inf
            rank = np.argsort(scores)[::-1][:top_n]
            recs = pd.DataFrame({"hotel_id":[items[i] for i in rank], "score":[scores[i] for i in rank]})
            st.dataframe(recs.merge(catalog,on="hotel_id",how="left")[["name","location","score"]], use_container_width=True, hide_index=True)
            st.caption("User IDs are pseudonymous dataset identifiers and are not personal identities.")

elif page == "Similar Hotels":
    st.title("Similar Hotels")
    if catalog.empty or missing:
        st.warning("The hotel catalogue and model artifacts must be uploaded before similarity search is available.")
    else:
        options = catalog.hotel_id.tolist()
        chosen = st.selectbox("Choose a hotel", options, format_func=lambda h: f"{catalog.loc[catalog.hotel_id==h,'name'].iloc[0]} — {catalog.loc[catalog.hotel_id==h,'location'].iloc[0]}")
        n = st.slider("Number of similar hotels", 5, 20, 10, key="sim_n")
        if st.button("Find similar hotels", type="primary"):
            items = [str(x) for x in meta["items"]]
            ii = {str(k): int(v) for k,v in meta["item_idx"].items()}
            idx = ii[str(chosen)]
            sims = np.asarray(content @ content[idx].T).ravel()
            sims[idx] = -np.inf
            order = np.argsort(sims)[::-1][:n]
            out = pd.DataFrame({"hotel_id":[items[i] for i in order], "similarity":[sims[i] for i in order]})
            st.dataframe(out.merge(catalog,on="hotel_id",how="left")[["name","location","similarity"]],use_container_width=True,hide_index=True)

elif page == "Model Performance":
    st.title("Model Performance")
    results = load_csv(RESULTS / "model_comparison.csv")
    if results.empty:
        st.warning("Upload the experiment result CSV files to the results/ folder.")
    else:
        st.dataframe(results, use_container_width=True, hide_index=True)
        if "ndcg@10" in results:
            st.bar_chart(results.set_index("model")[["ndcg@10"]])
        best = results.sort_values("ndcg@10",ascending=False).iloc[0]
        st.success(f"Highest NDCG@10 in this experiment: {best['model']} ({best['ndcg@10']:.5f}).")
        prep = load_csv(RESULTS / "preprocessing_comparison.csv")
        sens = load_csv(RESULTS / "rating_ge_7_sensitivity.csv")
        with st.expander("Preprocessing comparison"): st.dataframe(prep,use_container_width=True,hide_index=True)
        with st.expander("Rating ≥ 7 sensitivity analysis"): st.dataframe(sens,use_container_width=True,hide_index=True)
        st.caption("The current preprocessing ablation compares two interaction-data cleaning variants; it does not isolate the effect of every preprocessing technique.")

elif page == "About":
    st.title("About StayWise")
    st.write("StayWise is a non-commercial academic hotel recommendation prototype using the ViHoRec dataset.")
    st.markdown("**Methods:** popularity baseline, content TF-IDF, user-based collaborative filtering, matrix factorization, and a hybrid recommender.\n\n**Metrics:** Precision@10, Recall@10, Hit Rate@10, NDCG@10, and catalogue coverage.")
    st.warning("ViHoRec is licensed CC BY-NC 4.0. Use this dataset and derived materials only within its non-commercial research permissions. StayWise does not show live availability or prices.")
