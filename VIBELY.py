import streamlit as st
import pandas as pd
import numpy as np
import re
import joblib
from huggingface_hub import hf_hub_download
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="VIBLEY", page_icon="🎧", layout="wide")

# ⬇️ Caricamento dati e modello da Hugging Face
@st.cache_resource
def load_optimized_data():
    try:
        df_path = hf_hub_download(repo_id="MarcoBaiguini/vibely-data", filename="df_with_embeddings_compressed.joblib")
        model_path = hf_hub_download(repo_id="MarcoBaiguini/vibely-data", filename="knn_model_compressed.joblib")

        df = joblib.load(df_path)
        knn = joblib.load(model_path)

        return df, knn
    except Exception as e:
        st.error(f"❌ Errore nel caricamento dei dati: {e}")
        return None, None

with st.spinner("🚀 Caricamento dati e modello..."):
    df, knn_model = load_optimized_data()

if df is None or knn_model is None:
    st.stop()

track_features = ['valence', 'danceability', 'energy']

# ⬇️ Estrai ID da input
def extract_track_ids(text):
    return re.findall(r"https://open\.spotify\.com/track/([A-Za-z0-9]+)", text)

# ⬇️ HEADER
st.markdown("""
<div style='text-align:center;padding:30px 0;'>
  <h1 style='font-size:70px;color:#1DB954;font-weight:bold;'>VIBLEY</h1>
  <p style='font-size:20px;color:#ffffffcc;'>Scopri nuova musica con il potere dell'intelligenza artificiale 🎧</p>
  <hr style='border:1px solid #1DB954;margin-top:20px;'>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([2, 1], gap="large")

with col1:
    st.markdown("### 🎵 Incolla i link delle tue canzoni Spotify")
    user_input = st.text_area("Esempio: https://open.spotify.com/track/...", height=150)

    if st.button("Scansiona Playlist"):
        with st.spinner("🎧 Analisi in corso..."):
            track_ids = extract_track_ids(user_input)
            matched = df[df['spotify_id'].isin(track_ids)]

            if matched.empty:
                st.warning("❗ Nessuna traccia trovata nel database.")
            else:
                st.success(f"✅ {len(matched)} brani trovati.")

                playlist_embeddings = np.array(matched["embedding_list"].tolist())
                mean_embedding = playlist_embeddings.mean(axis=0).reshape(1, -1)

                # Trova i 300 più simili con KNN
                _, indices = knn_model.kneighbors(mean_embedding, n_neighbors=300)
                similar_tracks = df.iloc[indices[0]].copy()
                similar_tracks = similar_tracks[~similar_tracks["spotify_id"].isin(track_ids)]

                # Calcola similarità embedding
                embed_matrix = np.array(similar_tracks["embedding_list"].tolist())
                similar_tracks["sim_embed"] = cosine_similarity(embed_matrix, mean_embedding).flatten()

                # Score finale (solo embedding)
                similar_tracks["final_score"] = similar_tracks["sim_embed"]

                # Ordina e limita a max 2 canzoni per artista
                similar_tracks = similar_tracks.sort_values("final_score", ascending=False)
                recommended = similar_tracks.groupby("artist").head(2)
                recommended = recommended.drop_duplicates(subset="spotify_id").head(10)

                st.markdown("#### 🔮 Suggerimenti consigliati:")
                for _, row in recommended.iterrows():
                    link = f"https://open.spotify.com/track/{row['spotify_id']}"
                    st.markdown(f"""
                    <div style='display:flex;align-items:center;background:#1e1e1e;padding:15px;border-radius:12px;margin-bottom:12px;'>
                        <img src="{row['img']}" alt="cover" style="width:90px;height:90px;border-radius:10px;margin-right:20px;">
                        <div style='flex-grow:1'>
                            <h4 style='color:#1DB954;margin:0;'>{row['name']}</h4>
                            <p style='color:#ccc;margin:0 0 8px;'>{row['artist']}</p>
                            <div style='font-size:14px;color:#aaa;margin-bottom:6px;'>
                                🎯 Valence: {row['valence']:.2f} | 🔥 Energy: {row['energy']:.2f} | 💃 Danceability: {row['danceability']:.2f}
                            </div>
                            <audio controls style='width:100%;'>
                                <source src="{row['preview']}" type="audio/mpeg">
                                Il tuo browser non supporta l'audio HTML5.
                            </audio>
                            <div style='margin-top:5px;'>
                                <a href="{link}" target="_blank" style='color:#1DB954;'>🎧 Ascolta su Spotify</a>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

with col2:
    if 'matched' in locals() and not matched.empty:
        st.markdown("### 👤 Profilo musicale utente")
        avg = matched[track_features].mean()
        interpretations = {
            'valence': ["Introspettivo", "Equilibrato", "Positivo"][int(avg['valence']*3)],
            'energy': ["Rilassato", "Bilanciato", "Energico"][int(avg['energy']*3)],
            'danceability': ["Contemplativo", "Ritmico", "Da ballare"][int(avg['danceability']*3)]
        }
        st.metric("⚡ Valence", f"{avg['valence']:.2f} → {interpretations['valence']}")
        st.metric("💥 Energy", f"{avg['energy']:.2f} → {interpretations['energy']}")
        st.metric("🕺 Danceability", f"{avg['danceability']:.2f} → {interpretations['danceability']}")

st.markdown("""
<hr style='margin-top:50px;border:0.5px solid #444;'>
<p style='text-align:center;color:#777;font-size:14px;'>© 2025 VIBLEY — Scopri musica su misura.</p>
""", unsafe_allow_html=True)
