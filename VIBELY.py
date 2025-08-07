import streamlit as st
import pandas as pd
import pickle
import requests
import io
import numpy as np
from sklearn.neighbors import NearestNeighbors

st.set_page_config(page_title="Vibely - AI Music Recommender", page_icon="🎵", layout="centered")

# ⬇️ Funzione robusta per caricare file pickle da Hugging Face
@st.cache_resource
def load_optimized_data():
    df_url = "https://huggingface.co/MarcoBaiguini/vibely-data/resolve/main/df_with_embeddings.pkl"
    knn_url = "https://huggingface.co/MarcoBaiguini/vibely-data/resolve/main/knn_model.pkl"

    def download_pickle(url):
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()

            content_type = response.headers.get("Content-Type", "")
            if "text/html" in content_type:
                raise ValueError(f"Il contenuto ricevuto da {url} è HTML, non un file binario.")

            return pickle.load(io.BytesIO(response.content))

        except requests.exceptions.RequestException as e:
            st.error(f"❌ Errore HTTP durante il download da {url}:\n{e}")
        except ValueError as ve:
            st.error(f"⚠️ Errore di contenuto:\n{ve}")
        except pickle.UnpicklingError as pe:
            st.error(f"🧨 Errore durante il caricamento pickle:\n{pe}")
        except Exception as ex:
            st.error(f"🚨 Errore generico:\n{ex}")
        return None

    with st.spinner("⏳ Caricamento dei dati..."):
        df = download_pickle(df_url)
        knn_model = download_pickle(knn_url)

    if df is None or knn_model is None:
        st.error("❌ Impossibile caricare i dati. Controlla la connessione o gli URL.")
    return df, knn_model

# ⬇️ Caricamento dati
df, knn_model = load_optimized_data()

# ⬇️ Funzione di suggerimento
def get_recommendations(input_ids, df, knn_model, k=5):
    input_embeddings = df[df['id'].isin(input_ids)]['embedding_list'].tolist()
    if not input_embeddings:
        return []

    input_vector = np.mean(input_embeddings, axis=0).reshape(1, -1)
    distances, indices = knn_model.kneighbors(input_vector, n_neighbors=k + len(input_ids))
    indices = indices[0]
    recommended = df.iloc[indices]
    recommended = recommended[~recommended['id'].isin(input_ids)]
    return recommended.head(k)

# ⬇️ UI
st.title("🎧 Vibely")
st.markdown("Espandi i tuoi orizzonti musicali con l'intelligenza artificiale")

user_input = st.text_area("Incolla qui i link delle tue canzoni da Spotify", height=150)

if st.button("🎵 Scopri nuova musica"):
    links = user_input.strip().split("\n")
    input_ids = [link.split("/")[-1].split("?")[0] for link in links if "spotify.com/track" in link]

    if not input_ids:
        st.warning("⚠️ Inserisci almeno un link valido di brano Spotify.")
    else:
        recommendations = get_recommendations(input_ids, df, knn_model)

        if recommendations is None or len(recommendations) == 0:
            st.warning("😕 Nessun suggerimento trovato. Riprova con altri brani.")
        else:
            st.subheader("✨ Brani consigliati:")
            for _, row in recommendations.iterrows():
                st.markdown(f"- **{row['Track']}** di *{row['Artist']}*")
