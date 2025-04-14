import numpy as np
import psycopg2
import logging
from gensim.models import KeyedVectors

# -------------------------------------------------------
# DATABASE CONFIGURATION
# -------------------------------------------------------
DB_CONFIG = {
    'host': '172.26.192.1',
    'database': 'mealplanning',
    'user': 'postgres',
    'password': 'new-website-app',
    'port': '5432'
}

# -------------------------------------------------------
# LOAD FASTTEXT
# -------------------------------------------------------
def load_fasttext_model(path="crawl-300d-2M-subword.vec"):
    logging.info(f"Loading FastText model from {path}...")
    model = KeyedVectors.load_word2vec_format(path, binary=False)
    logging.info("FastText model loaded.")
    return model

# -------------------------------------------------------
# EMBEDDING TEXT
# -------------------------------------------------------
def embed_text_gensim(text, ft_model):
    tokens = text.lower().split()
    vectors = [ft_model[t] for t in tokens if t in ft_model.key_to_index]
    return np.mean(vectors, axis=0) if vectors else None

# -------------------------------------------------------
# LOAD USDA / LOCAL EMBEDDINGS FROM DB
# -------------------------------------------------------
def load_usda_embedding(usda_id):
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("SELECT embedding FROM public.usda_ingredient_embeddings WHERE usda_ingredient_id = %s;", (usda_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    return np.array(row[0], dtype=np.float32) if row else None

def load_local_embedding(ingredient_id):
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("SELECT embedding FROM public.ingredient_embeddings WHERE ingredient_id = %s;", (ingredient_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    return np.array(row[0], dtype=np.float32) if row else None
