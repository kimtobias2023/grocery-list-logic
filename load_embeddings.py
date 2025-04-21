import psycopg2
import logging
import numpy as np
from gensim.models import KeyedVectors

##################################################
# CONFIG
##################################################
DB_HOST = "localhost"
DB_NAME = "mealplanning"
DB_USER = "postgres"
DB_PASSWORD = "new-website-app"
DB_PORT = "5432"
FT_PATH = "models/crawl-300d-2M-subword.vec"

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')

##################################################
# LOAD MODEL
##################################################
def load_fasttext_model():
    logging.info(f"Loading FastText from {FT_PATH} ...")
    ft_model = KeyedVectors.load_word2vec_format(FT_PATH, binary=False)
    logging.info("Model loaded.")
    return ft_model

def embed_chunk(text, ft_model):
    text = text.lower().strip()
    if text in ft_model.key_to_index:
        return ft_model[text]
    return None

##################################################
# DB
##################################################
def get_db():
    return psycopg2.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        port=DB_PORT
    )

##################################################
# INGREDIENT EMBEDDINGS
##################################################
def store_local_sub_embeddings(ft_model):
    """
    For each row in ingredient_modifiers, we join to 'modifiers' 
    to get the chunk text, embed it, store in ingredient_sub_embeddings.
    We'll skip if embedding is None.
    """
    conn = get_db()
    cur = conn.cursor()

    # We might want to clear old embeddings first, or do an upsert approach
    # e.g. cur.execute("TRUNCATE TABLE ingredient_sub_embeddings")
    # conn.commit()

    # We'll do a LEFT JOIN so we get all bridging rows
    cur.execute("""
        SELECT im.ingredient_id, m.modifier_name, im.weight
          FROM ingredient_modifiers im
          JOIN modifiers m ON im.modifier_id = m.id
         ORDER BY im.ingredient_id
    """)

    rows = cur.fetchall()
    logging.info(f"Found {len(rows)} local bridging rows to embed.")

    insert_sql = """
    INSERT INTO ingredient_sub_embeddings
    (ingredient_id, sub_label, sub_text, embedding, weight)
    VALUES (%s,%s,%s,%s,%s)
    ON CONFLICT (ingredient_id, sub_label, sub_text)
    DO UPDATE SET embedding = EXCLUDED.embedding,
                weight    = EXCLUDED.weight
    """

    for ing_id, mod_text, w in rows:
        if not mod_text:
            continue
        vec = embed_chunk(mod_text, ft_model)
        if vec is not None:
            vec_list = [float(x) for x in vec]  # convert to python float list
            # sub_label can be e.g. 'modifier' or just 'chunk'
            # sub_text is the chunk text
            cur.execute(insert_sql,
                        (ing_id, "modifier", mod_text, vec_list, w if w else 1.0))

    conn.commit()
    cur.close()
    conn.close()
    logging.info("Done storing local sub embeddings.")

def store_usda_sub_embeddings(ft_model):
    """
    For each row in usda_ingredient_modifiers, join to usda_modifiers,
    embed chunk, store in usda_sub_embeddings.
    """
    conn = get_db()
    cur = conn.cursor()

    # Possibly clear existing
    # cur.execute("TRUNCATE TABLE usda_sub_embeddings")
    # conn.commit()

    cur.execute("""
        SELECT uim.usda_ingredient_id, um.modifier_name, uim.weight
         FROM usda_ingredient_modifiers uim
         JOIN usda_modifiers um ON uim.modifier_id = um.id
    """)
    rows = cur.fetchall()
    logging.info(f"Found {len(rows)} USDA bridging rows to embed.")

    insert_sql = """
        INSERT INTO usda_sub_embeddings
         (usda_ingredient_id, sub_label, sub_text, embedding, weight)
         VALUES (%s, %s, %s, %s, %s)
    """

    for usda_id, mod_text, w in rows:
        if not mod_text:
            continue
        vec = embed_chunk(mod_text, ft_model)
        if vec is not None:
            vec_list = [float(x) for x in vec]
            cur.execute(insert_sql,
                        (usda_id, "modifier", mod_text, vec_list, w if w else 1.0))

    conn.commit()
    cur.close()
    conn.close()
    logging.info("Done storing USDA sub embeddings.")

def main():
    ft_model = load_fasttext_model()
    #store_local_sub_embeddings(ft_model)
    store_usda_sub_embeddings(ft_model)

if __name__ == "__main__":
    main()
