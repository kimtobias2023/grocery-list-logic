import psycopg2
import logging
import re
import numpy as np
from rapidfuzz import fuzz
from gensim.models import KeyedVectors

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# -------------------------------------------------------
# DATABASE CONFIG
# -------------------------------------------------------
DB_HOST = "172.26.192.1"
DB_NAME = "mealplanning"
DB_USER = "postgres"
DB_PASSWORD = "new-website-app"
DB_PORT = "5432"

# Load your model (FastText or Word2Vec) if desired
def load_fasttext_model(ft_path="crawl-300d-2M-subword.vec"):
    logging.info(f"Loading FastText model from {ft_path}...")
    ft_model = KeyedVectors.load_word2vec_format(ft_path, binary=False)
    logging.info("FastText model loaded.")
    return ft_model

def embed_text_gensim(text, ft_model):
    """
    For short chunk 'white' or 'fish', just embed the single word.
    If the text is not in the vocab, return None.
    """
    if text in ft_model.key_to_index:
        return ft_model[text]
    return None

####################################
# DB Utility
####################################
def get_db_connection():
    return psycopg2.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        port=DB_PORT
    )

def fetch_local_ingredient_modifiers():
    """
    Return a dictionary: local_ing_mods[ingredient_id] = [ (modifier_text, weight), ... ]
    So we can do a batch approach (no repeated DB calls).
    """
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT im.ingredient_id, m.modifier_name, im.weight
          FROM ingredient_modifiers im
          JOIN modifiers m ON im.modifier_id = m.id
         ORDER BY im.ingredient_id
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()

    local_ing_mods = {}
    for ing_id, mod_name, wt in rows:
        local_ing_mods.setdefault(ing_id, []).append((mod_name.lower().strip(), wt))
    return local_ing_mods

def fetch_usda_ingredient_modifiers():
    """
    Return a dictionary: usda_ing_mods[usda_id] = [ (modifier_text, weight), ... ]
    """
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT uim.usda_ingredient_id, um.modifier_name, uim.weight
          FROM usda_ingredient_modifiers uim
          JOIN usda_modifiers um ON uim.modifier_id = um.id
         ORDER BY uim.usda_ingredient_id
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()

    usda_ing_mods = {}
    for usda_id, mod_name, wt in rows:
        usda_ing_mods.setdefault(usda_id, []).append((mod_name.lower().strip(), wt))
    return usda_ing_mods

###########################################
# PARTIAL MATCH FUNCTION (CHUNK vs CHUNK)
###########################################
def chunk_matches(chunk_a, chunk_b, ft_model=None, embed_thresh=0.75, fuzzy_thresh=80):
    """
    Returns True if chunk_a and chunk_b are considered synonyms/partial matches.
    1) Exact match => True
    2) Fuzzy match => ratio > fuzzy_thresh
    3) Embedding similarity => > embed_thresh
    """
    if chunk_a == chunk_b:
        return True

    # Fuzzy
    ratio = fuzz.ratio(chunk_a, chunk_b)
    if ratio >= fuzzy_thresh:
        return True

    # Embedding
    if ft_model:
        vec_a = embed_text_gensim(chunk_a, ft_model)
        vec_b = embed_text_gensim(chunk_b, ft_model)
        if vec_a is not None and vec_b is not None:
            dot = np.dot(vec_a, vec_b)
            normA = np.linalg.norm(vec_a)
            normB = np.linalg.norm(vec_b)
            if normA > 0 and normB > 0:
                cos_sim = dot / (normA * normB)
                if cos_sim >= embed_thresh:
                    return True

    return False

###########################################
# JACCARD SCORING ON SETS OF CHUNKS
###########################################
def jaccard_score(local_chunks, usda_chunks, ft_model=None):
    """
    local_chunks: list of (chunk_text, weight)
    usda_chunks:  list of (chunk_text, weight)

    We'll treat them as sets (or lists) of chunk_text ignoring weight for the Jaccard measure,
    but we do partial matches with chunk_matches(...).

    Jaccard = (# matched) / (# local + # usda - # matched)
    """
    matched = 0
    # We'll make copies so we can track which usda chunk we matched
    usda_matched_flags = [False]*len(usda_chunks)

    for (lc_text, lc_wt) in local_chunks:
        # Attempt to find a usda chunk that matches
        matched_any = False
        for idx, (uc_text, uc_wt) in enumerate(usda_chunks):
            if not usda_matched_flags[idx]:
                if chunk_matches(lc_text, uc_text, ft_model):
                    matched += 1
                    usda_matched_flags[idx] = True
                    matched_any = True
                    break
        # if no match found => leftover local chunk (nothing special needed here)

    # matched is the total # chunk pairs matched
    local_count = len(local_chunks)
    usda_count = len(usda_chunks)
    # Jaccard formula
    denom = (local_count + usda_count - matched)
    if denom <= 0:
        # edge case => if both sets are empty or somehow matched is bigger
        return 0.0
    jac = matched / denom
    return jac

###########################################
# BATCH MATCH
###########################################
def batch_match_all_local(local_ing_mods, usda_ing_mods, ft_model=None):
    """
    local_ing_mods: dict of ingredient_id => [(chunk_text, weight), ...]
    usda_ing_mods: dict of usda_ingredient_id => [(chunk_text, weight), ...]

    For each local ingredient, compute a jaccard score with each USDA ingredient,
    pick the best. Return a dict best_matches[local_id] = (best_usda_id, best_score).
    """
    best_matches = {}
    usda_ids = list(usda_ing_mods.keys())

    logging.info(f"Starting batch match. local_count={len(local_ing_mods)}, usda_count={len(usda_ids)}")

    for local_id, local_chunks in local_ing_mods.items():
        best_uid = None
        best_score = -1.0

        for uid in usda_ids:
            usda_chunks = usda_ing_mods[uid]
            score = jaccard_score(local_chunks, usda_chunks, ft_model)
            if score > best_score:
                best_score = score
                best_uid = uid

        best_matches[local_id] = (best_uid, best_score)
        logging.debug(f"Local {local_id} => best usda {best_uid}, jaccard={best_score:.2f}")

    return best_matches

#############################
# MAIN DEMO
#############################
if __name__ == "__main__":
    # 1) Load your embedding model if you want synonyms
    logging.info("Loading model (optional).")
    ft_model = load_fasttext_model("crawl-300d-2M-subword.vec")
    # ft_model = None  # if you skip embedding usage

    # 2) Fetch local ingredient chunks, USDA ingredient chunks
    logging.info("Fetching local ingredient chunks.")
    local_ing_mods = fetch_local_ingredient_modifiers()
    logging.info("Fetching USDA ingredient chunks.")
    usda_ing_mods = fetch_usda_ingredient_modifiers()

    # 3) Run a batch match
    logging.info("Running batch match with Jaccard + partial synonyms.")
    best_matches = batch_match_all_local(local_ing_mods, usda_ing_mods, ft_model)

    # 4) Print or store results
    # best_matches[local_id] => (usda_id, jaccard_score)
    # you can store them in a table or do something else
    for local_id, (uid, sc) in best_matches.items():
        logging.info(f"Local {local_id} best USDA => {uid}, jaccard={sc:.2f}")

    logging.info("Done.")
