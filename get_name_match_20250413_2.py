import psycopg2
import logging
import re
import numpy as np
from rapidfuzz import fuzz
from gensim.models import KeyedVectors
from collections import defaultdict

# -------------------------------------------------------
# DATABASE CONFIG
# -------------------------------------------------------
DB_HOST = "172.26.192.1"
DB_NAME = "mealplanning"
DB_USER = "postgres"
DB_PASSWORD = "new-website-app"
DB_PORT = "5432"

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

#############################
#   EMBEDDING HELPERS
#############################
def load_fasttext_model(ft_path="crawl-300d-2M-subword.vec"):
    """
    Load a FastText or Word2Vec model in Gensim format if you want partial synonyms.
    """
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
    Returns a dict: local_ing_mods[ingredient_id] = [ (chunk_text, weight), ... ]
    We'll also fetch the ingredient's canonical_name so we can print for debugging.
    """
    conn = get_db_connection()
    cur = conn.cursor()
    # We'll also fetch the canonical_name from local ingredients
    cur.execute("""
        SELECT i.id, i.canonical_name, m.modifier_name, im.weight
          FROM ingredients i
          LEFT JOIN ingredient_modifiers im ON i.id = im.ingredient_id
          LEFT JOIN modifiers m ON im.modifier_id = m.id
         ORDER BY i.id
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()

    # structure: local_ing_mods[ing_id] = {
    #    "name": str,
    #    "chunks": [ (mod_name, weight), ... ]
    # }
    local_ing_mods = {}
    for ing_id, ing_name, mod_name, wt in rows:
        if ing_id not in local_ing_mods:
            local_ing_mods[ing_id] = {
                "name": ing_name or "",
                "chunks": []
            }
        if mod_name:
            local_ing_mods[ing_id]["chunks"].append((mod_name.lower().strip(), wt if wt else 1.0))

    return local_ing_mods

def fetch_usda_ingredient_modifiers():
    """
    Returns a dict: usda_ing_mods[usda_id] = {
       "name": canonical_name,
       "chunks": [ (chunk_text, weight), ...]
    }
    """
    conn = get_db_connection()
    cur = conn.cursor()
    # We'll also fetch canonical_name from usda_ingredients
    cur.execute("""
        SELECT ui.id, ui.canonical_name, um.modifier_name, uim.weight
          FROM usda_ingredients ui
          LEFT JOIN usda_ingredient_modifiers uim ON ui.id = uim.usda_ingredient_id
          LEFT JOIN usda_modifiers um ON uim.modifier_id = um.id
         ORDER BY ui.id
    """)
    rows = cur.fetchall()

    # The above won't work if your schema for bridging is different. If you store bridging in `usda_ingredient_modifiers(uim)`,
    # you have `uim.modifier_id => usda_modifiers`, we do:
    #   SELECT ui.id, ui.canonical_name, um.modifier_name, uim.weight
    #   FROM usda_ingredients ui
    #   JOIN usda_ingredient_modifiers uim ON ui.id = uim.usda_ingredient_id
    #   JOIN usda_modifiers um ON uim.modifier_id = um.id
    # depending on your actual schema. Adjust as needed.
    # We'll do an example that might be more correct:

    cur.execute("""
        SELECT ui.id, ui.canonical_name, um.modifier_name, uim.weight
          FROM usda_ingredients ui
          LEFT JOIN usda_ingredient_modifiers uim ON ui.id = uim.usda_ingredient_id
          LEFT JOIN usda_modifiers um ON uim.modifier_id = um.id
         ORDER BY ui.id
    """)
    rows = cur.fetchall()

    usda_ing_mods = {}
    for usda_id, usda_name, mod_name, wt in rows:
        if usda_id not in usda_ing_mods:
            usda_ing_mods[usda_id] = {
                "name": usda_name or "",
                "chunks": []
            }
        if mod_name:
            usda_ing_mods[usda_id]["chunks"].append((mod_name.lower().strip(), wt if wt else 1.0))

    cur.close()
    conn.close()
    return usda_ing_mods

####################################
#   BUILD A TOKEN -> set(USDA IDs) INDEX
####################################
def build_usda_token_index(usda_ing_mods):
    """
    We'll do an EXACT index from chunk_text => set of USDA IDs
    """
    token_index = defaultdict(set)
    for usda_id, data in usda_ing_mods.items():
        for (txt, wt) in data["chunks"]:
            token_index[txt].add(usda_id)
    return token_index

###########################################
# PARTIAL MATCH FUNCTION (CHUNK vs CHUNK)
###########################################
def chunk_matches(chunk_a, chunk_b, ft_model=None, embed_thresh=0.75, fuzzy_thresh=80):
    """
    If chunk_a == chunk_b => True
    else if fuzzy => ratio >= 80
    else if embed => cos_sim >= 0.75
    else => False
    """
    if chunk_a == chunk_b:
        return True

    ratio = fuzz.ratio(chunk_a, chunk_b)
    if ratio >= fuzzy_thresh:
        return True

    if ft_model:
        vec_a = embed_text_gensim(chunk_a, ft_model)
        vec_b = embed_text_gensim(chunk_b, ft_model)
        if vec_a is not None and vec_b is not None:
            dot = np.dot(vec_a, vec_b)
            normA = np.linalg.norm(vec_a)
            normB = np.linalg.norm(vec_b)
            if normA > 0 and normB > 0:
                cos_sim = dot / (normA*normB)
                if cos_sim >= embed_thresh:
                    return True

    return False

###########################################
# JACCARD
###########################################
def jaccard_score(local_chunks, usda_chunks, ft_model=None):
    matched = 0
    usda_matched = [False]*len(usda_chunks)

    for (l_txt, _) in local_chunks:
        for i, (u_txt, _) in enumerate(usda_chunks):
            if not usda_matched[i]:
                if chunk_matches(l_txt, u_txt, ft_model):
                    matched += 1
                    usda_matched[i] = True
                    break

    local_count = len(local_chunks)
    usda_count = len(usda_chunks)
    denom = local_count + usda_count - matched
    if denom <= 0:
        return 0.0
    return matched / denom

###########################################
# FILTER + MATCH
###########################################
def candidate_usda_ids_for_local(local_chunks, usda_token_index, ft_model=None):
    """
    For each chunk, if chunk in index => union sets
    If chunk not in index, we do partial check with all chunk_keys => can be large.
    """
    candidate_ids = set()
    # We'll do a param to limit partial synonyms (we skip for brevity).
    # We'll do a naive approach: try exact first, else do partial synonyms with chunk_keys.

    for (l_txt, _) in local_chunks:
        if l_txt in usda_token_index:
            candidate_ids |= usda_token_index[l_txt]
        else:
            # partial synonyms approach => check chunk_keys
            # for performance, limit if len(l_txt) >= 4
            if len(l_txt) >= 4:
                for key_txt, usid_set in usda_token_index.items():
                    if chunk_matches(l_txt, key_txt, ft_model):
                        candidate_ids |= usid_set

    return candidate_ids

def best_usda_for_local(ing_id, local_data, usda_ing_mods, usda_token_index, ft_model=None):
    """
    local_data => { "name": <str>, "chunks": [ (txt, wt), ...] }
    returns (best_uid, best_score)
    """
    local_name = local_data["name"]
    local_chunks = local_data["chunks"]
    cands = candidate_usda_ids_for_local(local_chunks, usda_token_index, ft_model)
    if not cands:
        # fallback => no candidate => score=0
        return None, 0.0

    best_id = None
    best_score = -1.0
    for uid in cands:
        usda_name = usda_ing_mods[uid]["name"]
        usda_chunks = usda_ing_mods[uid]["chunks"]
        sc = jaccard_score(local_chunks, usda_chunks, ft_model)
        if sc > best_score:
            best_score = sc
            best_id = uid

    return best_id, best_score

def batch_match_token_filtered(local_ing_mods, usda_ing_mods, ft_model=None):
    usda_token_index = build_usda_token_index(usda_ing_mods)
    best_matches = {}

    for local_id, data in local_ing_mods.items():
        b_id, sc = best_usda_for_local(local_id, data, usda_ing_mods, usda_token_index, ft_model)
        best_matches[local_id] = (b_id, sc)
    return best_matches


#############################
# MAIN DEMO
#############################
if __name__ == "__main__":
    # 1) Load model
    logging.info("Loading fasttext model (optional).")
    ft_model = load_fasttext_model("crawl-300d-2M-subword.vec")
    # ft_model = None  # if you skip embedding

    # 2) Fetch local + USDA
    logging.info("Fetching local data.")
    local_ing_mods = fetch_local_ingredient_modifiers()
    logging.info("Fetching usda data.")
    usda_ing_mods = fetch_usda_ingredient_modifiers()

    # 3) Run batch
    logging.info("Starting batch match.")
    best_matches = batch_match_token_filtered(local_ing_mods, usda_ing_mods, ft_model)

    # 4) Print results
    # We'll also show the local ingredient name + USDA name
    for local_id, (u_id, score) in list(best_matches.items())[:10]:
        local_name = local_ing_mods[local_id]["name"]
        if u_id is not None:
            usda_name = usda_ing_mods[u_id]["name"]
            logging.info(f"Local ID={local_id} '{local_name}' => USDA ID={u_id} '{usda_name}', jaccard={score:.2f}")
        else:
            logging.info(f"Local ID={local_id} '{local_name}' => No match found.")

    logging.info("Done.")


