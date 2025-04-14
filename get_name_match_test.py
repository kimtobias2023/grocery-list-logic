import psycopg2
import logging
import re
import numpy as np
from rapidfuzz import fuzz
from gensim.models import KeyedVectors
from collections import defaultdict

import nltk
from nltk.corpus import wordnet

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# -------------------------------------------------------
# DATABASE CONFIG
# -------------------------------------------------------
DB_HOST = "172.26.192.1"
DB_NAME = "mealplanning"
DB_USER = "postgres"
DB_PASSWORD = "new-website-app"
DB_PORT = "5432"

#########################################################
# EMBEDDING
#########################################################
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

#########################################################
# WORDNET EXPANSIONS
#########################################################
_syn_cache = {}

def get_wordnet_synonyms(word):
    """
    Return a set of synonyms for 'word' from WordNet, all lowercased.
    We use a global _syn_cache to avoid repeated lookups.
    """
    word = word.lower().strip()
    if word in _syn_cache:
        return _syn_cache[word]

    syns = set()
    for synset in wordnet.synsets(word):
        for lemma in synset.lemmas():
            syns.add(lemma.name().lower().strip())
    _syn_cache[word] = syns
    return syns

#########################################################
# DB
#########################################################
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
    Returns a dict: local_ing_mods[ingredient_id] = {
      "name": <canonical_name>,
      "chunks": [ (chunk_text, weight), ... ]
    }
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

    local_ing_mods = {}
    for ing_id, ing_name, mod_name, wt in rows:
        if ing_id not in local_ing_mods:
            local_ing_mods[ing_id] = {
                "name": ing_name or "",
                "chunks": []
            }
        if mod_name:
            local_ing_mods[ing_id]["chunks"].append(
                (mod_name.lower().strip(), wt if wt else 1.0)
            )

    return local_ing_mods

def fetch_usda_ingredient_modifiers():
    """
    Returns a dict: usda_ing_mods[usda_id] = {
      "name": <canonical_name>,
      "chunks": [ (chunk_text, weight), ... ]
    }
    """
    conn = get_db_connection()
    cur = conn.cursor()

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
            usda_ing_mods[usda_id]["chunks"].append(
                (mod_name.lower().strip(), wt if wt else 1.0)
            )

    cur.close()
    conn.close()
    return usda_ing_mods

#########################################################
# BUILD TOKEN INDEX
#########################################################
def build_usda_token_index(usda_ing_mods):
    """
    EXACT chunk => set of USDA IDs
    """
    token_index = defaultdict(set)
    for usda_id, data in usda_ing_mods.items():
        for (txt, wt) in data["chunks"]:
            token_index[txt].add(usda_id)
    return token_index

#########################################################
# PARTIAL MATCH (CHUNK vs CHUNK)
#########################################################
def chunk_matches(chunk_a, chunk_b, ft_model=None, embed_thresh=0.75, fuzzy_thresh=80):
    """
    1) Exact
    2) WordNet synonyms
    3) Fuzzy
    4) Embedding
    """
    a_low = chunk_a.lower().strip()
    b_low = chunk_b.lower().strip()

    if a_low == b_low:
        return True

    # WordNet synonyms check
    # if b_low in synonyms_of(a_low)
    a_syns = get_wordnet_synonyms(a_low)
    if b_low in a_syns:
        return True

    # fuzzy
    ratio = fuzz.ratio(a_low, b_low)
    if ratio >= fuzzy_thresh:
        return True

    # embedding
    if ft_model:
        vec_a = embed_text_gensim(a_low, ft_model)
        vec_b = embed_text_gensim(b_low, ft_model)
        if vec_a is not None and vec_b is not None:
            dot = np.dot(vec_a, vec_b)
            normA = np.linalg.norm(vec_a)
            normB = np.linalg.norm(vec_b)
            if normA > 0 and normB > 0:
                cos_sim = dot / (normA*normB)
                if cos_sim >= embed_thresh:
                    return True

    return False

#########################################################
# WEIGHTED JACCARD
#########################################################
def weighted_jaccard_score(local_chunks, usda_chunks, ft_model=None):
    """
    Weighted Jaccard approach:
      matched_weight = sum( min(l_w, u_w) ) for each matched chunk
      total_local = sum of local chunk weights
      total_usda  = sum of usda chunk weights

    WeightedJaccard = matched_weight / ( total_local + total_usda - matched_weight )

    partial synonyms => if chunk_matches => we consider them "the same" chunk pair
    We'll greedily match each local chunk to at most one usda chunk.
    """
    usda_used = [False]*len(usda_chunks)

    matched_weight = 0.0
    for (l_txt, l_wt) in local_chunks:
        best_match_idx = None
        for i, (u_txt, u_wt) in enumerate(usda_chunks):
            if not usda_used[i]:
                if chunk_matches(l_txt, u_txt, ft_model):
                    # we match them
                    # accumulate matched_weight => min(l_wt, u_wt)
                    mw = min(l_wt, u_wt)
                    matched_weight += mw
                    usda_used[i] = True
                    break

    total_local = sum(w for (_, w) in local_chunks)
    total_usda  = sum(w for (_, w) in usda_chunks)
    denom = total_local + total_usda - matched_weight
    if denom <= 0:
        return 0.0
    return matched_weight / denom


###########################################
# TOKEN FILTER
###########################################
def candidate_usda_ids_for_local(local_chunks, usda_token_index, ft_model=None):
    """
    We'll do exact index => union. If chunk not found => partial synonyms.
    """
    candidate_ids = set()
    for (l_txt, _) in local_chunks:
        # exact
        if l_txt in usda_token_index:
            candidate_ids |= usda_token_index[l_txt]
        else:
            # partial synonyms approach => might be large if we do it for all chunk_keys
            # We'll do a small check: only if len(l_txt)>3
            if len(l_txt)>=4:
                for ckey, cset in usda_token_index.items():
                    if chunk_matches(l_txt, ckey, ft_model):
                        candidate_ids |= cset
    return candidate_ids

def best_usda_for_local(ing_id, local_data, usda_ing_mods, usda_token_index, ft_model=None):
    """
    local_data => { "name": <str>, "chunks": [ (txt, wt), ...] }
    """
    local_name = local_data["name"]
    local_chunks = local_data["chunks"]
    cands = candidate_usda_ids_for_local(local_chunks, usda_token_index, ft_model)
    if not cands:
        return None, 0.0

    best_id = None
    best_score = -1.0
    for uid in cands:
        usda_name = usda_ing_mods[uid]["name"]
        usda_chunks = usda_ing_mods[uid]["chunks"]
        sc = weighted_jaccard_score(local_chunks, usda_chunks, ft_model)
        if sc > best_score:
            best_score = sc
            best_id = uid

    return best_id, best_score

def batch_match_token_filtered(local_ing_mods, usda_ing_mods, ft_model=None):
    """
    Return a dict local_id => (usda_id, score)
    """
    usda_token_index = build_usda_token_index(usda_ing_mods)
    results = {}
    for local_id, data in local_ing_mods.items():
        b_id, sc = best_usda_for_local(local_id, data, usda_ing_mods, usda_token_index, ft_model)
        results[local_id] = (b_id, sc)
    return results


#############################
# MAIN DEMO
#############################
if __name__ == "__main__":
    # 1) Load model
    logging.info("Loading fasttext model (optional).")
    ft_model = load_fasttext_model("crawl-300d-2M-subword.vec")
    # ft_model = None  # if you skip embedding usage

    # 2) Fetch local + USDA
    logging.info("Fetching local data.")
    local_ing_mods = fetch_local_ingredient_modifiers()
    logging.info("Fetching usda data.")
    usda_ing_mods = fetch_usda_ingredient_modifiers()

    # 3) Batch match with Weighted Jaccard + synonyms + token filtering
    logging.info("Starting batch match with Weighted Jaccard and partial synonyms.")
    best_matches = batch_match_token_filtered(local_ing_mods, usda_ing_mods, ft_model)

    # 4) Print results
    # We'll also show the local ingredient name + USDA name
    for local_id, (u_id, score) in list(best_matches.items())[:10]:
        local_name = local_ing_mods[local_id]["name"]
        if u_id is not None:
            usda_name = usda_ing_mods[u_id]["name"]
            logging.info(f"Local ID={local_id} '{local_name}' => USDA ID={u_id} '{usda_name}', WeightedJaccard={score:.2f}")
        else:
            logging.info(f"Local ID={local_id} '{local_name}' => No match found.")

    logging.info("Done.")
