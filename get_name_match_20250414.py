import psycopg2
import logging
import re
import numpy as np
from rapidfuzz import fuzz
from collections import defaultdict
from gensim.models import KeyedVectors

import nltk
from nltk.corpus import wordnet

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

DB_HOST = "localhost"
DB_NAME = "mealplanning"
DB_USER = "postgres"
DB_PASSWORD = "new-website-app"
DB_PORT = "5432"
FT_PATH = "crawl-300d-2M-subword.vec"

_syn_cache = {}

def get_wordnet_synonyms(word):
    word = word.lower().strip()
    if word in _syn_cache:
        return _syn_cache[word]
    syns = set()
    for synset in wordnet.synsets(word):
        for lemma in synset.lemmas():
            syns.add(lemma.name().lower().strip())
    _syn_cache[word] = syns
    return syns

def get_db():
    return psycopg2.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        port=DB_PORT
    )

##################################################
# LOAD MODEL
##################################################
def load_fasttext_model():
    logging.info(f"Loading FastText from {FT_PATH} ...")
    ft_model = KeyedVectors.load_word2vec_format(FT_PATH, binary=False)
    logging.info("Model loaded.")
    return ft_model

################################################
# LOAD LOCAL SUB EMBEDDINGS
################################################
def fetch_local_sub_embeddings():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT ingredient_id, sub_text, embedding, weight, sub_label
          FROM ingredient_sub_embeddings
         ORDER BY ingredient_id
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()

    local_subs = defaultdict(list)
    for ing_id, txt, emb, w, lbl in rows:
        w_f = float(w)  # fix decimal => float
        emb_np = np.array(emb, dtype='float32')
        local_subs[ing_id].append((txt.lower().strip(), emb_np, w_f, lbl))
    return local_subs

def fetch_usda_sub_embeddings():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT usda_ingredient_id, sub_text, embedding, weight, sub_label
          FROM usda_sub_embeddings
         ORDER BY usda_ingredient_id
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()

    usda_subs = defaultdict(list)
    for uid, txt, emb, w, lbl in rows:
        w_f = float(w)
        emb_np = np.array(emb, dtype='float32')
        usda_subs[uid].append((txt.lower().strip(), emb_np, w_f, lbl))
    return usda_subs

################################################
# PARTIAL MATCH
################################################
def chunk_matches_stored_emb(l_txt, l_emb, l_syns, u_txt, u_emb, ft_model=None, embed_thresh=0.75, fuzzy_thresh=80):
    """
    If we have stored embeddings (l_emb, u_emb). 
    Also do WordNet synonyms, fuzzy, etc.
    We'll do a quick check on first letter or length difference to skip big mismatches.
    """
    # quick skip if first letter not match and length diff > 3
    # (assuming alpha. If numeric, handle differently)
    # This is optional but can help performance.
    if l_txt and u_txt:
        if l_txt[0] != u_txt[0]:
            if abs(len(l_txt) - len(u_txt)) > 3:
                return False

    if l_txt == u_txt:
        return True

    # WordNet synonyms
    if u_txt in l_syns:
        return True

    # fuzzy
    ratio = fuzz.ratio(l_txt, u_txt)
    if ratio >= fuzzy_thresh:
        return True

    # embedding from DB
    if l_emb is not None and u_emb is not None and len(l_emb)>0 and len(u_emb)>0:
        dot = float(np.dot(l_emb, u_emb))
        normA = float(np.linalg.norm(l_emb))
        normB = float(np.linalg.norm(u_emb))
        if normA>0 and normB>0:
            cos_sim = dot/(normA*normB)
            if cos_sim >= embed_thresh:
                return True

    return False

################################################
# WEIGHTED JACCARD
################################################
def weighted_jaccard_stored_emb(local_list, usda_list, ft_model=None):
    """
    local_list => [ (sub_text, embedding_vector, weight, sub_label) ]
    usda_list  => [ (sub_text, embedding_vector, weight, sub_label) ]
    We'll do partial synonyms with chunk_matches_stored_emb(...).
    """
    used_usda = [False]*len(usda_list)
    matched_weight = 0.0

    for (l_txt, l_emb, l_w, l_lbl) in local_list:
        l_syns = get_wordnet_synonyms(l_txt)
        # find best match
        for i, (u_txt, u_emb, u_w, u_lbl) in enumerate(usda_list):
            if not used_usda[i]:
                if chunk_matches_stored_emb(l_txt, l_emb, l_syns, u_txt, u_emb, ft_model):
                    mw = min(l_w, u_w)
                    matched_weight += mw
                    used_usda[i] = True
                    break

    total_local = sum(x[2] for x in local_list)
    total_usda  = sum(x[2] for x in usda_list)
    denom = total_local + total_usda - matched_weight
    if denom<=0:
        return 0.0
    return matched_weight/denom

################################################
# BUILD EXACT TEXT INDEX
################################################
def build_usda_token_index_stored_emb(usda_subs):
    """
    EXACT text => set of usda_ids
    """
    token_idx = defaultdict(set)
    for uid, chunk_list in usda_subs.items():
        for (txt, emb, w, lbl) in chunk_list:
            token_idx[txt].add(uid)
    return token_idx

################################################
# CANDIDATE
################################################
def candidate_usda_ids_stored_emb(local_list, usda_token_index, ft_model):
    """
    For each local chunk, if l_txt in index => union set of IDs
    else do partial synonyms with the index's keys (which can be big).
    We'll do the same "first letter & length" skip for performance.
    """
    cands = set()
    for (l_txt, _, _, _) in local_list:
        if l_txt in usda_token_index:
            cands |= usda_token_index[l_txt]
        else:
            if len(l_txt)>=4:
                # check partial synonyms among the index keys
                # we do a quick skip if first letter different & length difference>3
                for key_txt, idset in usda_token_index.items():
                    # quick skip:
                    if key_txt and (key_txt[0] != l_txt[0]):
                        if abs(len(key_txt) - len(l_txt))>3:
                            continue
                    # do a partial check:
                    if chunk_matches_stored_emb(l_txt, None, get_wordnet_synonyms(l_txt), key_txt, None, ft_model):
                        cands |= idset
    return cands

################################################
# BATCH
################################################
def batch_match_stored_embeddings(local_subs, usda_subs, ft_model=None):
    """
    local_subs => ingredient_id => [ (sub_text, emb_vector, weight, sub_label), ...]
    usda_subs  => usda_id => [ (sub_text, emb_vector, weight, sub_label), ...]
    We'll do the same token-based filter approach with an EXACT index on usda side.
    """
    usda_token_index = build_usda_token_index_stored_emb(usda_subs)
    results = {}

    for local_id, loc_list in local_subs.items():
        cands = candidate_usda_ids_stored_emb(loc_list, usda_token_index, ft_model)
        if not cands:
            results[local_id] = (None, 0.0)
            continue

        best_uid = None
        best_score = -1.0
        for uid in cands:
            sc = weighted_jaccard_stored_emb(loc_list, usda_subs[uid], ft_model)
            if sc>best_score:
                best_score = sc
                best_uid = uid
        results[local_id] = (best_uid, best_score)
    return results

################################################
# MAIN
################################################
if __name__=="__main__":
    logging.info("Will load stored embeddings from sub_embeddings table.")
    # If you still want the model for synonyms?
    ft_model = load_fasttext_model()

    # 1) load local & usda sub embeddings from DB
    local_subs = fetch_local_sub_embeddings()
    usda_subs  = fetch_usda_sub_embeddings()

    # 2) do the batch match
    best_matches = batch_match_stored_embeddings(local_subs, usda_subs, ft_model)

    # 3) optionally fetch the names so we can print
    # e.g. local ID => name
    # usda ID => name
    # We'll do quick approach
    conn = get_db()
    cur = conn.cursor()
    # local names
    local_names = {}
    cur.execute("SELECT id, canonical_name FROM ingredients")
    for r in cur.fetchall():
        local_names[r[0]] = r[1]
    # usda names
    usda_names = {}
    cur.execute("SELECT id, canonical_name FROM usda_ingredients")
    for r in cur.fetchall():
        usda_names[r[0]] = r[1]
    cur.close()
    conn.close()

    # 4) print results
    for local_id, (uid, sc) in list(best_matches.items())[:20]:
        lname = local_names.get(local_id,"?")
        if uid:
            uname = usda_names.get(uid,"?")
            logging.info(f"Local {local_id} '{lname}' => USDA {uid} '{uname}', WeightedJaccard={sc:.2f}")
        else:
            logging.info(f"Local {local_id} '{lname}' => no match found, sc=0.0")
