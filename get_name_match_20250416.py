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
FT_PATH = "models/crawl-300d-2M-subword.vec"

_syn_cache = {}

#############################
#  DB GET
#############################
def get_db_connection():
    return psycopg2.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        port=DB_PORT
    )

#############################
#  STOP CHUNKS, ANTONYMS, DOMAIN WEIGHTS
#############################
def fetch_stop_chunks():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT chunk_text FROM stop_chunks")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return set(row[0].lower().strip() for row in rows)

def fetch_antonyms():
    """
    We store them in a dict: antonyms[a].add(b)
    so if chunk_a => chunk_b, we consider them contradictory.
    """
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT chunk_a, chunk_b FROM antonyms")
    rows = cur.fetchall()
    cur.close()
    conn.close()

    antonyms_dict = defaultdict(set)
    for a,b in rows:
        a_low = a.lower().strip()
        b_low = b.lower().strip()
        antonyms_dict[a_low].add(b_low)
    return antonyms_dict

def fetch_domain_weights():
    """
    chunk_text => multiplier
    e.g. "chicken" => 2.0, "fish" => 1.5
    """
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT chunk_text, weight_multiplier FROM domain_weights")
    rows = cur.fetchall()
    cur.close()
    conn.close()

    dw = {}
    for t,m in rows:
        dw[t.lower().strip()] = float(m)
    return dw

def store_match_result(local_id, usda_id, jaccard_score, explanation=None, fuzzy_score=None, rank=None):
    """
    Insert or update a row in ingredient_usda_matches
      local_ingredient_id = local_id,
      usda_ingredient_id = usda_id,
      jaccard_score = jaccard_score,
      matched_at = now(),
      similarity_explanation = explanation,
      fuzzy_score = fuzzy_score,
      rank = rank
    """
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO ingredient_usda_matches 
          (local_ingredient_id, usda_ingredient_id, jaccard_score, similarity_explanation, fuzzy_score, rank)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (local_ingredient_id, usda_ingredient_id)
        DO UPDATE 
        SET jaccard_score = EXCLUDED.jaccard_score,
            matched_at    = now(),
            similarity_explanation = EXCLUDED.similarity_explanation,
            fuzzy_score   = EXCLUDED.fuzzy_score,
            rank          = EXCLUDED.rank
        """,
        (local_id, usda_id, jaccard_score, explanation, fuzzy_score, rank)
    )
    conn.commit()
    cur.close()
    conn.close()

#############################
#  WORDNET expansions
#############################
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

#############################
#  LOAD MODEL
#############################
def load_fasttext_model():
    logging.info(f"Loading FastText from {FT_PATH} ...")
    ft_model = KeyedVectors.load_word2vec_format(FT_PATH, binary=False)
    logging.info("Model loaded.")
    return ft_model

#############################
#  LOAD EMBEDDINGS FROM DB
#############################
def fetch_local_sub_embeddings():
    """
    local_subs[ingredient_id] => [ (chunk_text, emb, weight, sub_label), ...]
    """
    conn = get_db_connection()
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
        w_f = float(w)
        emb_np = np.array(emb, dtype='float32')
        local_subs[ing_id].append((txt.lower().strip(), emb_np, w_f, lbl))
    return local_subs

def fetch_usda_sub_embeddings():
    """
    usda_subs[usda_id] => [ (chunk_text, emb, weight, sub_label), ...]
    """
    conn = get_db_connection()
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

#############################
#  PARTIAL MATCH with synonyms, fuzzy, embed, antonyms
#############################
def chunk_matches_stored_emb(
    l_txt, l_emb, l_syns, 
    u_txt, u_emb, 
    ft_model=None, 
    embed_thresh=0.75, 
    fuzzy_thresh=80,
    antonyms_dict=None):
    """
    If antonyms found => immediate mismatch.
    Then do exact, synonyms, fuzzy, embed check.
    Quick skip if first letter & length difference big.
    """
    a_low = l_txt
    b_low = u_txt

    # check antonyms:
    if antonyms_dict and a_low in antonyms_dict:
        if b_low in antonyms_dict[a_low]:
            # direct contradiction
            return False

    # quick skip if first letter mismatch & length diff>3
    if a_low and b_low:
        if a_low[0]!=b_low[0]:
            if abs(len(a_low)-len(b_low))>3:
                return False

    if a_low == b_low:
        return True

    if b_low in l_syns:
        return True

    # fuzzy
    ratio = fuzz.ratio(a_low, b_low)
    if ratio >= fuzzy_thresh:
        return True

    # embed
    if l_emb is not None and u_emb is not None and len(l_emb)>0 and len(u_emb)>0:
        dot = float(np.dot(l_emb,u_emb))
        normA = float(np.linalg.norm(l_emb))
        normB = float(np.linalg.norm(u_emb))
        if normA>0 and normB>0:
            cos_sim = dot/(normA*normB)
            if cos_sim>=embed_thresh:
                return True

    return False

#############################
# Weighted Jaccard with domain-coded weighting, stop-chunk penalty, antonyms
#############################
def weighted_jaccard_stored_emb(
    local_list, 
    usda_list, 
    ft_model=None,
    stop_chunks=None, 
    stop_penalty=0.2,
    antonyms_dict=None,
    domain_weights=None
    ):
    """
    local_list => [ (l_txt, l_emb, l_w, l_lbl), ... ]
    usda_list  => [ (u_txt, u_emb, u_w, u_lbl), ... ]

    We'll do partial synonyms with chunk_matches_stored_emb(...).
    We'll also do domain-coded weighting:
      if domain_weights has chunk_text => multiply its weight by domain_weights[chunk_text].
    We'll do leftover penalty if chunk in stop_chunks => multiply leftover by stop_penalty.
    We'll do antonyms => immediate mismatch => 0 or skip.
    """
    used_usda = [False]*len(usda_list)
    matched_weight=0.0

    # apply domain-coded weighting
    # e.g. if domain_weights["chicken"] = 2.0 => multiply that chunk's weight
    # We'll mutate the local_list in memory (or create a copy).
    # For a minimal example, let's do it in place:
    local_copy = []
    for (txt, emb, w, lbl) in local_list:
        mult = domain_weights.get(txt, 1.0) if domain_weights else 1.0
        new_w = w*mult
        local_copy.append((txt, emb, new_w, lbl))

    usda_copy = []
    for (txt, emb, w, lbl) in usda_list:
        mult = domain_weights.get(txt, 1.0) if domain_weights else 1.0
        new_w = w*mult
        usda_copy.append((txt, emb, new_w, lbl))

    for (l_txt, l_emb, l_w, l_lbl) in local_copy:
        l_syns = get_wordnet_synonyms(l_txt)
        for i, (u_txt, u_emb, u_w, u_lbl) in enumerate(usda_copy):
            if not used_usda[i]:
                # check antonyms
                # if chunk_matches => matched
                if chunk_matches_stored_emb(
                    l_txt, l_emb, l_syns,
                    u_txt, u_emb,
                    ft_model=ft_model,
                    antonyms_dict=antonyms_dict
                ):
                    mw = min(l_w,u_w)
                    matched_weight+=mw
                    used_usda[i]=True
                    break

    total_local = sum(x[2] for x in local_copy)

    total_usda = 0.0
    for i, (u_txt, u_emb, u_w, u_lbl) in enumerate(usda_copy):
        if used_usda[i]:
            total_usda+=u_w
        else:
            # leftover => if in stop_chunks => partial penalty
            if stop_chunks and u_txt in stop_chunks:
                total_usda+=stop_penalty*u_w
            else:
                total_usda+=u_w

    denom = total_local + total_usda - matched_weight
    if denom<=0:
        return 0.0
    return matched_weight/denom

#############################
# EXACT TOKEN INDEX
#############################
def build_usda_token_index_stored_emb(usda_subs):
    tindex = defaultdict(set)
    for uid, chunk_list in usda_subs.items():
        for (txt, emb, w, lbl) in chunk_list:
            tindex[txt].add(uid)
    return tindex

def candidate_usda_ids_stored_emb(local_list, usda_token_index, ft_model):
    cands=set()
    for (l_txt, l_emb, l_w, l_lbl) in local_list:
        if l_txt in usda_token_index:
            cands|=usda_token_index[l_txt]
        else:
            if len(l_txt)>=4:
                l_syns = get_wordnet_synonyms(l_txt)
                for key_txt, idset in usda_token_index.items():
                    if key_txt and key_txt[0]!=l_txt[0]:
                        if abs(len(key_txt)-len(l_txt))>3:
                            continue
                    if chunk_matches_stored_emb(l_txt,None,l_syns,key_txt,None,ft_model):
                        cands|=idset
    return cands

def batch_match_stored_embeddings(local_subs, usda_subs, ft_model=None):
    """
    1) Build index of USDA
    2) For each local_id, find best match => (best_uid, best_score)
    3) Store the results in ingredient_usda_matches
    4) Return best_matches dict
    """
    usda_token_index = build_usda_token_index_stored_emb(usda_subs)
    results = {}

    for local_id, loc_list in local_subs.items():
        cands = candidate_usda_ids_stored_emb(loc_list, usda_token_index, ft_model)
        if not cands:
            results[local_id] = (None, 0.0)
            # optional: store a "no match" row or skip
            continue

        best_uid = None
        best_score = -1.0
        for uid in cands:
            sc = weighted_jaccard_stored_emb(loc_list, usda_subs[uid], ft_model)
            if sc>best_score:
                best_score=sc
                best_uid=uid

        # ^ best match found
        results[local_id] = (best_uid, best_score)

        if best_uid:
            # Here we store the Weighted Jaccard in ingredient_usda_matches
            # Possibly pass an explanation or fuzzy score if you have them
            store_match_result(
                local_id=local_id,
                usda_id=best_uid,
                jaccard_score=best_score,
                explanation=None,   # or build some string
                fuzzy_score=None,   # e.g. if you track an avg fuzzy ratio
                rank=1              # if you store top-1. Otherwise store rank in a loop
            )

    return results

##########################
# MAIN
##########################
if __name__=="__main__":
    logging.info("Will load stored embeddings from sub_embeddings table.")
    ft_model = load_fasttext_model()
    local_subs = fetch_local_sub_embeddings()
    usda_subs  = fetch_usda_sub_embeddings()

    logging.info("Running batch match for Weighted Jaccard + storing matches.")
    best_matches = batch_match_stored_embeddings(local_subs, usda_subs, ft_model)

    # If you want to print results
    conn = get_db_connection()
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

    for local_id, (uid, sc) in list(best_matches.items())[:20]:
        lname = local_names.get(local_id,"?")
        if uid:
            uname = usda_names.get(uid,"?")
            logging.info(f"Local {local_id} '{lname}' => USDA {uid} '{uname}', WeightedJaccard={sc:.2f}")
        else:
            logging.info(f"Local {local_id} '{lname}' => no match found, sc=0.0")