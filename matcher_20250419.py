#!/usr/bin/env python
"""
Match every *local* ingredient to the best USDA ingredient.

• reads pre‑computed embeddings in  ingredient_sub_embeddings / usda_sub_embeddings
• uses stop‑chunks / antonyms / domain‑weights that live in the DB
• writes the top‑1 result to  ingredient_usda_matches  (UPSERT)
• prints the first few matches for manual inspection
"""
import psycopg2, logging, numpy as np, re
from collections import defaultdict
from rapidfuzz import fuzz
from gensim.models import KeyedVectors
from nltk.corpus import wordnet                # nltk.download('wordnet') once

#####################################################################
# CONFIG
#####################################################################
DB = dict(host="localhost", dbname="mealplanning",
          user="postgres",  password="new-website-app", port=5432)
FT_PATH = "models/crawl-300d-2M-subword.vec"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s: %(message)s")

_syn_cache: dict[str,set[str]] = {}            # WordNet cache
#####################################################################
# DB helpers
#####################################################################
def db():
    return psycopg2.connect(**DB)

#####################################################################
# One‑shot lookup tables – we fetch them **once** at start‑up 🔹
#####################################################################
with db() as _conn, _conn.cursor() as cur:
    # stop‑chunks
    cur.execute("SELECT chunk_text FROM stop_chunks")
    STOP_CHUNKS: set[str] = {r[0].lower().strip() for r in cur.fetchall()}

    # antonyms
    cur.execute("SELECT chunk_a, chunk_b FROM antonyms")
    ANTONYM: dict[str,set[str]] = defaultdict(set)
    for a,b in cur.fetchall():
        ANTONYM[a.lower().strip()].add(b.lower().strip())

    # domain weights
    cur.execute("SELECT chunk_text, weight_multiplier FROM domain_weights")
    DOMAIN_W: dict[str,float] = {t.lower().strip(): float(m) for t,m in cur.fetchall()}

#####################################################################
# WordNet synonyms  (cached)
#####################################################################
def syns(word: str) -> set[str]:
    w = word.lower().strip()
    if w not in _syn_cache:
        _syn_cache[w] = {l.name().lower() for s in wordnet.synsets(w)
                                            for l in s.lemmas()}
    return _syn_cache[w]

#####################################################################
# FastText    (loads once – a few minutes)
#####################################################################
def load_ft():
    logging.info("Loading FastText vectors …")
    model = KeyedVectors.load_word2vec_format(FT_PATH, binary=False)
    logging.info("FastText ready.")
    return model

FT = load_ft()                                  # 🔹

#####################################################################
# Read stored SUB‑embeddings
#####################################################################
def load_sub_embeddings(table: str, id_col: str) -> dict[int,list]:
    sql = f"""
       SELECT {id_col}, sub_text, embedding, weight
       FROM   {table}
       ORDER  BY {id_col}
    """
    data = defaultdict(list)
    with db() as conn, conn.cursor() as cur:
        cur.execute(sql)
        for _id, txt, emb, w in cur.fetchall():
            data[_id].append((txt.lower().strip(),
                              np.array(emb, dtype='float32'),
                              float(w)))
    return data

LOCAL = load_sub_embeddings("ingredient_sub_embeddings", "ingredient_id")
USDA  = load_sub_embeddings("usda_sub_embeddings",        "usda_ingredient_id")

#####################################################################
#  low‑level chunk‑match
#####################################################################
def chunks_match(a_txt, a_vec, b_txt, b_vec,
                 embed_thr=.75, fuzz_thr=80) -> bool:

    # quick reject: antonyms 🔹
    if b_txt in ANTONYM.get(a_txt, ()):          # salted vs unsalted etc.
        return False

    if a_txt == b_txt:
        return True
    if b_txt in syns(a_txt):
        return True
    if fuzz.ratio(a_txt, b_txt) >= fuzz_thr:
        return True
    if (a_vec is not None and b_vec is not None and
        (na:=np.linalg.norm(a_vec)) and (nb:=np.linalg.norm(b_vec)) and
        np.dot(a_vec, b_vec)/(na*nb) >= embed_thr):
        return True
    return False

#####################################################################
# weighted‑Jaccard
#####################################################################
def wjacc(local, usda) -> float:
    used = [False]*len(usda)
    match_w = 0.0

    # apply domain weights 🔹
    loc  = [(t,v, w*DOMAIN_W.get(t,1.0)) for t,v,w in local]
    usda = [(t,v, w*DOMAIN_W.get(t,1.0)) for t,v,w in usda]

    for lt, lv, lw in loc:
        for i,(ut, uv, uw) in enumerate(usda):
            if used[i]:
                continue
            if chunks_match(lt, lv, ut, uv):
                match_w += min(lw, uw)
                used[i] = True
                break

    total_l = sum(w for _,_,w in loc)
    total_u = 0.0
    for (ut,_,uw), flag in zip(usda, used):
        if flag:                      # matched
            total_u += uw
        else:                         # leftover
            pen = .2 if ut in STOP_CHUNKS else 1.0
            total_u += pen * uw

    denom = total_l + total_u - match_w
    return 0.0 if denom<=0 else match_w/denom

#####################################################################
# Build EXACT token index for fast candidate shortlist
#####################################################################
TOK_IDX: dict[str,set[int]] = defaultdict(set)
for uid, chs in USDA.items():
    for t,_,_ in chs:
        TOK_IDX[t].add(uid)

def candidates(local_chunks):
    cset = set()
    for t,_,_ in local_chunks:
        if t in TOK_IDX:
            cset |= TOK_IDX[t]
        else:
            if len(t) < 4:                       # too short => skip fuzzy
                continue
            sy = syns(t)
            for key in sy & TOK_IDX.keys():
                cset |= TOK_IDX[key]
    return cset

#####################################################################
# UPSERT helper
#####################################################################
UPSERT_SQL = """
INSERT INTO ingredient_usda_matches
  (local_ingredient_id, usda_ingredient_id, jaccard_score, rank)
VALUES (%s,%s,%s,%s)
ON CONFLICT (local_ingredient_id, usda_ingredient_id)
DO UPDATE SET jaccard_score=EXCLUDED.jaccard_score,
              matched_at   =now(),
              rank         =EXCLUDED.rank;
"""

def save_match(local_id, pairs):                 # pairs already sorted by score desc
    with db() as conn, conn.cursor() as cur:
        for rank,(uid,score) in enumerate(pairs, start=1):
            cur.execute(UPSERT_SQL, (local_id, uid, round(score,3), rank))
        conn.commit()

#####################################################################
# MAIN batch
#####################################################################
def run(top_k=3):
    logging.info("=== matching starts ===")
    for lid, loc_chunks in LOCAL.items():
        cands = candidates(loc_chunks)
        if not cands:
            continue
        scored = [(uid, wjacc(loc_chunks, USDA[uid])) for uid in cands]
        scored = [(u,s) for u,s in scored if s>0.0]
        if not scored:
            continue
        best = sorted(scored, key=lambda x: -x[1])[:top_k]
        results[lid] = best
        save_match(lid, best)                   # 🔹 write to DB

    logging.info("matching finished.")

# ---------------- public API -----------------
def best_usda_for_local(local_id: int) -> tuple[int | None, float]:
    """
    Return (usda_id, score) for ONE local ingredient id.
    Uses the in‑memory LOCAL / USDA dictionaries – so it’s fast.
    """
    local_chunks = LOCAL.get(local_id)
    if not local_chunks:
        return None, 0.0

    cand_ids = candidates(local_chunks)
    if not cand_ids:
        return None, 0.0

    best_uid, best_sc = max(
        ((uid, wjacc(local_chunks, USDA[uid])) for uid in cand_ids),
        key=lambda x: x[1]
    )
    return best_uid, best_sc

def best_usda_for_text(raw_text: str) -> tuple[int | None, float]:
    # treat the whole string as a single chunk with weight 1
    tmp_id = -abs(hash(raw_text))          # unique negative key
    LOCAL[tmp_id] = [(raw_text.lower(), None, 1.0)]
    return best_usda_for_local(tmp_id)


# what other modules can import ----------------------------------------------
__all__ = ["FT", "LOCAL", "USDA",
           "best_usda_for_local", "batch_match_stored_embeddings", "best_usda_for_text"]

batch_match_stored_embeddings = run          # keep the old name for compatibility
# -----------------------------------------------------------------------------

#####################################################################
# run & pretty print first 10
#####################################################################

if __name__ == "__main__":
    results = {}
    run()

    with db() as conn, conn.cursor() as cur:
        cur.execute("SELECT id, canonical_name FROM ingredients")
        LNAME = {i:n for i,n in cur.fetchall()}
        cur.execute("SELECT id, canonical_name FROM usda_ingredients")
        UNAME = {i:n for i,n in cur.fetchall()}

    for lid, lst in list(results.items())[:20]:
        for uid,sc in lst[:1]:                      # show top‑1
            logging.info(f"[{lid}] {LNAME.get(lid)}  →  "
                        f"[{uid}] {UNAME.get(uid)}   ({sc:.2f})")
        


