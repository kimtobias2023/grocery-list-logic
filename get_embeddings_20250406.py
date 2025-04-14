import psycopg2
import logging
from gensim.models import KeyedVectors
import numpy as np
import faiss
from collections import defaultdict

# -------------------------------------------------------
# DATABASE CONFIGURATION
# -------------------------------------------------------
DB_HOST = "172.26.192.1"
DB_NAME = "mealplanning"
DB_USER = "postgres"
DB_PASSWORD = "new-website-app"
DB_PORT = "5432"

# -------------------------------------------------------
# LOGGING CONFIG
# -------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s', force=True)

#############################
#   LOAD FASTTEXT MODEL
#############################
def load_fasttext_model(ft_path="crawl-300d-2M-subword.vec"):
    logging.info(f"Loading FastText model from {ft_path}...")
    ft_model = KeyedVectors.load_word2vec_format(ft_path, binary=False)
    logging.info("FastText model loaded.")
    return ft_model

#############################
#   EMBEDDING HELPERS
#############################
def embed_text_gensim(text, ft_model):
    tokens = text.lower().split()
    vectors = []
    for t in tokens:
        if t in ft_model.key_to_index:  # Gensim 4.x uses key_to_index
            vectors.append(ft_model[t])
    if not vectors:
        return None
    return np.mean(vectors, axis=0)

def cosine_similarity(vec_a, vec_b):
    if vec_a is None or vec_b is None:
        return 0.0
    dot = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)

#############################
#   CORE USDA QUERIES (NO SYNONYMS)
#############################

def get_usda_ingredients_with_modifiers():
    """
    Returns rows of form:
        (usda_id, canonical_name, [modifier1, modifier2, ...])
    where all USDA ingredient modifiers are aggregated into a single list.
    """
    conn = psycopg2.connect(
        host=DB_HOST, database=DB_NAME, user=DB_USER,
        password=DB_PASSWORD, port=DB_PORT
    )
    cur = conn.cursor()
    query = """
    SELECT i.id,
           i.canonical_name,
           COALESCE(array_agg(DISTINCT um.modifier_name), '{}') AS usda_mods
      FROM public.usda_ingredients i
      LEFT JOIN public.usda_ingredient_modifiers uim
             ON i.id = uim.usda_ingredient_id
      LEFT JOIN public.usda_modifiers um
             ON uim.modifier_id = um.id
     GROUP BY i.id, i.canonical_name
     ORDER BY i.id;
    """
    cur.execute(query)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows  # e.g. [(101, 'beef', ['roasted','unsalted']), ...]

def build_token_to_usda_index(usda_rows):
    """
    Build a mapping from each token in canonical_name+modifiers -> set of USDA ids
    """
    index = defaultdict(set)
    for uid, cname, mods in usda_rows:
        tokens = cname.lower().split()
        tokens += [m.lower() for m in mods if m]
        for tok in tokens:
            index[tok].add(uid)
    return index

#############################
#   SUB-EMBEDDINGS
#############################

def store_usda_sub_embedding(usda_id, sub_label, sub_text, embedding, weight=1.0):
    float_list = [float(x) for x in embedding]
    conn = psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER,
                            password=DB_PASSWORD, port=DB_PORT)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO public.usda_sub_embeddings (usda_ingredient_id, sub_label, sub_text, embedding, weight)
        VALUES (%s, %s, %s, %s, %s)
    """, (usda_id, sub_label, sub_text, float_list, weight))
    conn.commit()
    cur.close()
    conn.close()

def get_combined_usda_embedding(usda_id):
    """
    Fetch all sub-embeddings for the given usda_ingredient_id,
    combine them by weight, then L2-normalize.
    """
    conn = psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER,
                            password=DB_PASSWORD, port=DB_PORT)
    cur = conn.cursor()
    cur.execute("""
        SELECT embedding, weight
          FROM public.usda_sub_embeddings
         WHERE usda_ingredient_id = %s
    """, (usda_id,))
    rows = cur.fetchall()
    cur.close()
    conn.close()

    if not rows:
        return None

    sum_vec = None
    for emb_array, w in rows:
        vec = np.array(emb_array, dtype=np.float32)
        weighted = vec * float(w)
        if sum_vec is None:
            sum_vec = weighted
        else:
            sum_vec += weighted

    norm = np.linalg.norm(sum_vec)
    if norm > 0:
        sum_vec /= norm
    return sum_vec

#############################
#   LOCAL SUB-EMBEDDINGS (similar pattern)
#############################

def store_local_sub_embedding(ingredient_id, sub_label, sub_text, embedding, weight=1.0):
    float_list = [float(x) for x in embedding]
    conn = psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER,
                            password=DB_PASSWORD, port=DB_PORT)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO public.ingredient_sub_embeddings (ingredient_id, sub_label, sub_text, embedding, weight)
        VALUES (%s, %s, %s, %s, %s)
    """, (ingredient_id, sub_label, sub_text, float_list, weight))
    conn.commit()
    cur.close()
    conn.close()

def get_combined_local_embedding(ingredient_id):
    conn = psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER,
                            password=DB_PASSWORD, port=DB_PORT)
    cur = conn.cursor()
    cur.execute("""
        SELECT embedding, weight
          FROM public.ingredient_sub_embeddings
         WHERE ingredient_id = %s
    """, (ingredient_id,))
    rows = cur.fetchall()
    cur.close()
    conn.close()

    if not rows:
        return None

    sum_vec = None
    for emb_array, w in rows:
        vec = np.array(emb_array, dtype=np.float32)
        weighted = vec * float(w)
        if sum_vec is None:
            sum_vec = weighted
        else:
            sum_vec += weighted

    norm = np.linalg.norm(sum_vec)
    if norm > 0:
        sum_vec /= norm
    return sum_vec

#############################
#   BUILD FULL NAME (NO SYNONYMS)
#############################
def build_usda_full_name_v2(canonical_name, modifiers):
    base = canonical_name.lower().strip()
    parts = [base]
    if modifiers:
        for m in modifiers:
            if m:
                parts.append(m.lower())
    return " ".join(parts)

#############################
#   RECIPE & LOCAL HELPER
#############################
def get_local_ingredient_name(parsed_ingredient_id):
    if not parsed_ingredient_id:
        return None
    conn = psycopg2.connect(
        host=DB_HOST, database=DB_NAME, user=DB_USER,
        password=DB_PASSWORD, port=DB_PORT
    )
    cur = conn.cursor()
    cur.execute("SELECT canonical_name FROM public.ingredients WHERE id = %s;", (parsed_ingredient_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    if row:
        return row[0]
    return None

def update_recipe_ingredient_with_usda(recipe_ing_id, usda_id, match_score, matched_unit=None):
    match_score = float(match_score)
    conn = psycopg2.connect(
        host=DB_HOST, database=DB_NAME, user=DB_USER,
        password=DB_PASSWORD, port=DB_PORT
    )
    cur = conn.cursor()
    cur.execute("""
        UPDATE public.recipe_ingredients
           SET usda_ingredient_id = %s,
               usda_match_score = %s,
               usda_matched_unit = %s
         WHERE id = %s
    """, (usda_id, match_score, matched_unit, recipe_ing_id))
    conn.commit()
    cur.close()
    conn.close()
    logging.info(f"[RecipeIng {recipe_ing_id}] => USDA {usda_id}, score={match_score:.2f}, unit='{matched_unit}'.")

def get_recipe_ingredients(min_id, max_id):
    conn = psycopg2.connect(
        host=DB_HOST, database=DB_NAME, user=DB_USER,
        password=DB_PASSWORD, port=DB_PORT
    )
    cur = conn.cursor()
    query = """
        SELECT id, recipe_id, ingredient_string, weight, quantity,
               parsed_ingredient_id, unit_id, usda_ingredient_id,
               usda_match_score, usda_matched_unit
          FROM public.recipe_ingredients
         WHERE id BETWEEN %s AND %s;
    """
    cur.execute(query, (min_id, max_id))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows

#############################
#   UNIT / CONVERSION LOOKUP
#############################
def normalize_unit(unit_str):
    mapping = {
        'ounce': 'oz',
        'ounces': 'oz',
    }
    return mapping.get(unit_str.lower(), unit_str)

def get_usda_conversion(usda_ingredient_id, unit):
    conn = psycopg2.connect(
        host=DB_HOST, database=DB_NAME, user=DB_USER,
        password=DB_PASSWORD, port=DB_PORT
    )
    cur = conn.cursor()
    query = """
    SELECT amount, oz, grams 
      FROM public.usda_ingredient_conversion
     WHERE usda_ingredient_id = %s
       AND LOWER(unit) = LOWER(%s)
     LIMIT 1;
    """
    cur.execute(query, (usda_ingredient_id, unit))
    row = cur.fetchone()
    cur.close()
    conn.close()
    return row  # (amount, oz, grams)

#############################
#   GET USDA FULL NAME
#############################
def get_full_usda_name(usda_id):
    """
    Return "canonical + modifiers" for logging.
    """
    rows = get_usda_ingredients_with_modifiers()
    for (uid, can_name, mod_list) in rows:
        if uid == usda_id:
            return build_usda_full_name_v2(can_name, mod_list)
    return ""

#############################
#   FAISS BATCH MATCH
#############################
def batch_faiss_match(local_vecs, faiss_index, usda_ids, threshold=0.75):
    if not local_vecs:
        return []
    matrix = np.vstack(local_vecs).astype("float32")
    faiss.normalize_L2(matrix)
    scores, indices = faiss_index.search(matrix, 1)

    results = []
    for score, idx in zip(scores, indices):
        best_score = score[0]
        best_idx = idx[0]
        if best_score >= threshold:
            results.append((usda_ids[best_idx], float(best_score)))
        else:
            results.append((None, float(best_score)))
    return results

def build_usda_text_map(usda_rows):
    """
    Build a dict: usda_id -> "canonical + modifiers"
    """
    return {
        uid: build_usda_full_name_v2(cname, mods)
        for uid, cname, mods in usda_rows
    }

#############################
#   Candidate Filter
#############################
def candidate_filter_with_tokens(local_tokens, token_index):
    candidate_ids = set()
    for token in local_tokens:
        candidate_ids |= token_index.get(token, set())
    return candidate_ids

def match_ingredient_token_filtered(local_string, ft_model, token_index):
    local_tokens = set(local_string.lower().split())
    candidate_ids = candidate_filter_with_tokens(local_tokens, token_index)
    if not candidate_ids:
        return None, 0.0

    local_vec = embed_text_gensim(local_string, ft_model)
    if local_vec is None:
        return None, 0.0

    best_id = None
    best_score = 0.0
    for usda_id in candidate_ids:
        usda_vec = get_combined_usda_embedding(usda_id)
        if usda_vec is None:
            continue
        score = cosine_similarity(local_vec, usda_vec)
        if score > best_score:
            best_score = score
            best_id = usda_id

    return best_id, best_score

#############################
#   aggregator
#############################
def aggregate_usda_matches_for_recipe(min_id, max_id, ft_model, faiss_index, usda_ids, embedding_threshold=0.75):
    recipe_rows = get_recipe_ingredients(min_id, max_id)
    logging.info(f"Found {len(recipe_rows)} recipe_ingredients between {min_id} and {max_id}.")

    local_rows = []
    local_vecs = []

    for row in recipe_rows:
        rec_id, _, _, _, _, parsed_ingredient_id, unit_id, usda_ing, usda_score, usda_unit = row
        local_name = get_local_ingredient_name(parsed_ingredient_id)
        if not local_name:
            logging.warning(f"[RecipeIng {rec_id}] Missing local name for ID={parsed_ingredient_id}.")
            continue

        local_vec = get_combined_local_embedding(parsed_ingredient_id)
        if local_vec is None:
            local_vec = embed_text_gensim(local_name, ft_model)

        if local_vec is not None:
            local_rows.append((row, local_name, local_vec))
            local_vecs.append(local_vec)

    matches = batch_faiss_match(local_vecs, faiss_index, usda_ids, threshold=embedding_threshold)
    for (row, local_str, _), (matched_usda_id, sim_score) in zip(local_rows, matches):
        rec_id, _, _, _, _, parsed_ingredient_id, unit_id, *_ = row
        if matched_usda_id is None:
            logging.info(f"[RecipeIng {rec_id}] '{local_str}' => No good match (score < {embedding_threshold}).")
            continue

        full_name = get_full_usda_name(matched_usda_id)
        update_recipe_ingredient_with_usda(rec_id, matched_usda_id, sim_score, None)
        logging.info(f"[RecipeIng {rec_id}] '{local_str}' => USDA {matched_usda_id} ('{full_name}'), sim={sim_score:.2f}.")


#############################
#   1) Build & Store sub-embeddings for all USDA items
#############################
def build_all_usda_sub_embeddings(ft_model, weight_canonical=2.0, weight_modifier=1.0):
    """
    For each USDA item, parse canonical name, each modifier,
    embed them separately, store in usda_sub_embeddings with given weights.
    """
    usda_data = get_usda_ingredients_with_modifiers()  # (id, canonical, [mod1, mod2,...])
    count = 0
    for (uid, cname, mods) in usda_data:
        # Build canonical embedding
        can_vec = embed_text_gensim(cname, ft_model)
        if can_vec is not None:
            store_usda_sub_embedding(uid, "canonical", cname, can_vec, weight=weight_canonical)
        # Each modifier
        for i, m in enumerate(mods):
            if m:
                mod_vec = embed_text_gensim(m, ft_model)
                if mod_vec is not None:
                    label = f"modifier{i}"
                    store_usda_sub_embedding(uid, label, m, mod_vec, weight=weight_modifier)
        count += 1
    logging.info(f"Built sub-embeddings for {count} USDA items.")


#############################
#   MAIN
#############################
if __name__ == "__main__":
    ft_model = load_fasttext_model("crawl-300d-2M-subword.vec")

    # 1) Build sub-embeddings for all USDA items:
    build_all_usda_sub_embeddings(ft_model, weight_canonical=2.0, weight_modifier=1.0)

    # 2) Build token index for searching
    usda_rows = get_usda_ingredients_with_modifiers()
    token_index = build_token_to_usda_index(usda_rows)

    # 3) Now test the approach with some strings
    test_strings = ["pepper green bell", "zucchini medium", "banana", "flour g"]
    for s in test_strings:
        best_id, score = match_ingredient_token_filtered(s, ft_model, token_index)
        if best_id:
            logging.info(f"'{s}' => USDA {best_id}, sim={score:.2f}")
        else:
            logging.info(f"'{s}' => No match found.")




