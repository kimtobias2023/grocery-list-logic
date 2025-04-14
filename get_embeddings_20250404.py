import psycopg2
import logging
from gensim.models import KeyedVectors
import numpy as np
import faiss


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
    """
    Loads a pre-trained FastText model in .vec (text) format using Gensim.
    Make sure you have the correct path/file name.
    """
    logging.info(f"Loading FastText model from {ft_path}...")
    ft_model = KeyedVectors.load_word2vec_format(ft_path, binary=False)
    logging.info("FastText model loaded.")
    return ft_model

#############################
#   EMBEDDING HELPERS
#############################
def embed_text_gensim(text, ft_model):
    """
    Returns the average embedding vector of the words in `text`.
    If a token is not in the vocabulary, it will be skipped.
    """
    tokens = text.lower().split()
    vectors = []
    for t in tokens:
        if t in ft_model.key_to_index:  # Gensim 4.x uses key_to_index
            vectors.append(ft_model[t])
    if not vectors:
        return None
    # Average them
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
#   GET/STORE USDA DATA
#############################
def get_usda_ingredients():
    """
    Returns list of (usda_id, canonical_name).
    """
    conn = psycopg2.connect(
        host=DB_HOST, database=DB_NAME, user=DB_USER,
        password=DB_PASSWORD, port=DB_PORT
    )
    cur = conn.cursor()
    query = "SELECT id, canonical_name FROM public.usda_ingredients;"
    cur.execute(query)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows  # [(id, 'milk'), (id, 'pepper'), ...]

def get_ingredient_modifiers():
    """
    Retrieves a mapping of local ingredient_id -> list of modifier names.
    Example return:
        {
          201: ["chopped", "fresh"],
          202: ["minced"],
          ...
        }
    """
    conn = psycopg2.connect(
        host=DB_HOST, database=DB_NAME, user=DB_USER,
        password=DB_PASSWORD, port=DB_PORT
    )
    cur = conn.cursor()
    query = """
    SELECT im.ingredient_id, m.modifier_name
      FROM public.ingredient_modifiers im
      JOIN public.modifiers m ON im.modifier_id = m.id;
    """
    cur.execute(query)
    rows = cur.fetchall()
    cur.close()
    conn.close()

    modifiers_map = {}
    for ing_id, modifier_name in rows:
        modifiers_map.setdefault(ing_id, []).append(modifier_name.lower())
    return modifiers_map

def get_usda_ingredient_modifiers():
    """
    Retrieves a mapping of usda_ingredient_id -> list of modifier names.
    Example return:
        {
          101: ["roasted", "unsalted"],
          102: ["low sodium"],
          ...
        }
    """
    conn = psycopg2.connect(
        host=DB_HOST, database=DB_NAME, user=DB_USER,
        password=DB_PASSWORD, port=DB_PORT
    )
    cur = conn.cursor()
    query = """
    SELECT uim.usda_ingredient_id, um.modifier_name
      FROM public.usda_ingredient_modifiers uim
      JOIN public.usda_modifiers um ON uim.modifier_id = um.id;
    """
    cur.execute(query)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    modifiers_map = {}
    for usda_ing_id, modifier_name in rows:
        modifiers_map.setdefault(usda_ing_id, []).append(modifier_name.lower())
    return modifiers_map

def fetch_synonym_map():
    """
    Returns a mapping like:
    { "g flour": "besan", "chickpea flour": "besan" }
    """
    conn = psycopg2.connect(
        host=DB_HOST, database=DB_NAME, user=DB_USER,
        password=DB_PASSWORD, port=DB_PORT
    )
    cur = conn.cursor()
    cur.execute("SELECT synonym, ingredient_name FROM public.ingredient_synonyms;")
    rows = cur.fetchall()
    cur.close()
    conn.close()

    return {syn.lower(): name.lower() for syn, name in rows}


#############################
#   OPTIONAL: STORING USDA EMBEDDINGS IN DB
#############################
def store_usda_embedding(usda_id, embedding):
    """
    Insert/Update the embedding in public.usda_ingredient_embeddings (float[]).
    """
    # Convert from numpy.float32 to native Python float
    float_list = [float(x) for x in embedding]

    conn = psycopg2.connect(
        host=DB_HOST, database=DB_NAME, user=DB_USER,
        password=DB_PASSWORD, port=DB_PORT
    )
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO public.usda_ingredient_embeddings (usda_ingredient_id, embedding)
        VALUES (%s, %s)
        ON CONFLICT (usda_ingredient_id) DO UPDATE
        SET embedding = EXCLUDED.embedding;
    """, (usda_id, float_list))
    conn.commit()
    cur.close()
    conn.close()


def load_usda_embedding(usda_id):
    """
    Retrieve the embedding float[] from usda_ingredient_embeddings.
    Returns a numpy array or None if not found.
    """
    conn = psycopg2.connect(
        host=DB_HOST, database=DB_NAME, user=DB_USER,
        password=DB_PASSWORD, port=DB_PORT
    )
    cur = conn.cursor()
    cur.execute("""
        SELECT embedding 
          FROM public.usda_ingredient_embeddings
         WHERE usda_ingredient_id = %s;
    """, (usda_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    if row:
        return np.array(row[0], dtype=np.float32)
    return None

#############################
#   PRECOMPUTE ALL USDA EMBEDDINGS
#############################
def build_usda_full_name(usda_id, canonical, modifiers_map):
    """
    Given the USDA ingredient ID, its canonical name, and a dict that maps
    usda_id -> list of modifier strings, return a combined text.
    """
    parts = [canonical.lower()]
    if usda_id in modifiers_map:
        for mod in modifiers_map[usda_id]:
            parts.append(mod)
    return " ".join(parts)

def precompute_usda_embeddings(ft_model, store_in_db=False):
    """
    Loads all USDA ingredients and merges in any modifiers for embedding.
    Returns a dict: { usda_id: numpy_vector }
    """
    usda_data = get_usda_ingredients()  # returns [(id, canonical_name), ...]
    modifiers_map = get_usda_ingredient_modifiers()  # {usda_id: ["roasted", "unsalted"], ...}
    usda_embeddings = {}
    for usda_id, canonical_name in usda_data:
        full_text = build_usda_full_name(usda_id, canonical_name, modifiers_map)
        vec = embed_text_gensim(full_text, ft_model)
        usda_embeddings[usda_id] = vec
        if store_in_db and vec is not None:
            store_usda_embedding(usda_id, vec)
    return usda_embeddings

def build_faiss_index(usda_embeddings):
    """
    Builds a FAISS index and returns (index, usda_ids).
    """
    usda_ids = list(usda_embeddings.keys())
    vectors = [usda_embeddings[uid] for uid in usda_ids if usda_embeddings[uid] is not None]
    matrix = np.vstack(vectors).astype('float32')

    faiss.normalize_L2(matrix)  # cosine similarity
    index = faiss.IndexFlatIP(matrix.shape[1])  # inner product = cosine similarity
    index.add(matrix)
    return index, usda_ids


#############################
#   RECIPE ING HELPER 
#############################
def build_local_full_name(ingredient_id, canonical, local_modifiers_map, synonym_map=None):
    """
    Returns combined ingredient name + modifiers, replacing synonyms if found.
    """
    canonical = canonical.lower()

    # Apply synonym override if match
    if synonym_map:
        canonical = synonym_map.get(canonical, canonical)

    parts = [canonical]
    if ingredient_id in local_modifiers_map:
        for mod in local_modifiers_map[ingredient_id]:
            parts.append(mod)
    return " ".join(parts)

def get_local_ingredient_name(parsed_ingredient_id):
    """
    Returns the canonical_name from the local 'ingredients' table, given an ID.
    """
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
    """
    Update recipe_ingredients with the matched USDA ingredient data:
      - usda_ingredient_id
      - usda_match_score
      - usda_matched_unit
    """
    match_score = float(match_score)  # 👈 Convert numpy.float32 to native float

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
    """
    Fetch recipe_ingredients rows from the DB for the given ID range.
    Returns list of (id, recipe_id, ingredient_string, weight, quantity,
                     parsed_ingredient_id, unit_id, usda_ingredient_id,
                     usda_match_score, usda_matched_unit).
    """
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

def store_local_embedding(ingredient_id, embedding):
    float_list = [float(x) for x in embedding]
    conn = psycopg2.connect(
        host=DB_HOST, database=DB_NAME, user=DB_USER,
        password=DB_PASSWORD, port=DB_PORT
    )
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO public.ingredient_embeddings (ingredient_id, embedding)
        VALUES (%s, %s)
        ON CONFLICT (ingredient_id) DO UPDATE
        SET embedding = EXCLUDED.embedding;
    """, (ingredient_id, float_list))
    conn.commit()
    cur.close()
    conn.close()

def load_local_embedding(ingredient_id):
    """
    Retrieve embedding from cache if available.
    """
    conn = psycopg2.connect(
        host=DB_HOST, database=DB_NAME, user=DB_USER,
        password=DB_PASSWORD, port=DB_PORT
    )
    cur = conn.cursor()
    cur.execute("SELECT embedding FROM public.ingredient_embeddings WHERE ingredient_id = %s;", (ingredient_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    if row:
        return np.array(row[0], dtype=np.float32)
    return None


#############################
#  UNIT / CONVERSION LOOKUP
#############################
def normalize_unit(unit_str):
    """
    Maps 'ounces' -> 'oz', etc.
    """
    mapping = {
        'ounce': 'oz',
        'ounces': 'oz',
    }
    return mapping.get(unit_str.lower(), unit_str)

def get_usda_conversion(usda_ingredient_id, unit):
    """
    Returns (amount, oz, grams) if found, else None.
    """
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
#  CORE MATCHING WITH EMBEDDINGS
#############################
def match_with_faiss(local_vec, faiss_index, usda_ids, top_k=1, threshold=0.75):
    """
    Returns the best USDA match from FAISS index.
    """
    if local_vec is None:
        return None, 0.0

    vec = local_vec.reshape(1, -1).astype('float32')
    faiss.normalize_L2(vec)
    scores, indices = faiss_index.search(vec, top_k)

    best_idx = indices[0][0]
    best_score = scores[0][0]
    if best_score >= threshold:
        return usda_ids[best_idx], float(best_score)
    return None, float(best_score)


def batch_faiss_match(local_vecs, faiss_index, usda_ids, threshold=0.75):
    """
    Returns a list of (best_usda_id, best_score) for each local vector.
    """
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


def aggregate_usda_matches_for_recipe(min_id, max_id, ft_model, faiss_index, usda_ids, embedding_threshold=0.75):
    """
    Batch-match recipe_ingredients to USDA ingredients using FAISS and store results.
    """

    # Load once
    local_mods_map = get_ingredient_modifiers()
    synonym_map = fetch_synonym_map()

    recipe_rows = get_recipe_ingredients(min_id, max_id)
    logging.info(f"Found {len(recipe_rows)} recipe_ingredients between {min_id} and {max_id}.")

    # 1. Collect local embeddings
    local_rows = []
    local_vecs = []

    for row in recipe_rows:
        rec_id, _, _, _, _, parsed_ingredient_id, *_ = row
        local_name = get_local_ingredient_name(parsed_ingredient_id)
        if not local_name:
            logging.warning(f"[RecipeIng {rec_id}] Missing local name for ID={parsed_ingredient_id}.")
            continue

        combined = build_local_full_name(parsed_ingredient_id, local_name, local_mods_map, synonym_map)

        # Try to load cached embedding
        local_vec = load_local_embedding(parsed_ingredient_id)
        if local_vec is None:
            local_vec = embed_text_gensim(combined, ft_model)
            if local_vec is not None:
                store_local_embedding(parsed_ingredient_id, local_vec)

        if local_vec is not None:
            local_rows.append((row, combined, local_vec))
            local_vecs.append(local_vec)

    # 2. Batch match with FAISS
    matches = batch_faiss_match(local_vecs, faiss_index, usda_ids, threshold=embedding_threshold)

    # 3. Update DB
    for (row, combined, _), (matched_usda_id, similarity_score) in zip(local_rows, matches):
        rec_id, _, _, _, _, parsed_ingredient_id, unit_id, *_ = row

        if matched_usda_id is None:
            logging.info(f"[RecipeIng {rec_id}] '{combined}' => No good match (score < {embedding_threshold}).")
            continue

        # Lookup matched unit
        matched_unit_str = None
        if unit_id:
            conn = psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASSWORD, port=DB_PORT)
            cur = conn.cursor()
            cur.execute("SELECT unit_name FROM public.units WHERE id = %s;", (unit_id,))
            unit_row = cur.fetchone()
            cur.close()
            conn.close()
            if unit_row:
                matched_unit_str = normalize_unit(unit_row[0])

        update_recipe_ingredient_with_usda(rec_id, matched_usda_id, similarity_score, matched_unit_str)

        # Unit conversion (if needed)
        if matched_unit_str:
            conv = get_usda_conversion(matched_usda_id, matched_unit_str)
            if conv:
                amount, oz, grams = conv
                logging.info(
                    f"[RecipeIng {rec_id}] '{combined}' => USDA {matched_usda_id}, "
                    f"sim={similarity_score:.2f}, unit='{matched_unit_str}', conversion=({amount}, {oz}, {grams})"
                )
            else:
                logging.info(
                    f"[RecipeIng {rec_id}] '{combined}' => USDA {matched_usda_id}, "
                    f"sim={similarity_score:.2f}, unit='{matched_unit_str}' but no conversion found."
                )
        else:
            logging.info(
                f"[RecipeIng {rec_id}] '{combined}' => USDA {matched_usda_id}, sim={similarity_score:.2f}. "
                f"No unit provided, skipping conversion."
            )

if __name__ == "__main__":
    ft_model = load_fasttext_model("crawl-300d-2M-subword.vec")
    usda_embeddings = precompute_usda_embeddings(ft_model, store_in_db=True)


    faiss_index, usda_ids = build_faiss_index(usda_embeddings)

    sample_ids = [246, 247, 248, 249, 250, 251, 252, 253, 254, 255]
    for rid in sample_ids:
        aggregate_usda_matches_for_recipe(rid, rid, ft_model, faiss_index, usda_ids, embedding_threshold=0.75)


    
    # Alternatively, you could call:
    # aggregate_usda_matches_for_recipe(246, 255, embedding_threshold=0.75)
    # to process them in one call (range).

