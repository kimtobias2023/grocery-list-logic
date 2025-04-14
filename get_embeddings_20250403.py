import psycopg2
import logging
from gensim.models import KeyedVectors
import numpy as np

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

#############################
#   OPTIONAL: STORING USDA EMBEDDINGS IN DB
#############################
def store_usda_embedding(usda_id, embedding):
    """
    Insert/Update the embedding in public.usda_ingredient_embeddings (float[]).
    """
    conn = psycopg2.connect(
        host=DB_HOST, database=DB_NAME, user=DB_USER,
        password=DB_PASSWORD, port=DB_PORT
    )
    cur = conn.cursor()
    # We pass the python list of floats as a Postgres array
    cur.execute("""
        INSERT INTO public.usda_ingredient_embeddings (usda_ingredient_id, embedding)
        VALUES (%s, %s)
        ON CONFLICT (usda_ingredient_id) DO UPDATE
        SET embedding = EXCLUDED.embedding;
    """, (usda_id, list(embedding)))
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

#############################
#   RECIPE ING HELPER 
#############################
def build_local_full_name(ingredient_id, canonical, local_modifiers_map):
    """
    Combine local ingredient's canonical_name + all its modifiers into one text.
    """
    parts = [canonical.lower()]
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
def match_ingredient_with_usda_embedding(local_text, ft_model, usda_embeddings, threshold=0.75):
    """
    1) Embeds local_text.
    2) Loops over usda_embeddings to find the best cosine similarity.
    3) Returns (best_usda_id, best_score).
    """
    local_vec = embed_text_gensim(local_text, ft_model)
    if local_vec is None:
        return None, 0.0

    best_id = None
    best_sim = 0.0
    for usda_id, usda_vec in usda_embeddings.items():
        if usda_vec is None:
            continue
        sim = cosine_similarity(local_vec, usda_vec)
        if sim > best_sim:
            best_sim = sim
            best_id = usda_id

    if best_sim >= threshold:
        return best_id, best_sim
    return None, best_sim

def aggregate_usda_matches_for_recipe(min_id, max_id, ft_model, usda_embeddings, embedding_threshold=0.75):

    """
    2) Precompute USDA embeddings in memory.
    3) For each recipe_ingredient between min_id and max_id, 
       build the local combined text, embed, find the best USDA match, 
       update recipe_ingredients if above threshold, and 
       attempt unit conversion if matched_unit_str is found.
    """

    # 2) Precompute USDA embeddings
    usda_embeddings = precompute_usda_embeddings(ft_model, store_in_db=False)

    # get the local modifiers map
    local_mods_map = get_ingredient_modifiers()

    # 3) Fetch recipe ingredients
    recipe_rows = get_recipe_ingredients(min_id, max_id)
    logging.info(f"Found {len(recipe_rows)} recipe_ingredients between {min_id} and {max_id}.")

    for (
        rec_id, recipe_id, ingredient_string, weight, quantity,
        parsed_ingredient_id, unit_id, existing_usda_id, existing_usda_score, existing_usda_unit
    ) in recipe_rows:

        local_name = get_local_ingredient_name(parsed_ingredient_id)
        if not local_name:
            logging.warning(f"[RecipeIng {rec_id}] Missing local ingredient name for ID={parsed_ingredient_id}.")
            continue

        # Combine local canonical name + modifiers
        combined_local = build_local_full_name(parsed_ingredient_id, local_name, local_mods_map)

        # 4) Match using embeddings
        matched_usda_id, similarity_score = match_ingredient_with_usda_embedding(
            combined_local, ft_model, usda_embeddings, threshold=embedding_threshold
        )
        if matched_usda_id is None:
            logging.info(f"[RecipeIng {rec_id}] '{combined_local}' => No good match (score < {embedding_threshold}).")
            continue

        # 5) Update DB
        matched_unit_str = None
        if unit_id:
            # get local unit
            conn = psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER,
                                    password=DB_PASSWORD, port=DB_PORT)
            cur = conn.cursor()
            cur.execute("SELECT unit_name FROM public.units WHERE id = %s;", (unit_id,))
            unit_row = cur.fetchone()
            cur.close()
            conn.close()
            if unit_row:
                matched_unit_str = normalize_unit(unit_row[0])

        update_recipe_ingredient_with_usda(rec_id, matched_usda_id, similarity_score, matched_unit_str)

        # 6) If unit found, attempt conversion
        if matched_unit_str:
            conv = get_usda_conversion(matched_usda_id, matched_unit_str)
            if conv:
                amount, oz, grams = conv
                logging.info(
                    f"[RecipeIng {rec_id}] '{combined_local}' => USDA {matched_usda_id}, "
                    f"sim={similarity_score:.2f}, unit='{matched_unit_str}', "
                    f"conversion=({amount}, {oz}, {grams})."
                )
            else:
                logging.info(
                    f"[RecipeIng {rec_id}] '{combined_local}' => USDA {matched_usda_id}, "
                    f"sim={similarity_score:.2f}, unit='{matched_unit_str}' but no conversion found."
                )
        else:
            logging.info(
                f"[RecipeIng {rec_id}] '{combined_local}' => USDA {matched_usda_id}, "
                f"sim={similarity_score:.2f}. No unit provided, skipping conversion."
            )

if __name__ == "__main__":
    ft_model = load_fasttext_model("crawl-300d-2M-subword.vec")  # ⏳ Load ONCE
    usda_embeddings = precompute_usda_embeddings(ft_model, store_in_db=False)  # 🧠 Compute ONCE

    sample_ids = [246, 247, 248, 249, 250, 251, 252, 253, 254, 255]
    for rid in sample_ids:
        aggregate_usda_matches_for_recipe(rid, rid, ft_model, usda_embeddings, embedding_threshold=0.75)

    
    # Alternatively, you could call:
    # aggregate_usda_matches_for_recipe(246, 255, embedding_threshold=0.75)
    # to process them in one call (range).

