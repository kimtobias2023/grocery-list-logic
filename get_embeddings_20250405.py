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
#   CORE USDA QUERIES
#############################
def fetch_synonym_map():
    """
    Returns a reverse synonym map:
    { general_name: specific_name }
    """
    conn = psycopg2.connect(
        host=DB_HOST, database=DB_NAME, user=DB_USER,
        password=DB_PASSWORD, port=DB_PORT
    )
    cur = conn.cursor()
    cur.execute("SELECT specific_name, general_name FROM public.ingredient_synonyms WHERE is_exact_match = TRUE;")
    rows = cur.fetchall()
    cur.close()
    conn.close()

    # Flip the direction to go general → specific
    return {general.lower(): specific.lower() for specific, general in rows}


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
    return rows  # e.g. [(101, 'beef', ['roasted','unsalted']), ... ]

def build_token_to_usda_index(usda_rows):
    """
    Given rows of USDA data [(id, canonical_name, [modifiers])],
    build a mapping from each token (canonical + modifier) → [usda_id, ...]
    """
    index = defaultdict(set)
    for uid, cname, mods in usda_rows:
        tokens = cname.lower().split()
        tokens += [m.lower() for m in mods if m]
        for tok in tokens:
            index[tok].add(uid)
    return index  # { "beef": {101, 102}, "ground": {101}, ... }

#############################
#   BUILD FULL NAME & EMBED
#############################
def get_synonyms(token, synonym_map):
    """
    Returns a list of synonyms for a given token from your synonym_map.
    Reverses the synonym_map for lookup.
    """
    reverse_map = defaultdict(list)
    for specific, general in synonym_map.items():
        reverse_map[general].append(specific)
    
    token = token.lower()
    return reverse_map.get(token, [])


def build_usda_full_name_v2(canonical_name, modifiers, synonym_map=None):
    """
    Create "canonical + all modifiers" in one text string.
    Example: "beef roasted unsalted" (all lowercased).
    """
    base = canonical_name.lower().strip()
    if synonym_map:
        # If there's a known “more specific” synonym, apply it
        base = synonym_map.get(base, base)

    parts = [base]
    if modifiers:
        for m in modifiers:
            if m:  # skip None or empty
                parts.append(m.lower())

    return " ".join(parts)


def store_usda_embedding(usda_id, embedding):
    """
    Insert/Update the embedding in public.usda_ingredient_embeddings (float[]).
    """
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


#############################
#   GET/STORE USDA DATA
#############################
def get_usda_ingredients():
    """
    Returns list of (usda_id, canonical_name, long_desc, [modifier1, modifier2, ...])
    by aggregating all USDA modifiers into a single array per USDA ingredient.
    """
    conn = psycopg2.connect(
        host=DB_HOST, database=DB_NAME, user=DB_USER,
        password=DB_PASSWORD, port=DB_PORT
    )
    cur = conn.cursor()

    # We GROUP BY the USDA ingredient ID, canonical_name, and long_desc,
    # and use array_agg to collect *all* linked modifiers in a single row.
    query = """
    SELECT i.id,
           i.canonical_name,
           i.long_desc,
           COALESCE(array_agg(DISTINCT um.modifier_name), '{}') AS usda_mods
      FROM public.usda_ingredients i
      LEFT JOIN public.usda_ingredient_modifiers uim
             ON i.id = uim.usda_ingredient_id
      LEFT JOIN public.usda_modifiers um
             ON uim.modifier_id = um.id
     GROUP BY i.id, i.canonical_name, i.long_desc
     ORDER BY i.id;
    """
    cur.execute(query)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows  # [(usda_id, canonical_name, long_desc, [mod1, mod2, ...]), ...]


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

def candidate_filter_with_tokens(local_tokens, token_index, synonym_map):
    """
    Filters USDA candidates that overlap with the given tokens or their synonyms.
    """
    candidate_ids = set()
    for token in local_tokens:
        candidate_ids |= token_index.get(token, set())
        for syn in get_synonyms(token, synonym_map):
            candidate_ids |= token_index.get(syn, set())
    return candidate_ids

def match_ingredient_token_filtered(local_string, ft_model, token_index, usda_text_map, synonym_map):
    local_tokens = set(local_string.lower().split())
    candidate_ids = candidate_filter_with_tokens(local_tokens, token_index, synonym_map)
    
    if not candidate_ids:
        return None, 0.0

    local_vec = embed_text_gensim(local_string, ft_model)
    if local_vec is None:
        return None, 0.0

    best_id = None
    best_score = 0.0
    for usda_id in candidate_ids:
        usda_text = usda_text_map[usda_id]
        usda_vec = embed_text_gensim(usda_text, ft_model)
        score = cosine_similarity(local_vec, usda_vec)
        if score > best_score:
            best_score = score
            best_id = usda_id

    return best_id, best_score

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

def get_full_usda_name(usda_id, synonym_map=None):
    """
    Returns the full USDA ingredient name (canonical + any linked modifiers)
    for the given usda_id, using build_usda_full_name_v2().
    """
    # Get all USDA ingredients WITH modifiers
    all_usda_rows = get_usda_ingredients_with_modifiers()
    # all_usda_rows is a list of: [(id, canonical_name, [mod1, mod2, ...]), ...]

    # Find the row matching this usda_id
    for (uid, can_name, mod_list) in all_usda_rows:
        if uid == usda_id:
            # Now build the "canonical + modifiers" name
            return build_usda_full_name_v2(can_name, mod_list, synonym_map)

    # Fallback if none found
    return ""


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

def build_usda_text_map(usda_rows, synonym_map=None):
    """
    Builds a map: usda_id → full_name (canonical + modifiers)
    """
    return {
        uid: build_usda_full_name_v2(cname, mods, synonym_map)
        for uid, cname, mods in usda_rows
    }


def aggregate_usda_matches_for_recipe(min_id, max_id, ft_model, faiss_index, usda_ids, embedding_threshold=0.75):
    """
    Batch-match recipe_ingredients to USDA ingredients using FAISS, update results,
    and print the full USDA name (canonical + modifiers) for debugging.
    """
    # Load local modifiers and synonym map
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

    # 3. Update DB and print full USDA match for debugging
    for (row, combined, _), (matched_usda_id, similarity_score) in zip(local_rows, matches):
        rec_id, _, _, _, _, parsed_ingredient_id, unit_id, *_ = row

        if matched_usda_id is None:
            logging.info(f"[RecipeIng {rec_id}] '{combined}' => No good match (score < {embedding_threshold}).")
            continue

        # Retrieve full USDA name for debugging
        full_usda_name = get_full_usda_name(matched_usda_id, synonym_map)

        # Lookup matched unit if available
        matched_unit_str = None
        if unit_id:
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

        # Log detailed match info including USDA full name
        if matched_unit_str:
            conv = get_usda_conversion(matched_usda_id, matched_unit_str)
            if conv:
                amount, oz, grams = conv
                logging.info(
                    f"[RecipeIng {rec_id}] '{combined}' => USDA {matched_usda_id} ('{full_usda_name}'), "
                    f"sim={similarity_score:.2f}, unit='{matched_unit_str}', "
                    f"conversion=({amount}, {oz}, {grams})."
                )
            else:
                logging.info(
                    f"[RecipeIng {rec_id}] '{combined}' => USDA {matched_usda_id} ('{full_usda_name}'), "
                    f"sim={similarity_score:.2f}, unit='{matched_unit_str}' but no conversion found."
                )
        else:
            logging.info(
                f"[RecipeIng {rec_id}] '{combined}' => USDA {matched_usda_id} ('{full_usda_name}'), "
                f"sim={similarity_score:.2f}. No unit provided, skipping conversion."
            )


if __name__ == "__main__":
    ft_model = load_fasttext_model("crawl-300d-2M-subword.vec")
    synonym_map = fetch_synonym_map()
    usda_rows = get_usda_ingredients_with_modifiers()
    token_index = build_token_to_usda_index(usda_rows)
    usda_text_map = build_usda_text_map(usda_rows, synonym_map)

    test_strings = [
        "pepper green bell",
        "zucchini medium",
        "banana",
        "flour g"
    ]

    for s in test_strings:
        best_id, score = match_ingredient_token_filtered(s, ft_model, token_index, usda_text_map, synonym_map)
        if best_id:
            logging.info(f"'{s}' => USDA {best_id} '{usda_text_map[best_id]}', score={score:.2f}")
        else:
            logging.info(f"'{s}' => No match found.")

    
    # Alternatively, you could call:
    # aggregate_usda_matches_for_recipe(246, 255, embedding_threshold=0.75)
    # to process them in one call (range).

