import psycopg2
import logging
from rapidfuzz import process, fuzz

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

def get_recipe_ingredients(min_id, max_id):
    """
    Fetch recipe_ingredients rows from the DB for the given ID range.
    """
    conn = psycopg2.connect(
        host=DB_HOST, database=DB_NAME, user=DB_USER,
        password=DB_PASSWORD, port=DB_PORT
    )
    cur = conn.cursor()
    query = """
    SELECT id, recipe_id, ingredient_string, weight, quantity, parsed_ingredient_id, unit_id,
           usda_ingredient_id, usda_match_score, usda_matched_unit
      FROM public.recipe_ingredients
     WHERE id BETWEEN %s AND %s;
    """
    cur.execute(query, (min_id, max_id))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows

def get_usda_ingredients():
    """
    Returns list of tuples (id, canonical_name) for all USDA ingredients.
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
    return rows

def get_usda_ingredient_modifiers():
    """
    Retrieve a mapping of USDA ingredient IDs -> list of associated modifier names.
    e.g., { usda_ingredient_id: ["roasted", "unsalted", ...], ... }
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
    for usda_id, modifier in rows:
        modifiers_map.setdefault(usda_id, []).append(modifier.lower())
    return modifiers_map

def get_usda_conversion(usda_ingredient_id, unit):
    """
    Returns a single row (amount, oz, grams) from usda_ingredient_conversion
    if a match is found for the given usda_ingredient_id and unit, else None.
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
    return row

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

def normalize_unit(unit_str):
    """
    Normalize unit strings to improve conversion lookups.
    For example, map 'ounce' or 'ounces' to 'oz'.
    Add more as needed.
    """
    mapping = {
        'ounce': 'oz',
        'ounces': 'oz',
        # Additional mappings if you want
    }
    return mapping.get(unit_str.lower(), unit_str)

def find_exact_match(local_name, usda_lookup, modifiers_map):
    """
    Attempts an exact match for local_name against:
      1) the first two comma-split tokens of USDA canonical_name
      2) the USDA ingredient's associated modifiers
    Returns (matched_name, matched_id, 100) if found, else None.
    """
    local_name_lower = local_name.lower()
    for usda_name, usda_id in usda_lookup:
        # Check the first two comma-split tokens of the USDA canonical_name
        tokens = [t.strip().lower() for t in usda_name.split(',')]
        for token in tokens[:2]:
            if local_name_lower == token or local_name_lower in token:
                return usda_name, usda_id, 100
        
        # Check the associated USDA modifiers
        modifiers = modifiers_map.get(usda_id, [])
        for modifier in modifiers:
            if local_name_lower == modifier or local_name_lower in modifier:
                return usda_name, usda_id, 100
    return None

def update_recipe_ingredient_with_usda(recipe_ing_id, usda_id, match_score, matched_unit=None):
    """
    Update recipe_ingredients with the matched USDA ingredient data:
      - usda_ingredient_id
      - usda_match_score
      - usda_matched_unit
    """
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
    logging.info(
        f"[RecipeIng {recipe_ing_id}] Updated with USDA ID {usda_id}, "
        f"score {match_score}, unit='{matched_unit}'."
    )

def get_ingredient_synonyms(local_name):
    """
    Return list of synonyms for a given local ingredient name.
    """
    conn = psycopg2.connect(
        host=DB_HOST, database=DB_NAME, user=DB_USER,
        password=DB_PASSWORD, port=DB_PORT
    )
    cur = conn.cursor()
    cur.execute("""
        SELECT synonym
          FROM public.ingredient_synonyms
         WHERE ingredient_name = %s;
    """, (local_name.lower(),))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [row[0] for row in rows]

def aggregate_usda_matches_for_recipe(min_id, max_id, similarity_threshold=80):
    """
    1) Get USDA data (canonical_name, modifiers).
    2) Retrieve recipe_ingredients rows in [min_id, max_id].
    3) For each recipe_ingredient, do:
         a) exact match (local_name)
         b) if none, exact match (synonyms)
         c) if none, fuzzy match (local_name)
         d) if still none, fuzzy match (synonyms)
         e) if match_score >= threshold, update recipe_ingredients
         f) attempt conversion if a unit is present
    """
    # 1) USDA data
    usda_data = get_usda_ingredients()  
    # Each row is (usda_id, canonical_name), so create a lookup (canonical_name, usda_id)
    usda_lookup = [(row[1], row[0]) for row in usda_data]  
    modifiers_map = get_usda_ingredient_modifiers()

    # 2) Get recipe ingredients
    recipe_rows = get_recipe_ingredients(min_id, max_id)
    logging.info(f"Found {len(recipe_rows)} recipe_ingredients rows between ids {min_id} and {max_id}.")

    for rec in recipe_rows:
        rec_id, recipe_id, ingredient_string, weight, quantity, parsed_ingredient_id, unit_id, \
            existing_usda_id, existing_usda_score, existing_usda_unit = rec
        
        # a) Get the local_name from 'ingredients' table
        local_name = get_local_ingredient_name(parsed_ingredient_id)
        if not local_name:
            logging.warning(
                f"[RecipeIng {rec_id}] No local ingredient name for parsed_ingredient_id={parsed_ingredient_id}."
            )
            continue

        # -----------------------------------------------------
        # Step 1: Exact match on the local_name
        # -----------------------------------------------------
        match_info = find_exact_match(local_name, usda_lookup, modifiers_map)
        matched_usda_name, usda_id, score = (None, None, 0)

        # -----------------------------------------------------
        # Step 2: If no exact match, check synonyms
        # -----------------------------------------------------
        if not match_info:
            synonyms = get_ingredient_synonyms(local_name)
            for syn in synonyms:
                match_info = find_exact_match(syn, usda_lookup, modifiers_map)
                if match_info:
                    matched_usda_name, usda_id, score = match_info
                    logging.info(f"[RecipeIng {rec_id}] Exact match found using synonym '{syn}'.")
                    break
        else:
            matched_usda_name, usda_id, score = match_info

        # -----------------------------------------------------
        # Step 3: Fuzzy match on the local_name (only if no exact match)
        # -----------------------------------------------------
        if not matched_usda_name:
            fuzzy_result = process.extractOne(local_name, usda_lookup, scorer=fuzz.token_set_ratio)
            if fuzzy_result:
                (fuzzy_name, fuzzy_id), fuzzy_score, _ = fuzzy_result
                matched_usda_name, usda_id, score = fuzzy_name, fuzzy_id, fuzzy_score

        # -----------------------------------------------------
        # Step 4: Fuzzy match synonyms (only if no match from local fuzzy)
        # -----------------------------------------------------
        if not matched_usda_name:
            synonyms = get_ingredient_synonyms(local_name)
            for syn in synonyms:
                fuzzy_result = process.extractOne(syn, usda_lookup, scorer=fuzz.token_set_ratio)
                if fuzzy_result:
                    (fuzzy_name, fuzzy_id), fuzzy_score, _ = fuzzy_result
                    # Keep track if it's the best so far
                    if fuzzy_score > score:
                        matched_usda_name, usda_id, score = fuzzy_name, fuzzy_id, fuzzy_score

        # -----------------------------------------------------
        # If still no match found, skip
        # -----------------------------------------------------
        if not matched_usda_name:
            logging.info(f"[RecipeIng {rec_id}] No USDA match found for '{local_name}' or its synonyms.")
            continue

        # -----------------------------------------------------
        # Check the match_score threshold
        # -----------------------------------------------------
        if score < similarity_threshold:
            logging.info(
                f"[RecipeIng {rec_id}] Low similarity ({score}) for '{local_name}'. "
                f"Best match: '{matched_usda_name}'."
            )
            continue

        # We have a match -> update recipe_ingredients
        matched_unit_str = None

        # If there's a unit_id, see which unit_name that is
        if unit_id:
            conn = psycopg2.connect(
                host=DB_HOST, database=DB_NAME, user=DB_USER,
                password=DB_PASSWORD, port=DB_PORT
            )
            cur = conn.cursor()
            cur.execute("SELECT unit_name FROM public.units WHERE id = %s;", (unit_id,))
            unit_row = cur.fetchone()
            cur.close()
            conn.close()
            if unit_row:
                matched_unit_str = normalize_unit(unit_row[0])

        # Update DB with usda_ingredient_id, score, matched_unit
        update_recipe_ingredient_with_usda(rec_id, usda_id, score, matched_unit_str)

        # Attempt conversion if we have a matched_unit_str
        if matched_unit_str:
            usda_conv = get_usda_conversion(usda_id, matched_unit_str)
            if usda_conv:
                amount, oz, grams = usda_conv
                logging.info(
                    f"[RecipeIng {rec_id}] Local '{local_name}' → USDA '{matched_usda_name}' (id {usda_id}), "
                    f"score={score}, unit='{matched_unit_str}' => amount={amount}, oz={oz}, grams={grams}."
                )
            else:
                logging.info(
                    f"[RecipeIng {rec_id}] Local '{local_name}' → USDA '{matched_usda_name}' (id {usda_id}), "
                    f"score={score}, but no USDA conversion for unit='{matched_unit_str}'."
                )
        else:
            logging.info(
                f"[RecipeIng {rec_id}] Local '{local_name}' → USDA '{matched_usda_name}' (id {usda_id}), "
                f"score={score}. No unit provided, skipping conversion."
            )

if __name__ == "__main__":
    # Example usage: match from recipe_ingredients.id 246 to 256
    aggregate_usda_matches_for_recipe(246, 256)

