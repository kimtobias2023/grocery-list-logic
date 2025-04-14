import psycopg2
import logging
import numpy as np
from embedding_utils import load_fasttext_model, embed_text_gensim, load_usda_embedding

# DB config
DB_CONFIG = {
    'host': '172.26.192.1',
    'database': 'mealplanning',
    'user': 'postgres',
    'password': 'new-website-app',
    'port': '5432'
}

logging.basicConfig(level=logging.INFO)

def fetch_usda_ingredients_by_category(cat_id):
    """
    Given a USDA food category ID, return all (id, canonical_name) for ingredients in that category.
    """
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("""
        SELECT ui.id, ui.canonical_name
        FROM public.usda_ingredients ui
        JOIN public.usda_ingredient_categories uic ON ui.id = uic.usda_ingredient_id
        WHERE uic.food_category_id = %s;
    """, (cat_id,))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows



def fetch_usda_modifiers():
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("""
        SELECT uim.usda_ingredient_id, um.modifier_name
        FROM public.usda_ingredient_modifiers uim
        JOIN public.usda_modifiers um ON uim.modifier_id = um.id;
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    mod_map = {}
    for ing_id, mod in rows:
        mod_map.setdefault(ing_id, []).append(mod.lower())
    return mod_map

def insert_synonym_suggestion(specific, general, score):
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO ingredient_synonym_suggestions (specific_name, general_name, similarity_score, approved)
        VALUES (%s, %s, %s, false)
        ON CONFLICT (specific_name, general_name) DO NOTHING;
    """, (specific.lower(), general.lower(), score))
    conn.commit()
    cur.close()
    conn.close()


def suggest_usda_synonyms_by_category(category_id, ft_model):
    usda_ings = fetch_usda_ingredients_by_category(category_id)
    mod_map = fetch_usda_modifiers()

    for usda_id, general_name in usda_ings:
        modifiers = mod_map.get(usda_id, [])
        
        # Prefer precomputed USDA embedding
        general_vec = load_usda_embedding(usda_id)
        if general_vec is None:
            general_vec = embed_text_gensim(general_name, ft_model)

        if general_vec is None:
            continue

        for mod in modifiers:
            mod_vec = embed_text_gensim(mod, ft_model)
            if mod_vec is None:
                continue

            sim = float(np.dot(general_vec, mod_vec) / (np.linalg.norm(general_vec) * np.linalg.norm(mod_vec)))
            insert_synonym_suggestion(mod, general_name, sim)
            logging.info(f"Suggested: {mod} → {general_name} (score={sim:.2f})")

if __name__ == "__main__":
    ft_model = load_fasttext_model("crawl-300d-2M-subword.vec")

    # Start with category 14: "nut and seed products"
    suggest_usda_synonyms_by_category(22, ft_model)
