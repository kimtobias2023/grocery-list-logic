import psycopg2

DB_CONFIG = {
    'host': '172.26.192.1',
    'database': 'mealplanning',
    'user': 'postgres',
    'password': 'new-website-app',
    'port': '5432'
}

def promote_and_clear_suggestions():
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    # Promote only approved suggestions
    cur.execute("""
        INSERT INTO ingredient_synonyms (specific_name, general_name, is_exact_match)
        SELECT specific_name, general_name, FALSE
        FROM ingredient_synonym_suggestions
        WHERE approved = TRUE
        ON CONFLICT (specific_name, general_name) DO NOTHING;
    """)

    # Clear the suggestion table (approved + unapproved)
    cur.execute("TRUNCATE TABLE ingredient_synonym_suggestions;")

    conn.commit()
    cur.close()
    conn.close()
    print("✅ Approved suggestions promoted, and suggestion table cleared.")

if __name__ == "__main__":
    promote_and_clear_suggestions()

