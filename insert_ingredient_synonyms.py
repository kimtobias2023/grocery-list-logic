import psycopg2
import logging

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
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')


def insert_synonym(specific_name, general_name, is_exact_match=False):
    """
    Insert a synonym into the new ingredient_synonyms table.
    Uses ON CONFLICT DO NOTHING to avoid duplicates.
    """
    conn = psycopg2.connect(
        host=DB_HOST, database=DB_NAME, user=DB_USER,
        password=DB_PASSWORD, port=DB_PORT
    )
    cur = conn.cursor()
    query = """
    INSERT INTO public.ingredient_synonyms (specific_name, general_name, is_exact_match)
    VALUES (%s, %s, %s)
    ON CONFLICT (specific_name, general_name) DO NOTHING
    RETURNING id;
    """
    cur.execute(query, (specific_name.lower(), general_name.lower(), is_exact_match))
    row = cur.fetchone()
    conn.commit()
    cur.close()
    conn.close()
    return row[0] if row else None


def main():
    # Format: (specific, general, is_exact_match)
    synonym_pairs = [
        ("farfelle", "pasta", True),
        ("fusilli", "pasta", True),
        ("bucatini", "pasta", True),
        ("linguine", "pasta", True),
        ("penne", "pasta", True),
        ("orecchiette", "pasta", True),
        ("fettuccine", "pasta", True),
        ("pappardelle", "pasta", True),
        ("rigatoni", "pasta", True),
        ("orzo", "pasta", True),
        ("lasagna", "pasta", True),
        ("ditalini", "pasta", True),
        ("ravioli", "pasta", True),
        ("cavatappi", "pasta", True),
        ("conchiglie", "pasta", True),
        ("gnocchi", "pasta", True),
        ("ziti", "pasta", True),
        ("cannelloni", "pasta", True),
        ("capellini", "pasta", True),
        ("tortellini", "pasta", True),
        ("campanelle", "pasta", True),
        ("gemelli", "pasta", True),
    ]

    for specific, general, is_exact in synonym_pairs:
        inserted_id = insert_synonym(specific, general, is_exact)
        if inserted_id:
            logging.info(f"Inserted: '{specific}' → '{general}' (exact={is_exact}) with id {inserted_id}")
        else:
            logging.info(f"Synonym '{specific}' → '{general}' already exists.")


if __name__ == "__main__":
    main()
