def cosine(a,b): return np.dot(a,b) / (np.linalg.norm(a)*np.linalg.norm(b)+1e-9)

def build_pool_vectors(conn):
    """average all sub‑embeddings per ingredient → 1 vector / ingredient"""
    cur = conn.cursor()
    cur.execute("""
        SELECT ingredient_id, avg(embedding)  -- pgvector avg
        FROM   ingredient_sub_embeddings
        GROUP  BY ingredient_id
    """)
    return {row[0]: np.array(row[1]) for row in cur.fetchall()}

def generate_suggestions():
    conn = psycopg2.connect(...)
    pool = build_pool_vectors(conn)

    ids = list(pool.keys())
    for ia, ib in itertools.combinations(ids, 2):
        sim = cosine(pool[ia], pool[ib])
        if sim < 0.72:                       # ❶   coarse threshold, tweak
            continue

        # ❷   cheap lexical rule to boost obvious cases
        cur = conn.cursor()
        cur.execute("SELECT canonical_name FROM ingredients WHERE id in (%s,%s)",
                    (ia, ib))
        a, b = [row[0] for row in cur.fetchall()]
        reason = "high cosine %.2f" % sim
        if a.split()[-1] == b.split()[-1]:   # share the last token (“butter”)
            sim += .1 ;  reason += " + same head word"

        if sim >= 0.80:
            cur.execute("""
                INSERT INTO ingredient_similarity_suggestions
                    (ingredient_id_a, ingredient_id_b, score, suggested_reason)
                VALUES (%s,%s,%s,%s)
                ON CONFLICT DO NOTHING
            """, (ia, ib, sim, reason))

    conn.commit()
