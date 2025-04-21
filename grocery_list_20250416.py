#!/usr/bin/env python
"""
Aggregate an arbitrary list of recipe_ingredient IDs into a
deduplicated grocery list (grams / ml / count) using:

    • ingredient_usda_matches   – top‑1 match per local ingredient
    • units                     – g/ml conversion factors
    • usda_ingredient_conversion – when only “1 cup”‑style info exists
    • (falls back to on‑the‑fly matching only if no cached match)

Result = list[{usda_id, usda_name, dimension, quantity}]
"""

import psycopg2, logging, numpy as np
from collections import defaultdict
# your own matcher (already stores results in ingredient_usda_matches)
# NEW
from matcher import best_usda_for_local, best_usda_for_text
 

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s: %(message)s")

DB = dict(host="localhost", dbname="mealplanning",
          user="postgres",  password="new-website-app", port=5432)

###############################################################################
# DB helpers
###############################################################################
def db():
    return psycopg2.connect(**DB)

def unit_info(unit_id):
    """return (dimension, conv_factor, unit_name) or (None,None,None)"""
    with db() as conn, conn.cursor() as cur:
        cur.execute("""SELECT dimension, conversion_factor, unit_name
                         FROM units WHERE id=%s""", (unit_id,))
        r = cur.fetchone()
        return (r[0], float(r[1]), r[2]) if r else (None,None,None)

###############################################################################
# (Optional) conversion cache  – saves expensive recalculation
###############################################################################
CREATE_CONV = """
CREATE TABLE IF NOT EXISTS riq_usda_converted (
   rq_id            INT    PRIMARY KEY,          -- recipe_ingredient_quantities.id
   usda_id          INT    NOT NULL,
   ref_quantity     NUMERIC(12,4) NOT NULL,      -- g or ml or count
   ref_dimension    TEXT  NOT NULL               -- 'mass' | 'volume' | 'count'
);
"""
with db() as conn, conn.cursor() as cur:         # autocreates the helper table
    cur.execute(CREATE_CONV)
    conn.commit()

def cached_conversion(rq_id):
    with db() as conn, conn.cursor() as cur:
        cur.execute("""SELECT usda_id, ref_quantity, ref_dimension
                         FROM riq_usda_converted WHERE rq_id=%s""",
                    (rq_id,))
        r = cur.fetchone()
        if r:
            return int(r[0]), float(r[1]), r[2]
    return None

def save_conversion(rq_id, usda_id, qty, dim):
    with db() as conn, conn.cursor() as cur:
        cur.execute("""INSERT INTO riq_usda_converted
                          (rq_id, usda_id, ref_quantity, ref_dimension)
                       VALUES (%s,%s,%s,%s)
                       ON CONFLICT (rq_id) DO NOTHING""",
                    (rq_id, usda_id, qty, dim))
        conn.commit()

###############################################################################
# helper : best USDA id for an ingredient_string
###############################################################################
def usda_by_cached_match(ingredient_id):
    with db() as conn, conn.cursor() as cur:
        cur.execute("""SELECT usda_ingredient_id
                         FROM ingredient_usda_matches
                        WHERE local_ingredient_id=%s
                        ORDER BY rank ASC   -- rank==1 is best
                        LIMIT 1""", (ingredient_id,))
        r = cur.fetchone()
        return int(r[0]) if r else None

###############################################################################
# MAIN aggregation
###############################################################################
def aggregate(recipe_ing_ids):
    grocery = defaultdict(lambda: dict(qty=0.0, dim=None, items=[]))

    # ------------------------------------------------------------------ fetch
    with db() as conn, conn.cursor() as cur:
        cur.execute("""SELECT ri.id, ri.ingredient_string
                        FROM recipe_ingredients ri
                        WHERE ri.id = ANY(%s)""",
                    (list(recipe_ing_ids),))     # list, not tuple
        rows = cur.fetchall()

        # child quantities
        rid_set = [r[0] for r in rows] or [0]     # list, non‑empty
        cur.execute("""SELECT id, recipe_ingredient_id,
                            quantity, unit_id, quantity_text
                        FROM recipe_ingredient_quantities
                        WHERE recipe_ingredient_id = ANY(%s)""",
                    (rid_set,))
        qrows = cur.fetchall()

    # index quantities by parent
    q_by_parent = defaultdict(list)
    for rqid, rid, qty, unit_id, qtxt in qrows:
        q_by_parent[rid].append(dict(rq_id=rqid,
                                     quantity=float(qty) if qty else None,
                                     unit_id=unit_id,
                                     qtxt=qtxt or ""))

    # ------------------------------------------------------------------ loop
    for rid, ing_str in rows:
        # guess local_ingredient_id via ingredients.canonical_name
        with db() as conn, conn.cursor() as cur2:
            cur2.execute("""SELECT id
                              FROM ingredients
                             WHERE lower(canonical_name)=lower(%s)
                             LIMIT 1""", (ing_str,))
            row = cur2.fetchone()
            local_ing_id = int(row[0]) if row else None

        # 1️⃣ decide best USDA id
        usda_id = None
        if local_ing_id:                          # if your schema has this FK
            usda_id = usda_by_cached_match(local_ing_id)

        if not usda_id:
            usda_id, _ = (
                best_usda_for_local(local_ing_id)
                if local_ing_id            # we have proper chunks
                else best_usda_for_text(ing_str)
            )

        if not usda_id:
            logging.warning("No USDA match for ingredient %s – skipped", ing_str)
            continue

        # 2️⃣ convert each quantity to a reference value
        for q in q_by_parent[rid]:
            rq_id = q["rq_id"]
            cached = cached_conversion(rq_id)
            if cached:
                uid, ref_qty, ref_dim = cached
                grocery[uid]['qty'] += ref_qty
                grocery[uid]['dim']  = ref_dim
                grocery[uid]['items'].append((rid, ref_qty, ref_dim, q["qtxt"]))
                continue

            base = q["quantity"] or 1.0
            unit_id = q["unit_id"]
            dim, cf, unit_name = (None,None,None)

            if unit_id:
                dim, cf, unit_name = unit_info(unit_id)

            ref_qty, ref_dim = base, "count"
            if dim == "mass"   and cf: ref_qty, ref_dim = base*cf, "mass"
            if dim == "volume" and cf: ref_qty, ref_dim = base*cf, "volume"

            grocery[usda_id]['qty'] += ref_qty
            grocery[usda_id]['dim']  = ref_dim
            grocery[usda_id]['items'].append((rid, ref_qty, ref_dim, q["qtxt"]))

            save_conversion(rq_id, usda_id, ref_qty, ref_dim)

    # ------------------------------------------------------------------ names
    if not grocery:
        return []

    with db() as conn, conn.cursor() as cur:
        cur.execute("""SELECT id, canonical_name FROM usda_ingredients
                        WHERE id = ANY(%s)""", (list(grocery.keys()),))
        for uid, name in cur.fetchall():
            grocery[uid]['name'] = name

    # tidy output
    out = []
    for uid, d in grocery.items():
        out.append(dict(usda_id=uid,
                        usda_name=d.get('name',""),
                        dimension=d['dim'],
                        quantity=round(d['qty'],2)))
    return out

###############################################################################
# DEMO
###############################################################################
if __name__ == "__main__":
    TEST_IDS = [2213, 2214, 2251, 2317, 2377, 2391, 2396, 2230, 2157]
    res = aggregate(TEST_IDS)
    logging.info("=== FINAL LIST ===")
    for r in res:
        logging.info("%s  – %.2f %s",
                     f"[{r['usda_id']}] {r['usda_name']}",
                     r['quantity'], r['dimension'])

