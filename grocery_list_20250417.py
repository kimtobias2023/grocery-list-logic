#!/usr/bin/env python
"""
Deduplicate a list of recipe_ingredient IDs into a grocery list.

Keeps a pointer to *which recipe* each quantity came from.
"""

import psycopg2, logging
from collections import defaultdict
from matcher import best_usda_for_local, best_usda_for_text          # your matcher

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s: %(message)s")

DB = dict(host="localhost", dbname="mealplanning",
          user="postgres",  password="new-website-app", port=5432)

# ─────────────────────────────────────────────────────────────────── DB helpers
def db():
    return psycopg2.connect(**DB)

def unit_info(unit_id):
    """dimension, factor‑to‑gram|ml, unit_name"""
    with db() as c, c.cursor() as cur:
        cur.execute("""SELECT dimension, conversion_factor, unit_name
                         FROM units WHERE id=%s""", (unit_id,))
        r = cur.fetchone()
        return (r[0], float(r[1]), r[2]) if r else (None, None, None)

# ───────────────────────────────────────────── riq_usda_converted cache (same)
CREATE_CONV = """
CREATE TABLE IF NOT EXISTS riq_usda_converted (
  rq_id INT PRIMARY KEY,
  usda_id INT NOT NULL,
  ref_quantity NUMERIC(12,4) NOT NULL,
  ref_dimension TEXT NOT NULL
);
"""
with db() as c, c.cursor() as cur:
    cur.execute(CREATE_CONV); c.commit()

def cached_conversion(rq_id):
    with db() as c, c.cursor() as cur:
        cur.execute("SELECT usda_id, ref_quantity, ref_dimension "
                    "FROM riq_usda_converted WHERE rq_id=%s", (rq_id,))
        if (r := cur.fetchone()):
            return int(r[0]), float(r[1]), r[2]

def save_conversion(rq, uid, qty, dim):
    with db() as c, c.cursor() as cur:
        cur.execute("""INSERT INTO riq_usda_converted
                         (rq_id, usda_id, ref_quantity, ref_dimension)
                       VALUES (%s,%s,%s,%s)
                       ON CONFLICT (rq_id) DO NOTHING""",
                    (rq, uid, qty, dim))
        c.commit()

# ───────────────────────────────────────────── helper: usda via cached match
def usda_by_cached_match(ing_id):
    with db() as c, c.cursor() as cur:
        cur.execute("""SELECT usda_ingredient_id
                         FROM ingredient_usda_matches
                        WHERE local_ingredient_id=%s
                        ORDER BY rank ASC LIMIT 1""",
                    (ing_id,))
        if (r := cur.fetchone()):
            return int(r[0])

# ────────────────────────────────────────────────────────────── canonical map
with db() as c, c.cursor() as cur:
    cur.execute("SELECT id, lower(canonical_name) FROM ingredients")
    CANON2ID = {name: iid for iid, name in cur.fetchall()}           # fast lookup

# ────────────────────────────────────────────────────────────── aggregator
def aggregate(recipe_ing_ids):
    grocery = defaultdict(lambda: dict(qty=0.0, dim=None, items=[]))

    # ---------- fetch all recipe‑ingredients and child quantities
    with db() as c, c.cursor() as cur:
        cur.execute("""SELECT ri.id,
                              ri.recipe_id,
                              ri.ingredient_id,  
                              ri.ingredient_string
                         FROM recipe_ingredients ri
                        WHERE ri.id = ANY(%s)""",
                    (list(recipe_ing_ids),))
        rows = cur.fetchall()

        rid_set = [r[0] for r in rows] or [0]
        cur.execute("""SELECT id, recipe_ingredient_id,
                              quantity, unit_id, quantity_text
                         FROM recipe_ingredient_quantities
                        WHERE recipe_ingredient_id = ANY(%s)""",
                    (rid_set,))
        qrows = cur.fetchall()

    q_by_parent = defaultdict(list)
    for rqid, rid, qty, unit_id, qtxt in qrows:
        q_by_parent[rid].append(dict(rq_id=rqid,
                                     quantity=float(qty) if qty else None,
                                     unit_id=unit_id,
                                     qtxt=qtxt or ""))

    # ---------- main loop
    for ri_id, recipe_id, local_id, ing_str in rows:
        # local_id is *usually* there; if NULL we can still fall back
        if not local_id:
            ing_key  = ing_str.lower().strip()
            local_id = CANON2ID.get(ing_key)

        # ① pick best USDA id
        if local_id:
            # 1) cached match → fast
            usda_id = usda_by_cached_match(local_id)
            # 2) no cached row yet → compute once (fast, in‑memory)
            if not usda_id:
                usda_id, _ = best_usda_for_local(local_id)
        else:
            # very rare: no FK, fall back to free‑text
            token    = ing_str.split(',')[0].split('(')[0].strip()
            usda_id, _ = best_usda_for_text(token.lower())

        if not usda_id:
            logging.warning("No USDA match for “%s” – skipped", ing_str)
            continue

        # ② convert each quantity to reference (g / ml / count)
        for q in q_by_parent[ri_id]:
            rq_id = q["rq_id"]

            cached = cached_conversion(rq_id)
            if cached:
                uid, ref_q, ref_d = cached
            else:
                base = q["quantity"] or 1.0
                dim, cf, _ = unit_info(q["unit_id"]) if q["unit_id"] else (None,None,None)
                ref_q, ref_d = base, "count"
                if dim == "mass"   and cf: ref_q, ref_d = base*cf, "mass"
                if dim == "volume" and cf: ref_q, ref_d = base*cf, "volume"
                save_conversion(rq_id, usda_id, ref_q, ref_d)

            grocery[usda_id]['qty'] += ref_q
            grocery[usda_id]['dim']  = ref_d
            grocery[usda_id]['items'].append(
                dict(recipe_id=recipe_id,
                     rq_id=rq_id,
                     ref_qty=ref_q,
                     ref_dim=ref_d,
                     original=q["qtxt"])
            )

    # ---------- resolve names
    if not grocery:
        return []

    with db() as c, c.cursor() as cur:
        cur.execute("SELECT id, canonical_name FROM usda_ingredients "
                    "WHERE id = ANY(%s)", (list(grocery.keys()),))
        for uid, name in cur.fetchall():
            grocery[uid]['name'] = name

    # ---------- final list
    return [dict(usda_id  = uid,
                 usda_name= d.get('name',''),
                 dimension= d['dim'],
                 quantity = round(d['qty'],2),
                 items    = d['items'])         # recipe provenance inside
            for uid, d in grocery.items()]

# ────────────────────────────────────────────────────────────── demo
if __name__ == "__main__":
    TEST_IDS = [2213, 2214, 2251, 2317, 2377, 2391, 2396, 2230, 2157]
    for row in aggregate(TEST_IDS):
        logging.info("[%s] %-40s  %.2f %s",
                     row["usda_id"], row["usda_name"],
                     row["quantity"], row["dimension"])

