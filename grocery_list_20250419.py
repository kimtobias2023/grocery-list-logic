#!/usr/bin/env python
"""
Aggregate an arbitrary list of recipe_ingredient IDs
into a deduplicated grocery list.

Rules
─────
produce / eggs        → count   (fallback mass)
meat / other          → mass
dairy / liquid        → volume  (fallback mass)

metric    → g / ml        (default)
imperial  → oz / fl oz
"""

import logging, psycopg2
from collections import defaultdict
from matcher import best_usda_for_local, best_usda_for_text   # ← your fast matcher

# ───────────────────────────── config ──────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s: %(message)s")

DB = dict(host="localhost", dbname="mealplanning",
          user="postgres",  password="new-website-app", port=5432)

USER_UNIT_SYSTEM = "metric"          # or "imperial"
# conversion constants
G2OZ      = 1 / 28.349523125
ML2FLOZ   = 1 / 29.5735295625
OZ2G      = 28.349523125
FLOZ2ML   = 29.5735295625
# preferred dimension per class
CLASS_PREF = {
    "produce": ("count",  "mass"),
    "eggs":    ("count",  "mass"),
    "meat":    ("mass",   None),
    "dairy":   ("volume", "mass"),
    "liquid":  ("volume", "mass"),
}
# ───────────────────────────── helpers ─────────────────────────────
def db(): return psycopg2.connect(**DB)

def unit_info(unit_id):
    """return (dimension, factor‑to‑g/ml, unit_name)"""
    with db() as c, c.cursor() as cur:
        cur.execute("""SELECT dimension, conversion_factor, unit_name
                         FROM units WHERE id=%s""", (unit_id,))
        if (r := cur.fetchone()):
            return r[0], float(r[1]), r[2]
    return None, None, None

# cache table (unchanged)
CREATE_CONV = """
CREATE TABLE IF NOT EXISTS riq_usda_converted (
  rq_id INT PRIMARY KEY,
  usda_id INT NOT NULL,
  ref_quantity NUMERIC(12,4) NOT NULL,
  ref_dimension TEXT NOT NULL
);
"""
with db() as c, c.cursor() as cur: cur.execute(CREATE_CONV); c.commit()

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
                    (rq, uid, qty, dim)); c.commit()

def usda_by_cached_match(ing_id):
    with db() as c, c.cursor() as cur:
        cur.execute("""SELECT usda_ingredient_id
                         FROM ingredient_usda_matches
                        WHERE local_ingredient_id=%s
                        ORDER BY rank LIMIT 1""", (ing_id,))
        if (r := cur.fetchone()): return int(r[0])

# fast look‑ups (load **once**)
with db() as c, c.cursor() as cur:
    cur.execute("SELECT id, lower(canonical_name) FROM ingredients")
    CANON2ID = {n: i for i, n in cur.fetchall()}

    cur.execute("SELECT ingredient_id, class FROM ingredient_classes")
    CLASS_OF  = {i: cls for i, cls in cur.fetchall()}

# ──────────────────────────── aggregation ─────────────────────────
# ────────────────────────────────────────────────────────────────
def aggregate(ri_ids):
    """
    Aggregate a list of recipe‑ingredient IDs into a grocery list.
    """
    grocery = defaultdict(lambda: {
        "buckets": defaultdict(float),
        "class":   None,
        "items":   []
    })

    # ─── pull parents + children in one trip ─────────────────────
    with db() as c, c.cursor() as cur:
        cur.execute("""
            SELECT ri.id, ri.recipe_id, ri.ingredient_id, ri.ingredient_string
              FROM recipe_ingredients ri
             WHERE ri.id = ANY(%s)
        """, (list(ri_ids),))
        parents = cur.fetchall()

        cur.execute("""
            SELECT id, recipe_ingredient_id,
                   quantity, unit_id, quantity_text, sub_label
              FROM recipe_ingredient_quantities
             WHERE recipe_ingredient_id = ANY(%s)
        """, ([p[0] for p in parents] or [0],))
        quants = cur.fetchall()

    # children → dict by parent
    Q = defaultdict(list)
    for rqid, rid, qty, unit_id, qtxt, sub_label in quants:
        Q[rid].append(dict(
            rq_id     = rqid,
            quantity  = qty,
            unit_id   = unit_id,
            qtxt      = qtxt or "",
            sub_label = (sub_label or "").lower()
        ))

    # ─── walk every parent row ───────────────────────────────────
    for ri_id, recipe_id, local_id, raw in parents:

        if not local_id:                                   # fallback lookup
            local_id = CANON2ID.get(raw.lower().strip())

        # ― best USDA id
        if local_id and (uid := usda_by_cached_match(local_id)):
            usda_id = uid
        else:
            if local_id:
                usda_id, _ = best_usda_for_local(local_id)
            else:
                token = raw.split(',')[0].split('(')[0]
                usda_id, _ = best_usda_for_text(token.lower())
        if not usda_id:
            logging.warning("No USDA match for “%s” – skipped", raw)
            continue

        # ― class
        ingr_class = CLASS_OF.get(local_id, "other")
        grocery[usda_id]['class'] = ingr_class

        # ― choose ONE primary quantity row ──────────────────────
        rows = Q[ri_id]
        if not rows:                                        # no numbers at all
            rows = [dict(rq_id=None, quantity=None, unit_id=None,
                         qtxt="", sub_label="")]

        # helper: dimension for a row
        def _dim(r):
            return unit_info(r['unit_id'])[0] if r['unit_id'] else None

        mass_rows   = [r for r in rows if _dim(r) == 'mass']
        volume_rows = [r for r in rows if _dim(r) == 'volume']

        if   mass_rows  : primary = max(mass_rows,   key=lambda r: float(r['quantity'] or 0))
        elif volume_rows: primary = max(volume_rows, key=lambda r: float(r['quantity'] or 0))
        else:
            primary = next((r for r in rows if r['sub_label'] == 'a'), rows[0])

        # ― convert / cache ──────────────────────────────────────
        rq   = primary['rq_id']
        base = float(primary['quantity']) if primary['quantity'] else 1.0
        dim, cf, _ = unit_info(primary['unit_id']) if primary['unit_id'] else (None, None, None)

        cache = cached_conversion(rq) if rq else None
        if cache:
            _, ref_qty, ref_dim = cache
        else:
            ref_qty, ref_dim = base, "count"
            if dim == "mass"   and cf: ref_qty, ref_dim = base * cf, "mass"
            if dim == "volume" and cf: ref_qty, ref_dim = base * cf, "volume"
            if rq:                                                 # only store when we have an rq_id
                save_conversion(rq, usda_id, ref_qty, ref_dim)

        grocery[usda_id]['buckets'][ref_dim] += ref_qty
        grocery[usda_id]['items'].append(dict(
            recipe_id = recipe_id,
            rq_id     = rq,
            ref_qty   = ref_qty,
            ref_dim   = ref_dim,
            original  = primary['qtxt']
        ))

    # ─── nothing aggregated ─────────────────────────────────────
    if not grocery:
        return []

    # ─── resolve USDA names ─────────────────────────────────────
    with db() as c, c.cursor() as cur:
        cur.execute("""
            SELECT id, canonical_name
              FROM usda_ingredients
             WHERE id = ANY(%s)
        """, (list(grocery.keys()),))
        NAME = {i: n for i, n in cur.fetchall()}

    # ─── format output ──────────────────────────────────────────
    out = []
    for uid, data in grocery.items():
        cls         = data['class']
        pref1, pref2= CLASS_PREF.get(cls, ("mass", None))
        buckets     = data['buckets']

        dim = (pref1 if buckets[pref1] else
               (pref2 if pref2 and buckets[pref2] else
                next((d for d, v in buckets.items() if v), "count")))
        qty = buckets[dim]

        # user‑unit conversion
        if dim == "mass":
            qty_disp  = qty if USER_UNIT_SYSTEM == "metric" else qty * G2OZ
            unit_disp = "g" if USER_UNIT_SYSTEM == "metric" else "oz"
        elif dim == "volume":
            qty_disp  = qty if USER_UNIT_SYSTEM == "metric" else qty * ML2FLOZ
            unit_disp = "ml" if USER_UNIT_SYSTEM == "metric" else "fl oz"
        else:
            qty_disp, unit_disp = qty, "count"

        out.append(dict(
            usda_id   = uid,
            usda_name = NAME.get(uid, ""),
            dimension = dim,
            quantity  = round(qty_disp, 2),
            unit      = unit_disp,
            items     = data['items']
        ))
    return out

# ────────────────────────────── demo ──────────────────────────────
if __name__ == "__main__":
    TEST = [5476, 5490, 5505, 5541, 5549, 5668, 5687, 5691, 5716, 5508, 5715]
    for row in aggregate(TEST):
        logging.info("[%s] %-40s %6.2f %s",
                     row["usda_id"], row["usda_name"],
                     row["quantity"], row["unit"])

