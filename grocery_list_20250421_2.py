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

import logging, psycopg2, psycopg2.extras
import time
from collections import defaultdict
from math import ceil
from functools import lru_cache
from matcher import best_usda_for_local, _usda_weights, volume_to_count, backfill_units_meta, LOCAL, USDA, candidates, wjacc   # ← your fast matcher

start = time.perf_counter()

# ───────────────────────────── config ──────────────────────────────
LOG = logging.getLogger("grocery")      # keep the global root clean
LOG.setLevel(logging.DEBUG)             # flip to INFO when you’re happy
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

with db() as conn, conn.cursor() as cur:
    cur.execute("""
        SELECT 'ingredient'::text AS kind,
               id::int,
               LOWER(canonical_name)::text,
               NULL::text,
               NULL::text,
               NULL::text
          FROM ingredients

        UNION ALL

        SELECT 'class'::text,
               ingredient_id::int,
               class::text,
               NULL::text,
               NULL::text,
               NULL::text
          FROM ingredient_classes

        UNION ALL

        SELECT 'unit'::text,
               id::int,
               dimension::text,
               conversion_factor::text,
               unit_name::text,
               NULL::text
          FROM units

        UNION ALL

        SELECT 'subtype'::text,
               id::int,
               NULL::text,
               NULL::text,
               NULL::text,
               piece_subtype::text
          FROM usda_ingredient_conversion
    """)

    CANON2ID = {}
    CLASS_OF = {}
    UNIT_CACHE = {}
    SUBTYPE_CACHE = {}

    for kind, a, b, c, d, e in cur.fetchall():
        if kind == "ingredient":
            CANON2ID[b] = a
        elif kind == "class":
            CLASS_OF[a] = b
        elif kind == "unit":
            UNIT_CACHE[a] = (b, float(c) if c is not None else None, d)
        elif kind == "subtype" and e:
            SUBTYPE_CACHE[a] = e

def record_gap(kind: str,
               raw_text: str | None = None,
               local_id: int | None = None,
               usda_id: int | None = None,
               info: dict | None = None):
    """
    Store one line in data_gaps (ignore duplicates –
    we only care that it was seen at least once).
    """
    with db() as c, c.cursor() as cur:
        cur.execute("""
            INSERT INTO data_gaps (gap_type, raw_text, local_id, usda_id, extra_info)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT DO NOTHING
        """, (kind, raw_text, local_id, usda_id, psycopg2.extras.Json(info or {})))
        c.commit()

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

def _add_note(bucket_dict: dict, msg: str):
    """
    Append a note string to bucket_dict['notes'] (create if missing).
    Used for human‑readable conversion tracing.
    """
    bucket_dict.setdefault('notes', []).append(msg)


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

@lru_cache(maxsize=4096)
def usda_by_cached_match(ing_id: int) -> int | None:
    with db() as c, c.cursor() as cur:
        cur.execute("""
            SELECT usda_ingredient_id
              FROM ingredient_usda_matches
             WHERE local_ingredient_id = %s
             ORDER BY rank
             LIMIT 1
        """, (ing_id,))
        if (r := cur.fetchone()):
            return int(r[0])

def unit_info(unit_id):
    return UNIT_CACHE.get(unit_id, (None, None, None))

def best_usda_for_text_inline(raw_text: str) -> tuple[int | None, float]:
    key = -abs(hash(raw_text))
    LOCAL[key] = [(raw_text.lower(), None, 1.0, 1.0)]
    chunks = LOCAL[key]
    cand_ids = candidates(chunks)
    if not cand_ids:
        return None, 0.0
    best_uid, best_sc = max(
        ((uid, wjacc(chunks, USDA[uid])) for uid in cand_ids),
        key=lambda x: x[1]
    )
    return best_uid, best_sc
# ──────────────────────────── aggregation ─────────────────────────
def aggregate(ri_ids):

    """
    Aggregate a list of recipe‑ingredient IDs into a grocery list.

    """
    t0 = time.perf_counter()
    grocery = defaultdict(lambda: {
        "buckets": defaultdict(float),
        "class":   None,
        "items":   []
    })

    # ─── pull parents + children in one trip ─────────────────────
    with db() as c, c.cursor() as cur:
        cur.execute("""
            SELECT ri.id, ri.recipe_id, ri.ingredient_id, ri.ingredient_string,
                q.id, q.quantity, q.unit_id, q.quantity_text, q.sub_label
            FROM recipe_ingredients ri
        LEFT JOIN recipe_ingredient_quantities q
                ON q.recipe_ingredient_id = ri.id
            WHERE ri.id = ANY(%s)
        """, (list(ri_ids),))

        rows = cur.fetchall()

        parents = []
        Q = defaultdict(list)

        for ri_id, recipe_id, local_id, raw, q_id, qty, unit_id, qtxt, sub_label in rows:
            if (ri_id, recipe_id, local_id, raw) not in parents:
                parents.append((ri_id, recipe_id, local_id, raw))

            if q_id is not None:  # skip if there's no quantity row
                Q[ri_id].append(dict(
                    rq_id     = q_id,
                    quantity  = qty,
                    unit_id   = unit_id,
                    qtxt      = qtxt or "",
                    sub_label = (sub_label or "").lower()
                ))
        LOG.debug("⏱ loaded parent + quantity rows in %.3fs", time.perf_counter() - t0)
        t1 = time.perf_counter()

    # ─── walk every parent row ───────────────────────────────────
    for ri_id, recipe_id, local_id, raw in parents:
        LOG.debug("─  parent id=%s  raw='%s'", ri_id, raw)
        if not local_id:                                   # fallback lookup
            local_id = CANON2ID.get(raw.lower().strip())
            LOG.debug("    resolved local_id=%s", local_id)

        # Determine if matching is needed
        skip_matching = False
        if local_id:
            ingr_class = CLASS_OF.get(local_id)
            if ingr_class == "produce":
                all_dims = {unit_info(r['unit_id'])[0] if r['unit_id'] else "count" for r in Q[ri_id]}
                if all_dims == {"count"}:
                    skip_matching = True
                    LOG.debug("🧺 Skipping USDA match for produce in count: %s", raw)  

        if skip_matching:
            usda_id = f"local:{local_id}"  # or just local_id if you're ok with mixing types
            score = 1.0
        else:
            if local_id and (uid := usda_by_cached_match(local_id)):
                usda_id, score = uid, 1.0
                LOG.debug("    cached USDA=%s (score=1.0)", uid)
            else:
                if local_id:
                    usda_id, score = best_usda_for_local(local_id)
                else:
                    token = raw.split(',')[0].split('(')[0]
                    usda_id, score = best_usda_for_text_inline(token.lower())
                LOG.debug("    matcher USDA=%s  score=%.3f", usda_id, score)

        if not usda_id:
            LOG.warning("No USDA match for “%s” – skipped", raw)
            record_gap("no_usda_match", raw_text=raw, local_id=local_id)
            continue

        # ― class
        ingr_class = CLASS_OF.get(local_id, "other")
        grocery[usda_id]['class'] = ingr_class

        # ― choose ONE primary quantity row ──────────────────────
        rows = Q[ri_id]
        if not rows:                                        # no numbers at all
            rows = [dict(rq_id=None, quantity=None, unit_id=None,
                         qtxt="", sub_label="")]
        LOG.debug("    %d quantity rows", len(rows))

        # Determine if matching is needed
        skip_matching = False
        if local_id:
            ingr_class = CLASS_OF.get(local_id)
            if ingr_class == "produce":
                all_dims = {unit_info(r['unit_id'])[0] if r['unit_id'] else "count" for r in rows}
                if all_dims == {"count"}:
                    skip_matching = True
                    LOG.debug("🧺 Skipping USDA match for produce in count: %s", raw)

        # helper: dimension for a row
        def _dim(r):
            return unit_info(r['unit_id'])[0] if r['unit_id'] else None

        mass_rows   = [r for r in rows if _dim(r) == 'mass']
        volume_rows = [r for r in rows if _dim(r) == 'volume']

        if   mass_rows  : primary = max(mass_rows,   key=lambda r: float(r['quantity'] or 0))
        elif volume_rows: primary = max(volume_rows, key=lambda r: float(r['quantity'] or 0))
        else:
            primary = next((r for r in rows if r['sub_label'] == 'a'), rows[0])
            LOG.debug("    primary row=%s", primary)

        # — extract size hint from quantity_text —
        size_hint = None
        if primary['qtxt']:
            for w in {"small", "medium", "large"}:
                if w in primary['qtxt'].lower():
                    size_hint = w
                    break

        # Default to medium if not specified
        size_hint = size_hint or "medium"
        LOG.debug("    size_hint=%s", size_hint)

        # ― convert / cache ──────────────────────────────────────
        rq   = primary['rq_id']
        base = float(primary['quantity']) if primary['quantity'] else 1.0
        dim, cf, _ = unit_info(primary['unit_id']) if primary['unit_id'] else (None, None, None)

        _add_note(grocery[usda_id], f"{primary['qtxt'] or '1×'} "
                                    f"→ {base:g} {dim or 'count'}")

        cache = cached_conversion(rq) if rq else None
        if cache:
            _, ref_qty, ref_dim = cache
        else:
            ref_qty, ref_dim = base, "count"
            if dim == "mass"   and cf: ref_qty, ref_dim = base * cf, "mass"
            if dim == "volume" and cf: ref_qty, ref_dim = base * cf, "volume"
            LOG.debug("    -> %.3f %s", ref_qty, ref_dim)
            if rq:                                                 # only store when we have an rq_id
                save_conversion(rq, usda_id, ref_qty, ref_dim)

        # ─── use subtype if available for piece dimension ────────
        subtype = SUBTYPE_CACHE.get(rq)

        # modified key logic
        if ref_dim == "count" and subtype:
            key = f"piece::{subtype}"
        else:
            key = ref_dim

        grocery[usda_id]['buckets'][key] += ref_qty

        grocery[usda_id]['items'].append(dict(
            recipe_id = recipe_id,
            rq_id     = rq,
            ref_qty   = ref_qty,
            ref_dim   = ref_dim,
            subtype   = subtype,
            original  = primary['qtxt']
        ))

    LOG.debug("⏱ matching USDA ingredients took %.3fs", time.perf_counter() - t1)
    t2 = time.perf_counter()

    # ─── nothing aggregated ─────────────────────────────────────
    if not grocery:
        return []

    # ─── resolve USDA names ─────────────────────────────────────
    with db() as c, c.cursor() as cur:
        real_usda_ids = [uid for uid in grocery if isinstance(uid, int)]
        with db() as c, c.cursor() as cur:
            cur.execute("""
                SELECT id, canonical_name
                FROM usda_ingredients
                WHERE id = ANY(%s)
            """, (real_usda_ids,))
            NAME = {i: n for i, n in cur.fetchall()}

    # ─── post‑process produce: convert vol/mass → count ──────────────
    for uid, data in grocery.items():
        if data['class'] != 'produce':
            continue

        if isinstance(uid, str) and uid.startswith("local:"):
            continue  # skip conversion logic, we already have count

        b = data['buckets']
        piece_wt, cup_wt = _usda_weights(uid, size_hint)

        if piece_wt:
            _add_note(data, f"(USDA) 1 piece ({size_hint}) = {piece_wt:.2f} g")
        if cup_wt:
            _add_note(data, f"(USDA) 1 cup = {cup_wt:.2f} g")


        # --- sanity gaps ---------------------------------------------
        if b['volume'] and cup_wt is None:
            record_gap("no_cup_wt", usda_id=uid,
                    info={"volume_ml": float(b['volume'])})
        if b['mass'] and piece_wt is None:
            record_gap("no_piece_wt", usda_id=uid,
                    info={"mass_g": float(b['mass'])})
            
        # ① volume → count directly if possible
        pcs_per_cup = volume_to_count(uid, size_hint)
        if b['volume'] and pcs_per_cup:
            pcs_float = b['volume'] / 240.0 * pcs_per_cup
            pcs = ceil(pcs_float)
            _add_note(data, f"{b['volume']:.2f} ml × {pcs_per_cup:.2f} pcs/cup = {pcs_float:.2f} → {pcs} pcs")
            b['count'] += pcs
            b['volume'] = 0.0

        # ② mass → pieces
        if b['mass'] and piece_wt:
            pcs_float = b['mass'] / piece_wt
            pcs = ceil(pcs_float)                     # ← round *up*
            _add_note(data,
                f"{b['mass']:.2f} g ÷ {piece_wt:.2f} g/pc = {pcs_float:.2f} → {pcs} pcs")
            b['count'] += pcs
            b['mass']   = 0.0

    LOG.debug("⏱ converted + aggregated all units in %.3fs", time.perf_counter() - t2)
    t3 = time.perf_counter()

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
            qty_disp  = ceil(qty)         # round up display
            unit_disp = "count"


        out.append(dict(
            usda_id   = uid,
            usda_name = NAME.get(uid, ""),
            dimension = dim,
            quantity  = round(qty_disp, 2),
            unit      = unit_disp,
            items     = data['items'],
            trace     = data.get('notes', [])          #  ←  NEW
        ))
    LOG.info("✅ TOTAL: aggregate() completed in %.3fs", time.perf_counter() - t0)

    return out

# ────────────────────────────── demo ──────────────────────────────
if __name__ == "__main__":
    backfill_units_meta()
    _usda_weights.cache_clear()

    TEST = [5476, 5490, 5505, 5541, 5549, 5668, 5687, 5691, 5716, 5508, 5715]
    total_count = 0
    for row in aggregate(TEST):
        logging.info("[%s] %-40s %6.2f %s",
                    row["usda_id"], row["usda_name"],
                    row["quantity"], row["unit"])
        if row["unit"] == "count":
            total_count += row["quantity"]
        for step in row.get("trace", []):
            print("     •", step)

    # 👇 Show final total
    print(f"\n🧮 Final total: {total_count:.0f} scallions\n")



