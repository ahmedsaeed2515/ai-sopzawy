# PHASE 2 — BUILD THE SMART BOT
# Run ONLY after Phase 1 is complete and verified

---

## PREREQUISITE — READ PHASE 0 + 1 OUTPUT FIRST

Before writing a single line of code, confirm these from your previous phases:

```
From Phase 0:
─────────────────────────────────────────────────────────────
  RAG file path              : [e.g. aiShopzawy/rag/search.py]
  RAG function signature     : [e.g. search(query: str, top_k: int) -> dict]
  RAG return structure       : [e.g. {"documents": [...], "metadatas": [...]}]
  Groq model name            : [e.g. "llama-3.3-70b-versatile"]
  GROQ_API_KEY env var name  : [e.g. GROQ_API_KEY]

From Phase 1 terminal output:
─────────────────────────────────────────────────────────────
  SQLite path  : aiShopzawy/database/shopzawy.db
  All category names (copy the exact list from Phase 1):
    [category 1]
    [category 2]
    ...
  All brand names (copy the exact list from Phase 1):
    [brand 1]
    [brand 2]
    ...
─────────────────────────────────────────────────────────────
```

You MUST replace:
- `# ← ADAPT: RAG import` with the real import line
- `# ← ADAPT: paste real categories` with the real category names
- `# ← ADAPT: paste real brands` with the real brand names

---

## WHAT YOU ARE BUILDING

```
File   : aiShopzawy/agent/bot.py

Architecture: Function Calling
  User message
      → LLM Round 1  →  decides which tool(s) to call
      → Tools execute (parallel if multiple)
      → LLM Round 2  →  formats a human-like Arabic response

Tools:
  search_by_meaning     →  Chroma RAG   (descriptions, recommendations, features)
  search_by_price       →  SQLite       (cheapest / most expensive / by brand+category)
  filter_by_price_range →  SQLite       (under X / between X and Y / above X)
  get_best_deals        →  SQLite       (highest discount %)
  compare_products      →  SQLite+FTS   (side-by-side comparison)
  check_availability    →  SQLite+FTS   (is this specific product in stock?)
```

---

## STEP 1 — CREATE `aiShopzawy/agent/bot.py`

Write the complete file below in one pass.
Replace every `# ← ADAPT` comment with the real value.

```python
"""
Shopzawy Smart Sales Bot
========================
Architecture : Function Calling (Groq LLM + 6 tools)
SQLite source: aiShopzawy/database/shopzawy.db
RAG source   : Chroma vector store

Flow per message:
  1. LLM Round 1  — reads user message, picks tool(s)
  2. Tool(s) run  — SQLite or RAG, parallel when multiple
  3. LLM Round 2  — receives real data, writes Arabic response
"""

import sqlite3
import json
import os
import sys
import time
import logging
import concurrent.futures
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
log = logging.getLogger("shopzawy_bot")


# ── Paths ────────────────────────────────────────────────────────────────────
_HERE     = os.path.dirname(os.path.abspath(__file__))
DB_PATH   = os.path.abspath(os.path.join(_HERE, "../database/shopzawy.db"))
ROOT_PATH = os.path.abspath(os.path.join(_HERE, "../"))

if not os.path.exists(DB_PATH):
    raise FileNotFoundError(
        f"SQLite database not found at: {DB_PATH}\n"
        "Run Phase 1 first: python aiShopzawy/database/build_sqlite.py"
    )


# ── RAG Import ───────────────────────────────────────────────────────────────
sys.path.insert(0, ROOT_PATH)

try:
    from rag.search import search as _rag_fn   # ← ADAPT: real import from Phase 0
    RAG_AVAILABLE = True
    log.info("RAG loaded OK")
except ImportError as e:
    RAG_AVAILABLE = False
    log.warning(f"RAG unavailable: {e} — semantic search will be disabled")
    def _rag_fn(query, **kwargs):
        return []


# ── Groq ─────────────────────────────────────────────────────────────────────
_api_key = os.getenv("GROQ_API_KEY")   # ← ADAPT: env var name from Phase 0
if not _api_key:
    raise EnvironmentError("GROQ_API_KEY not found in .env")

client = Groq(api_key=_api_key)
MODEL  = "llama-3.3-70b-versatile"     # ← ADAPT: model name from Phase 0


# ── Known Categories & Brands (from Phase 1 output) ─────────────────────────
# Replace these lists with the REAL names printed by Phase 1.
# These are used by _match_category() and _match_brand() to normalise
# what the LLM passes into tool arguments.

KNOWN_CATEGORIES: list[str] = [
    # ← ADAPT: paste real category names from Phase 1
    # Example:
    # "موبايلات",
    # "لاب توب",
    # "شنط",
    # "ملابس",
]

KNOWN_BRANDS: list[str] = [
    # ← ADAPT: paste real brand names from Phase 1
    # Example:
    # "Samsung",
    # "Apple",
    # "Huawei",
]


def _match_category(hint: str | None) -> str | None:
    """
    Fuzzy-match a category hint (from LLM) against known category names.
    Returns the best DB-safe category string, or None if no match.
    """
    if not hint:
        return None
    hint_lower = hint.lower().strip()
    # Exact match first
    for cat in KNOWN_CATEGORIES:
        if cat.lower() == hint_lower:
            return cat
    # Partial match
    for cat in KNOWN_CATEGORIES:
        if hint_lower in cat.lower() or cat.lower() in hint_lower:
            return cat
    # Fallback: return raw hint (SQL uses LIKE so partial works)
    return hint


def _match_brand(hint: str | None) -> str | None:
    """Same logic for brands."""
    if not hint:
        return None
    hint_lower = hint.lower().strip()
    for brand in KNOWN_BRANDS:
        if brand.lower() == hint_lower:
            return brand
    for brand in KNOWN_BRANDS:
        if hint_lower in brand.lower() or brand.lower() in hint_lower:
            return brand
    return hint


# ══════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """
أنت "شوبي" — مساعد مبيعات ذكي لمتجر Shopzawy الإلكتروني.

## شخصيتك
- ودود، سريع، محترف — زي أفضل مندوب مبيعات
- واثق لأن بياناتك من قاعدة البيانات الحقيقية مباشرة، مش تخمين
- صادق — لو المنتج مش موجود أو خلص، قول كده واقترح بديل قريب
- مختصر — ردود عملية وواضحة (أقل من 150 كلمة إلا في المقارنات)

## أدواتك — استخدمها دايماً لأسئلة المنتجات
1. search_by_meaning     → للبحث بالوصف والخصائص والاقتراحات
2. search_by_price       → الأرخص / الأغلى / بفئة أو براند
3. filter_by_price_range → تحت X / بين X وY / فوق X
4. get_best_deals        → العروض والخصومات والتخفيضات
5. compare_products      → مقارنة منتجين أو أكتر جنب بعض
6. check_availability    → هل منتج معين موجود وسعره كام

## قرار الأداة — اتبع هذا الجدول
ارخص / أقل سعر / cheapest           → search_by_price(order=asc)
أغلى / أفخم / أحسن / premium        → search_by_price(order=desc)
تحت X / بين X وY / فوق X            → filter_by_price_range
عروض / خصومات / تخفيضات / offers   → get_best_deals
قارن / الفرق بين / أيهم أحسن       → compare_products
هل عندكم X؟ / X متوفر؟ / سعر X     → check_availability
اقترح / محتاج / عندكم إيه / هدية   → search_by_meaning

## مزج الأدوات — جائز ومحبّذ
"عايز موبايل سامسونج رخيص ومتوفر"
→ استخدم search_by_price + check_availability معاً

## قواعد صارمة
- الأسعار والمخزون من الأداة دايماً — لا تخترع أرقام أبداً
- لو مفيش نتائج — اعترف بصدق واقترح بديل أو فئة مختلفة
- رد بنفس لغة العميل (عربي / إنجليزي / مزيج)
- اختم بـ call-to-action خفيف (أضفه للسلة؟ تحب تشوف صورته؟)
- لو السؤال مش عن المتجر — وجّه الحوار للمنتجات برفق
"""


# ══════════════════════════════════════════════════════════════════════════════
# TOOL DEFINITIONS — the JSON schema the LLM sees
# ══════════════════════════════════════════════════════════════════════════════

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_by_meaning",
            "description": (
                "Semantic search using product descriptions, features, and use cases. "
                "Use for: gift suggestions, 'I need something for X', feature-based queries, "
                "open-ended browsing, or any question not about a specific price or availability."
            ),
            "parameters": {
                "type": "object",
                "required": ["query"],
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search text — use the customer's words or a refined version"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results (default: 5, max: 10)"
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_by_price",
            "description": (
                "Returns products sorted by price. "
                "order=asc → cheapest first. order=desc → most expensive first. "
                "Use for: 'ارخص موبايل', 'أغلى حاجة عندكم', 'أحسن X'."
            ),
            "parameters": {
                "type": "object",
                "required": ["order"],
                "properties": {
                    "order": {
                        "type": "string",
                        "enum": ["asc", "desc"],
                        "description": "asc = cheapest first | desc = most expensive first"
                    },
                    "category": {
                        "type": "string",
                        "description": "Filter by category name — use exact names from the store"
                    },
                    "brand": {
                        "type": "string",
                        "description": "Filter by brand name — use exact names from the store"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of results (default: 5, max: 10)"
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "filter_by_price_range",
            "description": (
                "Returns in-stock products within a price range, sorted cheapest first. "
                "Use for: 'تحت 500 جنيه', 'بين 200 و1000', 'فوق 2000', 'under 300 EGP'."
            ),
            "parameters": {
                "type": "object",
                "required": ["min_price", "max_price"],
                "properties": {
                    "min_price": {
                        "type": "number",
                        "description": "Minimum price — use 0 if not specified"
                    },
                    "max_price": {
                        "type": "number",
                        "description": "Maximum price — use 999999 if not specified"
                    },
                    "category": {"type": "string"},
                    "brand":    {"type": "string"},
                    "limit":    {
                        "type": "integer",
                        "description": "Number of results (default: 8, max: 15)"
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_best_deals",
            "description": (
                "Returns products with the highest active discount percentage. "
                "Use for: 'عروض', 'خصومات', 'تخفيضات', 'offers', 'deals', 'أحسن صفقة'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "Filter deals by category (optional)"
                    },
                    "min_discount": {
                        "type": "number",
                        "description": "Minimum discount % to include (default: 1)"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of results (default: 8)"
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compare_products",
            "description": (
                "Fetches detailed data for 2+ products for side-by-side comparison. "
                "Returns price, stock, discount, variants, and description for each. "
                "Use for: 'قارن سامسونج وآبل', 'الفرق بين X وY', 'أيهم أحسن'."
            ),
            "parameters": {
                "type": "object",
                "required": ["product_names"],
                "properties": {
                    "product_names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of product names or brands to compare (2–4 items)"
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_availability",
            "description": (
                "Checks if a specific product exists, its current price, and stock level. "
                "Use for: 'هل عندكم X?', 'X متوفر؟', 'سعر X كام?', 'ابحث عن X'."
            ),
            "parameters": {
                "type": "object",
                "required": ["product_name"],
                "properties": {
                    "product_name": {
                        "type": "string",
                        "description": "Product name, brand, or partial name to search for"
                    }
                }
            }
        }
    }
]


# ══════════════════════════════════════════════════════════════════════════════
# DATABASE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _db() -> sqlite3.Connection:
    """Open a read-only SQLite connection with Row factory."""
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA cache_size=-16000")
    return conn


def _rows_to_str(rows: list[dict], header: str = "") -> str:
    """
    Format a list of product dicts into clean text for LLM consumption.
    Shows: name | brand | category | price (with discount) | stock | variants
    """
    if not rows:
        return "لا توجد منتجات مطابقة في قاعدة البيانات."

    lines = []
    if header:
        lines.append(header)

    for i, r in enumerate(rows, 1):
        sale = r.get("sale_price") or 0
        base = r.get("base_price") or 0
        disc = r.get("discount_percent") or 0
        qty  = r.get("stock_quantity") or 0

        # Price text
        if disc > 0 and base > sale:
            price_text = f"{sale} EGP (was {base} — {disc}% off)"
        else:
            price_text = f"{sale} EGP"

        # Stock text
        if r.get("in_stock") and qty > 0:
            stock_text = f"✅ In stock ({qty})"
        elif r.get("in_stock"):
            stock_text = "✅ In stock"
        else:
            stock_text = "❌ Out of stock"

        # Variants summary
        variant_text = ""
        try:
            vlist  = json.loads(r.get("variants") or "[]")
            colors = sorted(set(v.get("color", "") for v in vlist if v.get("color")))
            sizes  = sorted(set(v.get("size",  "") for v in vlist if v.get("size")))
            if colors: variant_text += f" | Colors: {', '.join(colors)}"
            if sizes:  variant_text += f" | Sizes: {', '.join(sizes)}"
        except Exception:
            pass

        lines.append(
            f"{i}. {r.get('name_ar', 'N/A')}"
            f" | {r.get('brand_name', '')}"
            f" | {r.get('category_name', '')}"
            f" | {price_text}"
            f" | {stock_text}"
            f"{variant_text}"
        )

    return "\n".join(lines)


def _fts_search(conn: sqlite3.Connection, terms: list[str], limit: int = 10) -> list[dict]:
    """
    Search products using the FTS5 virtual table built in Phase 1.
    Falls back to LIKE search if FTS returns nothing.
    """
    if not terms:
        return []

    # Try FTS5 first (fast, Arabic-aware tokenization)
    fts_query = " OR ".join(terms)
    try:
        rows = conn.execute(f"""
            SELECT p.name_ar, p.brand_name, p.category_name,
                   p.sale_price, p.base_price, p.discount_percent,
                   p.in_stock, p.stock_quantity, p.variants
            FROM products_fts fts
            JOIN products p ON p.id = fts.rowid
            WHERE products_fts MATCH ?
              AND p.is_published = 1
            ORDER BY p.in_stock DESC, p.sale_price ASC
            LIMIT ?
        """, (fts_query, limit)).fetchall()
        if rows:
            return [dict(r) for r in rows]
    except Exception as e:
        log.warning(f"FTS search failed ({e}), falling back to LIKE")

    # LIKE fallback
    conditions = " OR ".join(
        ["name_ar LIKE ? OR brand_name LIKE ? OR category_name LIKE ?" for _ in terms]
    )
    params: list = []
    for t in terms:
        params += [f"%{t}%", f"%{t}%", f"%{t}%"]
    params.append(limit)

    rows = conn.execute(f"""
        SELECT name_ar, brand_name, category_name,
               sale_price, base_price, discount_percent,
               in_stock, stock_quantity, variants
        FROM products
        WHERE is_published = 1 AND ({conditions})
        ORDER BY in_stock DESC, sale_price ASC
        LIMIT ?
    """, params).fetchall()
    return [dict(r) for r in rows]


# ══════════════════════════════════════════════════════════════════════════════
# TOOL IMPLEMENTATIONS
# ══════════════════════════════════════════════════════════════════════════════

def fn_search_by_price(
    order: str = "asc",
    category: str | None = None,
    brand: str | None = None,
    limit: int = 5,
) -> str:
    category = _match_category(category)
    brand    = _match_brand(brand)
    limit    = min(int(limit or 5), 10)

    sql    = """
        SELECT name_ar, brand_name, category_name,
               sale_price, base_price, discount_percent,
               in_stock, stock_quantity, variants
        FROM products
        WHERE in_stock = 1 AND is_published = 1 AND sale_price > 0
    """
    params: list = []
    if category:
        sql += " AND category_name LIKE ?"
        params.append(f"%{category}%")
    if brand:
        sql += " AND brand_name LIKE ?"
        params.append(f"%{brand}%")
    sql += f" ORDER BY sale_price {'ASC' if order == 'asc' else 'DESC'} LIMIT ?"
    params.append(limit)

    conn = _db()
    rows = [dict(r) for r in conn.execute(sql, params).fetchall()]
    conn.close()

    if not rows and (category or brand):
        return (
            f"لم أجد منتجات{'  في فئة ' + category if category else ''}"
            f"{'  من براند ' + brand if brand else ''}.\n"
            "جرّب فئة أو براند مختلف."
        )
    return _rows_to_str(rows)


def fn_filter_by_price_range(
    min_price: float = 0,
    max_price: float = 999_999,
    category: str | None = None,
    brand: str | None = None,
    limit: int = 8,
) -> str:
    category  = _match_category(category)
    brand     = _match_brand(brand)
    min_price = max(float(min_price or 0), 0)
    max_price = float(max_price or 999_999)
    limit     = min(int(limit or 8), 15)

    sql    = """
        SELECT name_ar, brand_name, category_name,
               sale_price, base_price, discount_percent,
               in_stock, stock_quantity, variants
        FROM products
        WHERE in_stock = 1 AND is_published = 1
          AND sale_price BETWEEN ? AND ?
    """
    params: list = [min_price, max_price]
    if category:
        sql += " AND category_name LIKE ?"
        params.append(f"%{category}%")
    if brand:
        sql += " AND brand_name LIKE ?"
        params.append(f"%{brand}%")
    sql += " ORDER BY sale_price ASC LIMIT ?"
    params.append(limit)

    conn = _db()
    rows = [dict(r) for r in conn.execute(sql, params).fetchall()]
    conn.close()

    header = f"منتجات بسعر بين {min_price} و{max_price} جنيه:"
    if not rows:
        return (
            f"لم أجد منتجات في النطاق السعري {min_price}–{max_price} جنيه"
            + (f" في فئة {category}" if category else "") + ".\n"
            "جرّب نطاق أوسع أو فئة مختلفة."
        )
    return _rows_to_str(rows, header)


def fn_get_best_deals(
    category: str | None = None,
    min_discount: float = 1,
    limit: int = 8,
) -> str:
    category     = _match_category(category)
    min_discount = max(float(min_discount or 1), 1)
    limit        = min(int(limit or 8), 15)

    sql    = """
        SELECT name_ar, brand_name, category_name,
               sale_price, base_price, discount_percent,
               in_stock, stock_quantity, variants
        FROM products
        WHERE in_stock = 1 AND is_published = 1
          AND discount_percent >= ?
    """
    params: list = [min_discount]
    if category:
        sql += " AND category_name LIKE ?"
        params.append(f"%{category}%")
    sql += " ORDER BY discount_percent DESC LIMIT ?"
    params.append(limit)

    conn = _db()
    rows = [dict(r) for r in conn.execute(sql, params).fetchall()]
    conn.close()

    if not rows:
        return "لا توجد عروض أو خصومات نشطة حالياً."
    return _rows_to_str(rows, f"🔥 أحسن {len(rows)} عروض دلوقتي:")


def fn_compare_products(product_names: list[str]) -> str:
    if not product_names or len(product_names) < 2:
        return "محتاج اسمين على الأقل للمقارنة."

    product_names = product_names[:4]   # cap at 4

    conn      = _db()
    all_rows: list[dict] = []

    for name in product_names:
        # Try FTS first, then LIKE
        rows = _fts_search(conn, [name], limit=2)
        if not rows:
            rows = [dict(r) for r in conn.execute("""
                SELECT name_ar, brand_name, category_name,
                       sale_price, base_price, discount_percent,
                       in_stock, stock_quantity, description, variants
                FROM products
                WHERE is_published = 1
                  AND (name_ar LIKE ? OR brand_name LIKE ? OR slug LIKE ?)
                ORDER BY in_stock DESC, sale_price ASC
                LIMIT 2
            """, [f"%{name}%"] * 3).fetchall()]
        all_rows.extend(rows)

    conn.close()

    if not all_rows:
        return "لم أجد المنتجات المطلوبة. تأكد من الأسماء وحاول مرة أخرى."

    # Deduplicate by name
    seen, unique_rows = set(), []
    for r in all_rows:
        key = r.get("name_ar", "")
        if key not in seen:
            seen.add(key)
            unique_rows.append(r)

    lines = [f"مقارنة {len(unique_rows)} منتجات:\n"]
    for i, r in enumerate(unique_rows, 1):
        sale = r.get("sale_price") or 0
        base = r.get("base_price") or 0
        disc = r.get("discount_percent") or 0
        qty  = r.get("stock_quantity") or 0
        stock_text = (
            f"✅ متوفر ({qty} قطعة)" if (r.get("in_stock") and qty > 0)
            else ("✅ متوفر" if r.get("in_stock") else "❌ نفد المخزون")
        )
        desc = (r.get("description") or "")[:250].strip()

        try:
            vlist  = json.loads(r.get("variants") or "[]")
            colors = sorted(set(v.get("color", "") for v in vlist if v.get("color")))
            sizes  = sorted(set(v.get("size",  "") for v in vlist if v.get("size")))
        except Exception:
            colors, sizes = [], []

        lines.append(
            f"{'─'*50}\n"
            f"#{i} {r.get('name_ar', 'N/A')}\n"
            f"  البراند   : {r.get('brand_name', '—')}\n"
            f"  الفئة     : {r.get('category_name', '—')}\n"
            f"  السعر     : {sale} جنيه"
            + (f" (كان {base} — خصم {disc}%)" if disc > 0 else "") + "\n"
            f"  المخزون   : {stock_text}\n"
            + (f"  الألوان   : {', '.join(colors)}\n" if colors else "")
            + (f"  المقاسات  : {', '.join(sizes)}\n"  if sizes  else "")
            + (f"  الوصف     : {desc}\n"              if desc   else "")
        )
    return "\n".join(lines)


def fn_check_availability(product_name: str) -> str:
    if not product_name:
        return "لم يتم تحديد اسم المنتج."

    conn  = _db()
    # Try FTS first
    rows  = _fts_search(conn, [product_name], limit=5)
    if not rows:
        term = f"%{product_name}%"
        rows = [dict(r) for r in conn.execute("""
            SELECT name_ar, brand_name, category_name,
                   sale_price, base_price, discount_percent,
                   in_stock, stock_quantity, variants
            FROM products
            WHERE is_published = 1
              AND (name_ar LIKE ? OR name_en LIKE ?
                   OR slug LIKE ? OR brand_name LIKE ?)
            ORDER BY in_stock DESC, sale_price ASC
            LIMIT 5
        """, [term, term, term, term]).fetchall()]
    conn.close()

    if not rows:
        return (
            f"لم أجد '{product_name}' في المتجر.\n"
            "جرّب: اسم مختلف، براند، أو فئة المنتج."
        )
    return _rows_to_str(rows, f"نتائج البحث عن '{product_name}':")


def fn_search_by_meaning(query: str, top_k: int = 5) -> str:
    if not RAG_AVAILABLE:
        # Graceful degradation: fall back to FTS keyword search
        log.warning("RAG unavailable — falling back to FTS keyword search")
        conn  = _db()
        terms = [w for w in query.split() if len(w) > 2][:5]
        rows  = _fts_search(conn, terms, limit=top_k)
        conn.close()
        return _rows_to_str(rows, f"نتائج بحث '{query}' (keyword fallback):")

    try:
        top_k   = min(int(top_k or 5), 10)
        results = _rag_fn(query, top_k=top_k)   # ← ADAPT if signature differs

        # Parse return format
        if isinstance(results, dict):
            docs  = results.get("documents") or []
            metas = results.get("metadatas") or []
        elif isinstance(results, list):
            docs, metas = results, []
        else:
            docs, metas = [str(results)], []

        if not docs:
            return "لم أجد نتائج مشابهة. جرّب وصف مختلف."

        lines = [f"نتائج البحث الدلالي ({len(docs)} منتج):"]
        for i, doc in enumerate(docs[:top_k], 1):
            text  = doc if isinstance(doc, str) else json.dumps(doc, ensure_ascii=False)
            meta  = metas[i - 1] if i <= len(metas) else {}
            extra = ""
            if isinstance(meta, dict):
                price    = meta.get("price") or meta.get("sale_price")
                in_stock = meta.get("in_stock")
                if price:     extra += f" — {price} جنيه"
                if in_stock is False: extra += " (❌ نفد)"
            lines.append(f"{i}. {text[:350]}{extra}")

        return "\n".join(lines)

    except Exception as e:
        log.error(f"RAG error: {e}")
        return f"حدث خطأ في البحث الدلالي: {e}"


# ── Tool Registry ────────────────────────────────────────────────────────────

TOOL_FUNCTIONS: dict = {
    "search_by_meaning":     fn_search_by_meaning,
    "search_by_price":       fn_search_by_price,
    "filter_by_price_range": fn_filter_by_price_range,
    "get_best_deals":        fn_get_best_deals,
    "compare_products":      fn_compare_products,
    "check_availability":    fn_check_availability,
}


# ══════════════════════════════════════════════════════════════════════════════
# THE BOT
# ══════════════════════════════════════════════════════════════════════════════

class ShopyBot:
    """
    Shopzawy smart sales bot.

    Usage:
        bot = ShopyBot()
        reply = bot.chat("ارخص موبايل سامسونج عندكم؟")
        bot.reset()   # clear conversation history
    """

    MAX_HISTORY = 8   # keep last 4 exchanges (user + assistant pairs)

    def __init__(self):
        self.history: list[dict] = []
        log.info(f"ShopyBot ready | DB: {DB_PATH} | RAG: {RAG_AVAILABLE}")

    # ── Public: chat ─────────────────────────────────────────────────────────
    def chat(self, user_message: str) -> str:
        """Send a message and get a response."""
        self.history.append({"role": "user", "content": user_message})

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *self.history[-self.MAX_HISTORY:],
        ]

        # ── Round 1: LLM decides which tool(s) to call ───────────────────────
        t0 = time.time()
        r1 = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.15,    # low → consistent tool selection
            max_tokens=800,
        )

        msg = r1.choices[0].message

        # ── No tool call → LLM answered directly (greetings, off-topic) ─────
        if not msg.tool_calls:
            reply = msg.content or "عذراً، حدث خطأ. أعد صياغة سؤالك."
            self.history.append({"role": "assistant", "content": reply})
            log.info(f"Direct reply ({time.time()-t0:.1f}s) — no tools called")
            return reply

        # ── Execute all tools (parallel) ─────────────────────────────────────
        tool_names = [tc.function.name for tc in msg.tool_calls]
        log.info(f"Tools called: {tool_names}")
        messages.append(msg)

        def _run(tc):
            fn = TOOL_FUNCTIONS.get(tc.function.name)
            if not fn:
                return tc.id, f"[Tool '{tc.function.name}' not found]"
            try:
                args   = json.loads(tc.function.arguments or "{}")
                result = fn(**args)
                log.info(f"  {tc.function.name} → {len(str(result))} chars")
                return tc.id, str(result)
            except Exception as e:
                log.error(f"  {tc.function.name} error: {e}")
                return tc.id, f"[Tool error: {e}]"

        tool_results: dict[str, str] = {}
        max_workers = min(len(msg.tool_calls), 3)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            for tid, result in pool.map(_run, msg.tool_calls):
                tool_results[tid] = result

        for tc in msg.tool_calls:
            messages.append({
                "role":         "tool",
                "tool_call_id": tc.id,
                "content":      tool_results.get(tc.id, "لا توجد نتائج."),
            })

        # ── Round 2: LLM formats the final response ───────────────────────────
        r2 = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.5,
            max_tokens=700,
        )

        reply = r2.choices[0].message.content or "عذراً، حدث خطأ في تنسيق الرد."
        self.history.append({"role": "assistant", "content": reply})
        log.info(f"Response ready ({time.time()-t0:.1f}s total)")
        return reply

    # ── Public: reset ─────────────────────────────────────────────────────────
    def reset(self):
        """Clear conversation history (start a new session)."""
        self.history.clear()
        log.info("Conversation history cleared")


# ══════════════════════════════════════════════════════════════════════════════
# QUICK SMOKE TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    bot = ShopyBot()

    TESTS = [
        ("cheapest by category", "ارخص موبايل عندكم؟"),
        ("cheapest by brand",    "ارخص منتج سامسونج عندكم؟"),
        ("price range",          "عندكم حاجة تحت 500 جنيه؟"),
        ("best deals",           "إيه أحسن عروض دلوقتي؟"),
        ("availability",         "هل عندكم آيفون؟"),
        ("comparison",           "قارن سامسونج وهواوي عندكم"),
        ("semantic",             "اقترحلي هدية لبنت عمرها 20 سنة"),
        ("compound",             "عايز موبايل رخيص ومتوفر دلوقتي"),
    ]

    print("=" * 65)
    print("  SHOPZAWY BOT — SMOKE TEST")
    print("=" * 65)

    passed = failed = 0
    for desc, question in TESTS:
        print(f"\n[{desc}]")
        print(f"❓  {question}")
        try:
            reply = bot.chat(question)
            print(f"🤖  {reply[:300]}")
            passed += 1
        except Exception as e:
            print(f"❌  ERROR: {e}")
            failed += 1
        bot.reset()
        print("─" * 65)

    print(f"\n  Results: {passed} passed / {failed} failed")
    print("  ✅ Ready for Phase 3" if failed == 0 else "  ❌ Fix errors before Phase 3")
```

---

## STEP 2 — ADAPT THE THREE SECTIONS

After creating the file, do these three adaptations:

### 2.1 — Fix the RAG import
Replace this line:
```python
from rag.search import search as _rag_fn   # ← ADAPT: real import from Phase 0
```
With the real path and function name from Phase 0, e.g.:
```python
from aiShopzawy.rag.embeddings import semantic_search as _rag_fn
```

### 2.2 — Paste real categories
Replace `KNOWN_CATEGORIES` with the exact list from Phase 1 output:
```python
KNOWN_CATEGORIES = [
    "موبايلات",      # ← paste real names here
    "لاب توب",
    "شنط وحقائب",
    ...
]
```

### 2.3 — Paste real brands
Replace `KNOWN_BRANDS` with the exact list from Phase 1 output:
```python
KNOWN_BRANDS = [
    "Samsung",       # ← paste real names here
    "Apple",
    "Huawei",
    ...
]
```

---

## STEP 3 — RUN THE SMOKE TEST

```bash
cd [monorepo root]
python aiShopzawy/agent/bot.py
```

---

## STEP 4 — SHOW THE COMPLETE OUTPUT

Paste the full terminal output for all 8 test questions.

For each question, verify:

| Check | Expected |
|-------|----------|
| No Python exceptions | ✅ |
| Price numbers appear | numbers like `250 EGP`, not `0` |
| Arabic response | coherent Arabic text |
| Stock mentioned | متوفر or نفد |
| Correct tool called | shown in logs |

---

## STEP 5 — FIX COMMON PROBLEMS

**Problem: RAG import error**
```
Fix: Open Phase 0 report → find exact file path → fix the import line
```

**Problem: `sale_price = 0` in all responses**
```
Fix: Open Phase 1 report → check extract_prices() found real values
Run: SELECT name_ar, sale_price FROM products LIMIT 5;
```

**Problem: Category filter returns nothing**
```
Fix: KNOWN_CATEGORIES list is wrong
Run: SELECT DISTINCT category_name FROM products;
Then replace the list with real values.
```

**Problem: FTS search returns nothing**
```
Fix: FTS table was not built correctly in Phase 1
Run: SELECT COUNT(*) FROM products_fts;
If 0, re-run build_sqlite.py — the FTS triggers should auto-populate it.
```

**Problem: "Tool not found" in logs**
```
Fix: A tool name in TOOLS list does not match a key in TOOL_FUNCTIONS dict
Check: print(list(TOOL_FUNCTIONS.keys()))
```

---

## STEP 6 — CONFIRM AND STOP

After all 8 tests pass without errors, write exactly:

```
Phase 2 complete ✅
  All 8 smoke tests passed
  RAG available   : YES / NO (if NO — fallback is FTS, still OK)
  Tools working   :
    ✅ search_by_meaning
    ✅ search_by_price
    ✅ filter_by_price_range
    ✅ get_best_deals
    ✅ compare_products
    ✅ check_availability
  Bot file        : aiShopzawy/agent/bot.py

Ready to proceed to PHASE 3.
```

**Do not start Phase 3 until this confirmation block is shown.**
