# PHASE 4 — FULL END-TO-END TEST SUITE
# Run ONLY after Phases 1, 2, and 3 are all complete and verified

---

## PREREQUISITE — ADAPT TEST CASES BEFORE RUNNING

Before writing this file, read the Phase 1 output and replace the
placeholder values in `TEST_CASES` with REAL names from your database:

```
From Phase 1 output:
─────────────────────────────────────────────────────────────────
  A real cheap category   : [e.g. "موبايلات" / "إلكترونيات"]
  A real expensive item   : [e.g. "لاب توب" / "تابلت"]
  A real brand name       : [e.g. "Samsung" / "سامسونج"]
  A second brand name     : [e.g. "Huawei" / "هواوي"]
  A real product name     : [e.g. "Galaxy A15" — pick one you know exists]
  A category with deals   : [e.g. a category that has discount_percent > 0]
  Price under threshold   : [e.g. 300 — pick a number with actual results]
  Price range min/max     : [e.g. 500–1500 — pick a range with actual results]
─────────────────────────────────────────────────────────────────
```

Replace every `# ← ADAPT` comment in TEST_CASES with these real values.

---

## WHAT THIS PHASE TESTS

```
5 test layers — run in order:

  Layer 1 — Database Integrity
    Verifies the SQLite DB itself is healthy before touching the bot.
    Checks: counts, prices, FTS, variants, sync_log.

  Layer 2 — SQL Tool Queries
    Calls each tool function directly (no LLM) to verify SQL returns real data.
    Fastest way to catch field name / category name mismatches.

  Layer 3 — Bot Tool Routing
    Sends questions to the bot and verifies the LLM called the RIGHT tool.
    12 test cases covering every tool and edge case.

  Layer 4 — Response Quality
    Checks the bot's Arabic responses for: price numbers, stock mentions,
    no hallucinated data, correct language.

  Layer 5 — Sync System
    Verifies daily_sync.py runs without errors and sync_log is written.
```

---

## STEP 1 — CREATE `aiShopzawy/agent/test_bot.py`

```python
"""
Shopzawy Full Test Suite
=========================
5 test layers covering: DB integrity, SQL tools, bot routing,
response quality, and sync system.

Run:
  python aiShopzawy/agent/test_bot.py           → all layers
  python aiShopzawy/agent/test_bot.py --layer 2 → single layer
"""

import os
import sys
import json
import time
import sqlite3
import logging
import argparse
import concurrent.futures
from datetime import datetime
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE     = Path(__file__).parent
_ROOT     = _HERE.parent
DB_PATH   = str(_ROOT / "database" / "shopzawy.db")
SYNC_FILE = str(_ROOT / "database" / "daily_sync.py")
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_HERE))

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.WARNING,      # suppress INFO noise during tests
    format="%(levelname)s  %(message)s",
)

# ── Import bot internals ──────────────────────────────────────────────────────
try:
    from bot import (
        ShopyBot,
        TOOL_FUNCTIONS,
        TOOLS,
        SYSTEM_PROMPT,
        MODEL,
        client,
        DB_PATH as BOT_DB_PATH,
        RAG_AVAILABLE,
    )
    BOT_IMPORTED = True
except ImportError as e:
    print(f"\n❌  FATAL: Cannot import bot.py — {e}")
    print("    Fix the import error in bot.py before running tests.")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# TEST CASES
# Replace every  # ← ADAPT  with real values from Phase 1 output.
# ══════════════════════════════════════════════════════════════════════════════

# fmt: off
TEST_CASES = [
    # Each entry:
    # (test_id, description, question, expected_tool, quality_checks)
    #
    # quality_checks = list of strings that MUST appear in the bot's reply
    # (price numbers, stock words, etc.)

    ("T01", "cheapest overall",
        "ارخص منتج عندكم؟",
        "search_by_price",
        ["جنيه", "متوفر"]),

    ("T02", "cheapest by category",
        "ارخص موبايل عندكم؟",         # ← ADAPT: use a real category keyword
        "search_by_price",
        ["جنيه"]),

    ("T03", "cheapest by brand",
        "ارخص منتج سامسونج عندكم؟",   # ← ADAPT: use a real brand name
        "search_by_price",
        ["جنيه"]),

    ("T04", "most expensive overall",
        "أغلى حاجة عندكم إيه؟",
        "search_by_price",
        ["جنيه"]),

    ("T05", "price range — under",
        "عندكم حاجة تحت 300 جنيه؟",   # ← ADAPT: pick a price with real results
        "filter_by_price_range",
        ["جنيه", "متوفر"]),

    ("T06", "price range — between",
        "عندكم حاجة بين 500 و1500 جنيه؟",  # ← ADAPT: range with real results
        "filter_by_price_range",
        ["جنيه"]),

    ("T07", "price range — above",
        "عندكم حاجة فوق 2000 جنيه؟",  # ← ADAPT: price with real results
        "filter_by_price_range",
        ["جنيه"]),

    ("T08", "best deals",
        "إيه أحسن عروض عندكم دلوقتي؟",
        "get_best_deals",
        ["خصم", "جنيه"]),

    ("T09", "availability — known product",
        "هل عندكم سامسونج؟",           # ← ADAPT: use a product you know exists
        "check_availability",
        []),

    ("T10", "availability — unknown product",
        "هل عندكم منتج مش موجود خالص؟",
        "check_availability",
        []),                            # no quality check — testing graceful failure

    ("T11", "compare two brands",
        "قارنلي بين سامسونج وهواوي",   # ← ADAPT: use two real brands
        "compare_products",
        ["السعر", "المخزون"]),

    ("T12", "semantic — gift suggestion",
        "اقترحلي هدية لبنت عمرها 20 سنة",
        "search_by_meaning",
        []),

    ("T13", "semantic — feature search",
        "محتاج حاجة مناسبة للشغل والسفر",
        "search_by_meaning",
        []),

    ("T14", "compound — cheap + available",
        "عايز موبايل رخيص ومتوفر دلوقتي",  # ← ADAPT: real category keyword
        "search_by_price",
        ["جنيه", "متوفر"]),

    ("T15", "greeting — no tool expected",
        "مرحبا",
        None,                           # None = no tool call expected
        []),

    ("T16", "off-topic redirect",
        "إيه أحسن فيلم شفته؟",
        None,                           # should answer without calling a tool
        []),
]
# fmt: on


# ══════════════════════════════════════════════════════════════════════════════
# TRACKED BOT — captures tool calls, timing, and raw tool results
# ══════════════════════════════════════════════════════════════════════════════

class TrackedBot(ShopyBot):
    """
    Extends ShopyBot to expose tool-call metadata for test assertions.
    Does NOT modify any bot logic — only wraps the LLM calls to record data.
    """

    def chat(self, user_message: str) -> str:
        self._tools_called:  list[str]       = []
        self._tool_args:     dict[str, dict] = {}
        self._tool_results:  dict[str, str]  = {}
        self._round1_tokens: int             = 0
        self._round2_tokens: int             = 0

        self.history.append({"role": "user", "content": user_message})
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *self.history[-8:],
        ]

        # Round 1 — tool selection
        r1 = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.15,
            max_tokens=800,
        )
        self._round1_tokens = r1.usage.total_tokens if r1.usage else 0
        msg = r1.choices[0].message

        # No tool call
        if not msg.tool_calls:
            reply = msg.content or ""
            self.history.append({"role": "assistant", "content": reply})
            return reply

        self._tools_called = [tc.function.name for tc in msg.tool_calls]
        for tc in msg.tool_calls:
            try:
                self._tool_args[tc.function.name] = json.loads(tc.function.arguments or "{}")
            except Exception:
                self._tool_args[tc.function.name] = {}

        messages.append(msg)

        # Execute tools in parallel
        def _run(tc):
            fn   = TOOL_FUNCTIONS.get(tc.function.name)
            args = self._tool_args.get(tc.function.name, {})
            try:
                result = fn(**args) if fn else f"[Tool not found: {tc.function.name}]"
            except Exception as e:
                result = f"[Tool error: {e}]"
            return tc.id, tc.function.name, str(result)

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            for tid, name, result in pool.map(_run, msg.tool_calls):
                self._tool_results[name] = result
                messages.append({
                    "role":         "tool",
                    "tool_call_id": tid,
                    "content":      result,
                })

        # Round 2 — final response
        r2 = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.5,
            max_tokens=700,
        )
        self._round2_tokens = r2.usage.total_tokens if r2.usage else 0
        reply = r2.choices[0].message.content or ""
        self.history.append({"role": "assistant", "content": reply})
        return reply


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 1 — DATABASE INTEGRITY
# ══════════════════════════════════════════════════════════════════════════════

def layer_1_db_integrity() -> dict:
    """
    Verifies the SQLite database is populated correctly before bot tests.
    No LLM calls — pure SQL assertions.
    """
    print("\n" + "═" * 65)
    print("  LAYER 1 — DATABASE INTEGRITY")
    print("═" * 65)

    checks  = []
    passed  = 0
    failed  = 0

    def check(name: str, query: str, assertion, fix: str):
        nonlocal passed, failed
        try:
            conn = sqlite3.connect(DB_PATH)
            result = conn.execute(query).fetchone()
            conn.close()
            value  = result[0] if result else 0
            ok     = assertion(value)
            status = "✅" if ok else "❌"
            checks.append({"name": name, "value": value, "ok": ok, "fix": fix})
            print(f"  {status}  {name:<45} → {value}")
            if ok:
                passed += 1
            else:
                failed += 1
                print(f"      FIX: {fix}")
        except Exception as e:
            failed += 1
            checks.append({"name": name, "value": str(e), "ok": False, "fix": fix})
            print(f"  ❌  {name:<45} → ERROR: {e}")
            print(f"      FIX: {fix}")

    print()

    # ── Core counts ──────────────────────────────────────────────────────────
    check("Total products > 0",
        "SELECT COUNT(*) FROM products",
        lambda v: v > 0,
        "Re-run Phase 1: python aiShopzawy/database/build_sqlite.py")

    check("In-stock products > 0",
        "SELECT COUNT(*) FROM products WHERE in_stock=1",
        lambda v: v > 0,
        "Check extract_stock() in build_sqlite.py — in_stock may always be 0")

    check("Products with sale_price > 0  (> 90%)",
        "SELECT ROUND(COUNT(*)*100.0/(SELECT COUNT(*) FROM products),1) FROM products WHERE sale_price > 0",
        lambda v: float(v or 0) >= 90.0,
        "Check extract_prices() in build_sqlite.py — field names may be wrong")

    check("Products with category name (> 80%)",
        "SELECT ROUND(COUNT(*)*100.0/(SELECT COUNT(*) FROM products),1) FROM products WHERE category_name != ''",
        lambda v: float(v or 0) >= 80.0,
        "Check extract_category() — category may be ID-only or nested wrongly")

    check("Products with brand name (> 70%)",
        "SELECT ROUND(COUNT(*)*100.0/(SELECT COUNT(*) FROM products),1) FROM products WHERE brand_name != ''",
        lambda v: float(v or 0) >= 70.0,
        "Check extract_brand() — brand may be ID-only or nested wrongly")

    check("Products with image_url (> 60%)",
        "SELECT ROUND(COUNT(*)*100.0/(SELECT COUNT(*) FROM products),1) FROM products WHERE image_url != ''",
        lambda v: float(v or 0) >= 60.0,
        "Check extract_image() — image field name may differ from Phase 0")

    check("Categories table populated",
        "SELECT COUNT(*) FROM categories",
        lambda v: v > 0,
        "Re-run build_sqlite.py — rebuild_lookup_tables() may have failed")

    check("Brands table populated",
        "SELECT COUNT(*) FROM brands",
        lambda v: v > 0,
        "Re-run build_sqlite.py — rebuild_lookup_tables() may have failed")

    # ── FTS5 ─────────────────────────────────────────────────────────────────
    check("FTS5 index populated",
        "SELECT COUNT(*) FROM products_fts",
        lambda v: v > 0,
        "FTS triggers failed during build. Re-run build_sqlite.py")

    try:
        conn   = sqlite3.connect(DB_PATH)
        fts_ok = conn.execute(
            "SELECT COUNT(*) FROM products_fts WHERE products_fts MATCH 'منتج'"
        ).fetchone()[0] > 0
        conn.close()
        status = "✅" if fts_ok else "⚠️ "
        print(f"  {status}  {'FTS5 Arabic search works':<45} → {'YES' if fts_ok else 'NO (non-critical)'}")
        if fts_ok:
            passed += 1
    except Exception as e:
        print(f"  ⚠️   FTS5 search test skipped: {e}")

    # ── Variants ─────────────────────────────────────────────────────────────
    check("Products with variants (at least some)",
        "SELECT COUNT(*) FROM products WHERE variants != '[]' AND variants IS NOT NULL",
        lambda v: v >= 0,       # 0 is OK if products have no variants
        "Non-critical — variants may not exist in this store")

    # ── Sync log ──────────────────────────────────────────────────────────────
    check("sync_log has at least 1 entry",
        "SELECT COUNT(*) FROM sync_log",
        lambda v: v > 0,
        "build_sqlite.py should write a sync_log entry — check for errors")

    check("No failed syncs in last 3 runs",
        "SELECT COUNT(*) FROM (SELECT status FROM sync_log ORDER BY id DESC LIMIT 3) WHERE status = 'failed'",
        lambda v: v == 0,
        "Check sync.log for error details — a recent sync failed")

    # ── Discount data ─────────────────────────────────────────────────────────
    try:
        conn = sqlite3.connect(DB_PATH)
        has_disc = conn.execute(
            "SELECT COUNT(*) FROM products WHERE discount_percent > 0"
        ).fetchone()[0]
        conn.close()
        disc_ok = has_disc > 0
        status  = "✅" if disc_ok else "⚠️ "
        print(f"  {status}  {'Products with discount > 0':<45} → {has_disc} (non-critical if 0)")
    except Exception:
        pass

    # ── Price range sanity ────────────────────────────────────────────────────
    try:
        conn   = sqlite3.connect(DB_PATH)
        pmin, pmax, pavg = conn.execute(
            "SELECT MIN(sale_price), MAX(sale_price), ROUND(AVG(sale_price),0) "
            "FROM products WHERE sale_price > 0 AND in_stock = 1"
        ).fetchone()
        conn.close()
        sane = pmin and pmin > 0 and pmax and pmax < 10_000_000
        status = "✅" if sane else "⚠️ "
        print(f"  {status}  {'Price range (in-stock)':<45} → min={pmin}  max={pmax}  avg={pavg}")
    except Exception as e:
        print(f"  ⚠️   Price range check skipped: {e}")

    total = passed + failed
    print(f"\n  Layer 1 result: {passed}/{total} checks passed")
    return {"passed": passed, "failed": failed, "checks": checks}


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 2 — SQL TOOL UNIT TESTS (no LLM)
# ══════════════════════════════════════════════════════════════════════════════

def layer_2_sql_tools() -> dict:
    """
    Calls each tool function directly with concrete arguments.
    Verifies SQL queries return real results — no LLM involved.
    """
    print("\n" + "═" * 65)
    print("  LAYER 2 — SQL TOOL UNIT TESTS  (no LLM)")
    print("═" * 65)

    # Pull real category/brand from DB for realistic test args
    try:
        conn     = sqlite3.connect(DB_PATH)
        real_cat = conn.execute(
            "SELECT category_name FROM products WHERE category_name != '' "
            "GROUP BY category_name ORDER BY COUNT(*) DESC LIMIT 1"
        ).fetchone()
        real_brd = conn.execute(
            "SELECT brand_name FROM products WHERE brand_name != '' "
            "GROUP BY brand_name ORDER BY COUNT(*) DESC LIMIT 1"
        ).fetchone()
        real_prd = conn.execute(
            "SELECT name_ar FROM products WHERE in_stock=1 AND sale_price>0 LIMIT 1"
        ).fetchone()
        mid_price = conn.execute(
            "SELECT ROUND(AVG(sale_price), 0) FROM products WHERE sale_price > 0 AND in_stock=1"
        ).fetchone()
        conn.close()

        cat   = real_cat[0] if real_cat else "موبايل"
        brand = real_brd[0] if real_brd else "Samsung"
        prd   = real_prd[0] if real_prd else "منتج"
        mprice = float(mid_price[0] or 1000) if mid_price else 1000
    except Exception as e:
        print(f"  ⚠️   Could not read real values from DB: {e}")
        cat, brand, prd, mprice = "موبايل", "Samsung", "منتج", 1000

    print(f"\n  Using real values from DB:")
    print(f"    category → {cat}")
    print(f"    brand    → {brand}")
    print(f"    product  → {prd}")
    print(f"    midprice → {mprice}")
    print()

    tool_tests = [
        # (test_name, function_name, kwargs, assertion_fn, fix_hint)
        ("search_by_price  asc",
            "search_by_price",
            {"order": "asc", "limit": 3},
            lambda r: "جنيه" in r and "لا توجد" not in r,
            "sale_price is 0 for all products — check Phase 1 extract_prices()"),

        ("search_by_price  asc + category",
            "search_by_price",
            {"order": "asc", "category": cat, "limit": 3},
            lambda r: len(r) > 10,
            f"No results for category '{cat}' — check KNOWN_CATEGORIES in bot.py"),

        ("search_by_price  asc + brand",
            "search_by_price",
            {"order": "asc", "brand": brand, "limit": 3},
            lambda r: len(r) > 10,
            f"No results for brand '{brand}' — check KNOWN_BRANDS in bot.py"),

        ("search_by_price  desc",
            "search_by_price",
            {"order": "desc", "limit": 3},
            lambda r: "جنيه" in r,
            "No products found — check in_stock flag"),

        ("filter_by_price_range  under midprice",
            "filter_by_price_range",
            {"min_price": 0, "max_price": mprice, "limit": 5},
            lambda r: "جنيه" in r and "لا توجد" not in r,
            f"No products under {mprice} — try a higher max_price"),

        ("filter_by_price_range  above midprice",
            "filter_by_price_range",
            {"min_price": mprice, "max_price": 999999, "limit": 5},
            lambda r: "جنيه" in r,
            f"No products above {mprice} — DB may have very limited price range"),

        ("get_best_deals",
            "get_best_deals",
            {"limit": 5},
            lambda r: len(r) > 20,        # some results (even 0 discount ones)
            "No deals — check discount_percent values in DB; Phase 1 may not compute discount"),

        ("check_availability  existing",
            "check_availability",
            {"product_name": prd[:10]},   # first 10 chars for fuzzy match
            lambda r: len(r) > 20,
            "FTS/LIKE search returning nothing — check products_fts table"),

        ("check_availability  non-existing",
            "check_availability",
            {"product_name": "xyznotexist12345"},
            lambda r: "لم" in r or "مش" in r or "not" in r.lower(),
            "Should return a graceful 'not found' message"),

        ("compare_products",
            "compare_products",
            {"product_names": [cat, brand]},
            lambda r: len(r) > 50,
            "No comparison results — try real product/brand names"),

        ("search_by_meaning  RAG or FTS fallback",
            "search_by_meaning",
            {"query": "هدية", "top_k": 3},
            lambda r: len(r) > 10,
            "RAG unavailable and FTS fallback also empty — check both"),
    ]

    passed = failed = 0
    for name, fn_name, kwargs, assertion, fix in tool_tests:
        fn = TOOL_FUNCTIONS.get(fn_name)
        if not fn:
            print(f"  ❌  {name:<45} → TOOL NOT FOUND")
            failed += 1
            continue
        try:
            t0     = time.time()
            result = fn(**kwargs)
            ms     = round((time.time() - t0) * 1000)
            ok     = assertion(result)
            status = "✅" if ok else "❌"
            print(f"  {status}  {name:<45} ({ms}ms)")
            if not ok:
                print(f"      result preview: {result[:120]}")
                print(f"      FIX: {fix}")
                failed += 1
            else:
                passed += 1
        except Exception as e:
            print(f"  ❌  {name:<45} → EXCEPTION: {e}")
            print(f"      FIX: {fix}")
            failed += 1

    total = passed + failed
    print(f"\n  Layer 2 result: {passed}/{total} checks passed")
    return {"passed": passed, "failed": failed}


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 3 — BOT TOOL ROUTING  (LLM calls — costs tokens)
# ══════════════════════════════════════════════════════════════════════════════

def layer_3_tool_routing() -> dict:
    """
    Sends each test question to the bot and verifies the LLM
    selected the correct tool. This is the most expensive layer.
    """
    print("\n" + "═" * 65)
    print("  LAYER 3 — BOT TOOL ROUTING  (LLM calls)")
    print("═" * 65)

    bot     = TrackedBot()
    results = {"passed": 0, "failed": 0, "no_tool": 0, "wrong_tool": 0}
    timings = []
    failures = []

    for tid, desc, question, expected_tool, _ in TEST_CASES:
        print(f"\n  [{tid}] {desc}")
        print(f"  ❓  {question}")

        t0 = time.time()
        try:
            reply   = bot.chat(question)
            elapsed = round(time.time() - t0, 1)
            tools   = bot._tools_called
            timings.append(elapsed)

            # Evaluate
            if expected_tool is None:
                # Expect NO tool call (greeting, off-topic)
                if not tools:
                    status = "✅ PASS (no tool — expected)"
                    results["passed"] += 1
                else:
                    status = f"⚠️  UNEXPECTED TOOL: {tools}"
                    results["wrong_tool"] += 1
            elif not tools:
                status = f"⚠️  NO TOOL (expected: {expected_tool})"
                results["no_tool"] += 1
                failures.append((tid, desc, expected_tool, "no_tool", tools))
            elif expected_tool in tools:
                status = "✅ PASS"
                results["passed"] += 1
            else:
                status = f"❌ WRONG TOOL: {tools} (expected: {expected_tool})"
                results["wrong_tool"] += 1
                failures.append((tid, desc, expected_tool, "wrong_tool", tools))

            print(f"  🔧  Tools  : {tools or 'none'}")
            print(f"  ⏱   Time   : {elapsed}s   {status}")
            print(f"  🤖  Reply  : {reply[:250]}")

            # Show tool result preview
            for tname, tres in bot._tool_results.items():
                print(f"      [{tname}]: {tres[:120]}...")

        except Exception as e:
            elapsed = round(time.time() - t0, 1)
            print(f"  ❌  EXCEPTION ({elapsed}s): {e}")
            results["failed"] += 1
            failures.append((tid, desc, expected_tool, "exception", str(e)))

        bot.reset()

    # Routing summary
    total     = len(TEST_CASES)
    pass_rate = results["passed"] / total * 100
    avg_time  = round(sum(timings) / max(len(timings), 1), 1)

    print(f"\n  Layer 3 result: {results['passed']}/{total} passed  ({pass_rate:.0f}%)")
    print(f"  Average response time : {avg_time}s")

    if failures:
        print(f"\n  FAILURES TO FIX:")
        for tid, desc, exp, reason, got in failures:
            print(f"    [{tid}] {desc}")
            print(f"           Expected: {exp}  |  Got: {got}  |  Reason: {reason}")
            _print_routing_fix(reason, exp, got)

    return {"passed": results["passed"], "failed": results["failed"] + results["no_tool"] + results["wrong_tool"], "pass_rate": pass_rate}


def _print_routing_fix(reason: str, expected: str, got) -> None:
    """Print a specific fix hint based on the failure type."""
    if reason == "no_tool":
        print(f"           FIX: LLM answered directly without calling any tool.")
        print(f"                Open bot.py → find TOOLS list → strengthen the")
        print(f"                'description' field for '{expected}' to be more")
        print(f"                specific about when to use it.")
    elif reason == "wrong_tool":
        print(f"           FIX: LLM called {got} instead of {expected}.")
        print(f"                The descriptions for both tools overlap too much.")
        print(f"                Make '{expected}' description more specific,")
        print(f"                and add a clear exclusion to {got}'s description.")


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 4 — RESPONSE QUALITY
# ══════════════════════════════════════════════════════════════════════════════

def layer_4_response_quality() -> dict:
    """
    Focuses on WHAT the bot says, not which tool it calls.
    Checks for: price numbers, stock mentions, no hallucination signals,
    Arabic language, no empty responses.
    """
    print("\n" + "═" * 65)
    print("  LAYER 4 — RESPONSE QUALITY")
    print("═" * 65)

    bot     = TrackedBot()
    passed  = failed = 0
    quality_issues = []

    for tid, desc, question, expected_tool, quality_checks in TEST_CASES:
        if not quality_checks:
            continue     # skip tests with no quality assertions

        try:
            reply = bot.chat(question)
            issues = []

            # Check all required strings appear in the reply
            for required in quality_checks:
                if required not in reply:
                    issues.append(f"missing '{required}'")

            # Universal quality rules
            if len(reply.strip()) < 10:
                issues.append("reply too short (< 10 chars)")
            if "I cannot" in reply or "I don't know" in reply:
                issues.append("English refusal in Arabic context")

            # Price hallucination check — if tool returned results,
            # reply must contain at least one digit
            if expected_tool in ("search_by_price", "filter_by_price_range", "get_best_deals"):
                if not any(c.isdigit() for c in reply):
                    issues.append("no price numbers in reply (possible hallucination)")

            if issues:
                print(f"\n  ❌  [{tid}] {desc}")
                print(f"       Question : {question}")
                print(f"       Issues   : {', '.join(issues)}")
                print(f"       Reply    : {reply[:200]}")
                failed += 1
                quality_issues.append((tid, issues))
            else:
                print(f"  ✅  [{tid}] {desc}")
                passed += 1

        except Exception as e:
            print(f"  ❌  [{tid}] {desc}  — EXCEPTION: {e}")
            failed += 1

        bot.reset()

    total = passed + failed
    print(f"\n  Layer 4 result: {passed}/{total} quality checks passed")
    return {"passed": passed, "failed": failed, "issues": quality_issues}


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 5 — SYNC SYSTEM
# ══════════════════════════════════════════════════════════════════════════════

def layer_5_sync_system() -> dict:
    """
    Tests daily_sync.py: import, PG connection, SQLite write, sync_log.
    Does NOT require PostgreSQL to be live — catches import errors at minimum.
    """
    print("\n" + "═" * 65)
    print("  LAYER 5 — SYNC SYSTEM")
    print("═" * 65)

    passed = failed = 0

    # ── 5.1: Can import daily_sync ────────────────────────────────────────────
    print("\n  [5.1] Import daily_sync.py")
    try:
        sys.path.insert(0, str(Path(SYNC_FILE).parent))
        import daily_sync
        print("  ✅  Import OK")
        passed += 1
    except ImportError as e:
        print(f"  ❌  Import failed: {e}")
        print(f"      FIX: Check daily_sync.py for syntax errors or missing deps")
        failed += 1
        return {"passed": passed, "failed": failed + 3}

    # ── 5.2: run_sync callable ────────────────────────────────────────────────
    print("\n  [5.2] run_sync function exists")
    if callable(getattr(daily_sync, "run_sync", None)):
        print("  ✅  run_sync is callable")
        passed += 1
    else:
        print("  ❌  run_sync not found in daily_sync.py")
        failed += 1

    # ── 5.3: PostgreSQL connection test ───────────────────────────────────────
    print("\n  [5.3] PostgreSQL connection")
    try:
        pg = daily_sync._get_pg()
        pg.close()
        print("  ✅  PostgreSQL connected OK")
        passed += 1

        # ── 5.4: Fetch changed products ───────────────────────────────────────
        print("\n  [5.4] Fetch changed products (last 25h)")
        try:
            from datetime import datetime, timedelta
            since = datetime.now() - timedelta(hours=25)
            products = daily_sync.fetch_changed_products(since)
            print(f"  ✅  Fetched {len(products)} changed product(s)")
            passed += 1

            # ── 5.5: Update SQLite with fetched products ──────────────────────
            if products:
                print(f"\n  [5.5] Update SQLite with {len(products)} product(s)")
                try:
                    stats = daily_sync.update_sqlite(products)
                    print(f"  ✅  SQLite updated: added={stats['added']}  updated={stats['updated']}  errors={stats['errors']}")
                    if stats["errors"] > 0:
                        print(f"      ⚠️  {stats['errors']} errors — check sync.log")
                    passed += 1
                except Exception as e:
                    print(f"  ❌  update_sqlite failed: {e}")
                    failed += 1
            else:
                print("\n  [5.5] No changed products — SQLite update skipped (OK)")
                passed += 1

        except Exception as e:
            print(f"  ❌  fetch_changed_products failed: {e}")
            print(f"      FIX: Adapt the SQL query in daily_sync.py to match your schema")
            failed += 2

    except Exception as e:
        print(f"  ⚠️   PostgreSQL connection failed: {e}")
        print(f"      FIX: Check DB_HOST / DB_PORT / DB_NAME / DB_USER / DB_PASSWORD in .env")
        print(f"      NOTE: Bot still works — it reads from SQLite only")
        failed += 3  # count PG + fetch + update as failed

    # ── 5.6: sync_log was written ─────────────────────────────────────────────
    print("\n  [5.6] sync_log table check")
    try:
        conn  = sqlite3.connect(DB_PATH)
        logs  = conn.execute(
            "SELECT run_at, sync_type, status FROM sync_log ORDER BY id DESC LIMIT 3"
        ).fetchall()
        conn.close()
        if logs:
            print(f"  ✅  sync_log has {len(logs)} recent entries:")
            for row in logs:
                print(f"       {row[0]}  {row[1]}  {row[2]}")
            passed += 1
        else:
            print("  ❌  sync_log is empty")
            failed += 1
    except Exception as e:
        print(f"  ❌  sync_log check failed: {e}")
        failed += 1

    # ── 5.7: scheduler.py exists ─────────────────────────────────────────────
    print("\n  [5.7] scheduler.py exists")
    sched_path = Path(SYNC_FILE).parent / "scheduler.py"
    if sched_path.exists():
        print(f"  ✅  {sched_path}")
        passed += 1
    else:
        print(f"  ❌  scheduler.py not found at {sched_path}")
        print(f"      FIX: Create it from Phase 3")
        failed += 1

    total = passed + failed
    print(f"\n  Layer 5 result: {passed}/{total} checks passed")
    return {"passed": passed, "failed": failed}


# ══════════════════════════════════════════════════════════════════════════════
# FINAL REPORT
# ══════════════════════════════════════════════════════════════════════════════

def print_final_report(layer_results: dict) -> bool:
    """Print overall pass/fail summary and production readiness verdict."""
    print("\n" + "═" * 65)
    print("  FULL TEST SUITE — FINAL REPORT")
    print(f"  {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("═" * 65)

    total_passed = total_failed = 0
    for layer, res in layer_results.items():
        p = res.get("passed", 0)
        f = res.get("failed", 0)
        total_passed += p
        total_failed += f
        bar    = "✅" if f == 0 else ("⚠️ " if p > f else "❌")
        pct    = p / max(p + f, 1) * 100
        print(f"  {bar}  {layer:<35} {p}/{p+f}  ({pct:.0f}%)")

    total    = total_passed + total_failed
    pass_pct = total_passed / max(total, 1) * 100
    l3_pct   = layer_results.get("Layer 3 — Tool Routing", {}).get("pass_rate", 0)

    print(f"\n  OVERALL: {total_passed}/{total} passed  ({pass_pct:.0f}%)")
    print()

    # ── Production readiness ──────────────────────────────────────────────────
    l1_ok = layer_results.get("Layer 1 — DB Integrity", {}).get("failed", 1) == 0
    l2_ok = layer_results.get("Layer 2 — SQL Tools",    {}).get("failed", 1) == 0
    l3_ok = l3_pct >= 80
    l4_ok = layer_results.get("Layer 4 — Response Quality", {}).get("failed", 1) == 0
    l5_ok = layer_results.get("Layer 5 — Sync System",  {}).get("failed", 1) <= 2

    print("  PRODUCTION READINESS CHECKLIST:")
    print(f"  {'✅' if l1_ok else '❌'}  Database integrity        (required)")
    print(f"  {'✅' if l2_ok else '❌'}  SQL tools return results  (required)")
    print(f"  {'✅' if l3_ok else '⚠️ '}  Bot routing ≥ 80%        (required — currently {l3_pct:.0f}%)")
    print(f"  {'✅' if l4_ok else '⚠️ '}  Response quality          (recommended)")
    print(f"  {'✅' if l5_ok else '⚠️ '}  Sync system               (recommended)")

    ready = l1_ok and l2_ok and l3_ok
    print()
    if ready:
        print("  🎉  BOT IS PRODUCTION READY")
        print()
        print("  Next steps:")
        print("    1. pm2 start aiShopzawy/database/ecosystem.config.js")
        print("    2. pm2 startup && pm2 save  (auto-start on reboot)")
        print("    3. pm2 logs shopzawy-sync   (monitor sync logs)")
    elif l1_ok and l2_ok:
        print("  ⚠️   BOT WORKS BUT NEEDS TUNING (routing < 80%)")
        print()
        print("  Fix routing by strengthening tool descriptions in bot.py TOOLS list.")
        print("  Re-run this test after each fix.")
    else:
        print("  🔴  BOT IS NOT READY — FIX LAYERS 1 AND 2 FIRST")
        print()
        print("  Layer 1 failures = data problem (re-run Phase 1)")
        print("  Layer 2 failures = SQL problem  (fix field names in bot.py)")

    print("═" * 65)
    return ready


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Shopzawy full test suite")
    parser.add_argument(
        "--layer", type=int, choices=[1, 2, 3, 4, 5],
        help="Run only a specific layer (1–5). Default: all layers."
    )
    parser.add_argument(
        "--skip-llm", action="store_true",
        help="Skip layers 3 and 4 (no LLM calls — faster, saves tokens)"
    )
    args = parser.parse_args()

    print("\n" + "═" * 65)
    print("  SHOPZAWY FULL TEST SUITE")
    print(f"  {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"  DB      : {DB_PATH}")
    print(f"  RAG     : {'available' if RAG_AVAILABLE else 'unavailable (FTS fallback)'}")
    print(f"  Model   : {MODEL}")
    print("═" * 65)

    layer_results = {}

    run_all   = args.layer is None
    skip_llm  = args.skip_llm

    if run_all or args.layer == 1:
        layer_results["Layer 1 — DB Integrity"]    = layer_1_db_integrity()

    if run_all or args.layer == 2:
        layer_results["Layer 2 — SQL Tools"]       = layer_2_sql_tools()

    if (run_all or args.layer == 3) and not skip_llm:
        layer_results["Layer 3 — Tool Routing"]    = layer_3_tool_routing()

    if (run_all or args.layer == 4) and not skip_llm:
        layer_results["Layer 4 — Response Quality"] = layer_4_response_quality()

    if run_all or args.layer == 5:
        layer_results["Layer 5 — Sync System"]     = layer_5_sync_system()

    if len(layer_results) > 1:
        ready = print_final_report(layer_results)
        sys.exit(0 if ready else 1)


if __name__ == "__main__":
    main()
```

---

## STEP 2 — RUN ALL LAYERS

```bash
cd [monorepo root]

# Full run (all 5 layers)
python aiShopzawy/agent/test_bot.py

# Fast run — skip LLM calls (layers 1, 2, 5 only)
python aiShopzawy/agent/test_bot.py --skip-llm

# Single layer
python aiShopzawy/agent/test_bot.py --layer 1   # DB only
python aiShopzawy/agent/test_bot.py --layer 2   # SQL tools only
python aiShopzawy/agent/test_bot.py --layer 3   # routing only
```

---

## STEP 3 — FIX BY FAILURE TYPE

### Layer 1 failures — database problem

| Failure | Root cause | Fix |
|---------|-----------|-----|
| `Total products = 0` | build_sqlite.py crashed | Re-run Phase 1 |
| `sale_price > 0 < 90%` | Wrong price field name | Fix `extract_prices()` in build_sqlite.py |
| `category_name < 80%` | Nested category not parsed | Fix `extract_category()` — check if it's `category.name` |
| `FTS5 index = 0` | Triggers not fired | Re-run build_sqlite.py; check for FTS trigger errors |
| `sync_log empty` | build_sqlite.py failed silently | Check build.log for errors |

### Layer 2 failures — SQL tool problem

| Failure | Root cause | Fix |
|---------|-----------|-----|
| `search_by_price` empty | `in_stock` always 0 | Fix `extract_stock()` in Phase 1 |
| `search_by_price + category` empty | Category name mismatch | Update `KNOWN_CATEGORIES` in bot.py with real names |
| `get_best_deals` empty | `discount_percent` always 0 | Fix `extract_prices()` discount calculation |
| `check_availability` no graceful msg | Missing empty-result handler | Check `fn_check_availability()` return when rows = [] |

### Layer 3 failures — LLM routing problem

| Failure | Root cause | Fix |
|---------|-----------|-----|
| `NO TOOL` for price questions | Tool description too vague | Strengthen `search_by_price` description with Arabic trigger words |
| `wrong_tool` between price + range | Descriptions overlap | Add exclusion clause: "do NOT use for 'under X' or 'between X and Y'" |
| `NO TOOL` for greetings | LLM over-triggers tools | Lower temperature to 0.1 or add explicit greeting handler |

### Layer 4 failures — response quality problem

| Failure | Root cause | Fix |
|---------|-----------|-----|
| No price numbers in reply | Tool returned 0 prices | Fix Layer 2 first |
| Missing `متوفر` | No stock info in tool output | Check `_rows_to_str()` stock_text logic |
| English refusal | LLM ignoring system prompt | Strengthen Arabic instruction in SYSTEM_PROMPT |

### Layer 5 failures — sync problem

| Failure | Root cause | Fix |
|---------|-----------|-----|
| PG connection refused | Wrong .env values | Check DB_HOST / DB_USER / DB_PASSWORD |
| SQL fetch fails | Wrong column names in query | Re-read migrations; adapt `fetch_changed_products()` SQL |
| `sync_log` empty | transaction failed silently | Check sync.log file for Python tracebacks |

---

## STEP 4 — FINAL CONFIRMATION

After all layers pass (or only non-critical failures remain), write exactly:

```
Phase 4 complete ✅

  Layer 1 — DB Integrity         : X/Y passed
  Layer 2 — SQL Tools            : X/Y passed
  Layer 3 — Tool Routing         : X/16  (XX%)
  Layer 4 — Response Quality     : X/Y passed
  Layer 5 — Sync System          : X/Y passed

  RAG available  : YES / NO
  PG connected   : YES / NO
  PM2 running    : YES / NO

  BOT STATUS: PRODUCTION READY ✅  /  NEEDS TUNING ⚠️  /  NOT READY 🔴

  pm2 start command:
    pm2 start aiShopzawy/database/ecosystem.config.js
    pm2 startup && pm2 save
```
