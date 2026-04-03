"""
TriageAI Deep Test Script
Run: python test_triageai.py
"""

import requests
import json
import time

BASE_URL = "https://ritanshupatel-triageai-backend-1.hf.space"

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

passed = []
failed = []

def ok(msg):
    passed.append(msg)
    print(f"  {GREEN}✅ PASS{RESET} — {msg}")

def fail(msg, detail=""):
    failed.append(msg)
    print(f"  {RED}❌ FAIL{RESET} — {msg}")
    if detail:
        print(f"         {YELLOW}   {detail}{RESET}")

def warn(msg):
    print(f"  {YELLOW}⚠️  WARN{RESET} — {msg}")

def section(title):
    print(f"\n{BOLD}{BLUE}{'='*60}{RESET}")
    print(f"{BOLD}{BLUE}  {title}{RESET}")
    print(f"{BOLD}{BLUE}{'='*60}{RESET}")

def analyze(text):
    resp = requests.post(f"{BASE_URL}/analyze", json={"text": text}, timeout=90)
    resp.raise_for_status()
    return resp.json()

# ── TEST NOTES ─────────────────────────────────────────────────────────
NOTE_STEMI = (
    "pt 67M c/o CP x2hr rad to L arm, diaphoretic, SOB+, "
    "PMH DM2 HTN, smoker 20pk/yr, meds metformin 500 BID "
    "lisinopril 10 QD atorvastatin 40 QHS, EKG ST elev V2-V4, "
    "trop pnd, A: r/o STEMI, P: ASA 325 STAT ntg SL heparin gtt "
    "cath lab activation, BP 158/94 HR 102 RR 18 O2 94% RA, allerg PCN hives"
)

NOTE_DIABETES = (
    "pt 54F DM2 f/u, HbA1c 8.2, c/o fatigue wt gain, "
    "curr meds metformin 1000 BID glipizide 5 QD, BP 138/86 HR 74, "
    "A: DM2 uncontrolled, P: inc metformin to 2000 BID"
)

NOTE_CHF = (
    "pt 82M nursing home transfer, PMH CHF HTN AFib CKD stage 3, "
    "meds furosemide 40 QD metoprolol 25 BID warfarin 5 QD lisinopril 5 QD, "
    "c/o worsening SOB x5d, LE edema ++, O2 92% RA, BP 162/98 HR 88 irreg"
)

NOTE_MINIMAL = "pt c/o headache, no meds, no PMH"

NOTE_THYROID = (
    "pt 34F f/u thyroid, tired, wt gain 8lbs/3mo, TSH 8.2 high, "
    "curr meds none, HR 58 BP 112/70, A: hypothyroidism, "
    "P: start levothyroxine 50mcg QAM"
)

# ═══════════════════════════════════════════════════════════════════════════
section("TEST 1: Health Check")
# ═══════════════════════════════════════════════════════════════════════════
try:
    r = requests.get(f"{BASE_URL}/health", timeout=15)
    if r.status_code == 200:
        ok(f"Backend live — {r.json()}")
    else:
        fail("Non-200 response", str(r.status_code))
except Exception as e:
    fail("Backend unreachable", str(e))

# ═══════════════════════════════════════════════════════════════════════════
section("TEST 2: STEMI Note — Full Pipeline (Bug #1,2,3,4,6,8,9,10)")
# ═══════════════════════════════════════════════════════════════════════════
try:
    print(f"  Sending STEMI note... (may take 20-30s)")
    result = analyze(NOTE_STEMI)
    report = result.get("report", {})
    pc = report.get("patient_card", {})

    # Bug #2: age should be 67 not null
    age = pc.get("age")
    if age == 67:
        ok("Bug #2 fixed — age=67 (not null)")
    elif age is not None:
        warn(f"Bug #2 partial — age={age} (expected 67)")
    else:
        fail("Bug #2 NOT fixed — age is null", f"patient_card={pc}")

    # Bug #2: gender should be male not null
    gender = pc.get("gender")
    if gender and "male" in str(gender).lower():
        ok("Bug #2 fixed — gender=male (not null)")
    else:
        fail("Bug #2 NOT fixed — gender is null or wrong", f"gender={gender}")

    # Bug #1 + #10: conditions_with_codes should have ICD codes
    cwc = pc.get("conditions_with_codes", [])
    if cwc:
        codes_filled = [c for c in cwc if c.get("icd_code") and c["icd_code"] not in ("", "Pending", "Not found")]
        if codes_filled:
            ok(f"Bug #1+#10 fixed — {len(codes_filled)}/{len(cwc)} conditions have ICD codes")
            for c in cwc:
                print(f"         → {c.get('condition')} : {c.get('icd_code')}")
        else:
            fail("Bug #1+#10 NOT fixed — all ICD codes empty", str(cwc[:2]))
    else:
        fail("Bug #1+#10 NOT fixed — conditions_with_codes is empty")

    # Bug #8: vitals should not be empty
    vitals = pc.get("vitals", {})
    vitals_filled = {k: v for k, v in vitals.items() if v}
    if vitals_filled:
        ok(f"Bug #8 fixed — vitals present: {vitals_filled}")
    else:
        fail("Bug #8 NOT fixed — vitals empty", str(vitals))

    # Bug #9: medications should not be empty
    meds = pc.get("medications", [])
    if meds:
        ok(f"Bug #9 fixed — {len(meds)} medications found")
        for m in meds:
            print(f"         → {m.get('name')} {m.get('dose')} {m.get('frequency')}")
    else:
        fail("Bug #9 NOT fixed — medications empty")

    # Bug #3 + #4: drug warnings should not be empty (metformin, lisinopril, atorvastatin, ASA, ntg, heparin)
    warnings = report.get("drug_warnings", [])
    if warnings:
        ok(f"Bug #3+#4 fixed — {len(warnings)} drug warnings found")
        for w in warnings:
            print(f"         → {w.get('drug')} [{w.get('severity')}]")
    else:
        fail("Bug #3+#4 NOT fixed — drug_warnings is empty")

    # Bug #5: plain_english_summary should not be null or raw medical text
    summary = report.get("plain_english_summary", "")
    if summary and len(summary) > 20 and "pt " not in summary[:10]:
        ok(f"Bug #5 fixed — summary is readable")
        print(f"         → {summary[:120]}...")
    elif summary:
        warn(f"Bug #5 partial — summary exists but may be raw text: {summary[:80]}")
    else:
        fail("Bug #5 NOT fixed — summary is empty/null")

    # Allergy check
    allergies = pc.get("allergies", [])
    if allergies and isinstance(allergies[0], str):
        ok(f"Allergies normalized — {allergies}")
    elif allergies:
        warn(f"Allergies not plain strings — {allergies}")
    else:
        warn("No allergies found (expected PCN)")

    # Errors check
    errors = report.get("errors", [])
    if not errors:
        ok("No errors in pipeline")
    else:
        warn(f"{len(errors)} pipeline errors: {errors}")

except Exception as e:
    fail("STEMI test crashed", str(e))
    import traceback; traceback.print_exc()

# ═══════════════════════════════════════════════════════════════════════════
section("TEST 3: Diabetes Note — Age/Gender/Medications")
# ═══════════════════════════════════════════════════════════════════════════
try:
    print(f"  Sending diabetes note...")
    result = analyze(NOTE_DIABETES)
    report = result.get("report", {})
    pc = report.get("patient_card", {})

    age = pc.get("age")
    gender = pc.get("gender")
    meds = pc.get("medications", [])
    cwc = pc.get("conditions_with_codes", [])

    if age == 54:
        ok("Age=54 correct")
    else:
        fail(f"Age wrong — got {age}, expected 54")

    if gender and "female" in str(gender).lower():
        ok("Gender=female correct")
    else:
        fail(f"Gender wrong — got {gender}")

    if len(meds) >= 2:
        ok(f"{len(meds)} medications found (metformin + glipizide)")
    else:
        fail(f"Medications incomplete — only {len(meds)} found")

    if cwc:
        ok(f"{len(cwc)} conditions with codes")
    else:
        fail("No conditions_with_codes")

except Exception as e:
    fail("Diabetes test crashed", str(e))

# ═══════════════════════════════════════════════════════════════════════════
section("TEST 4: CHF Note — Multiple Drugs + Drug Warnings")
# ═══════════════════════════════════════════════════════════════════════════
try:
    print(f"  Sending CHF note (furosemide + warfarin + metoprolol + lisinopril)...")
    result = analyze(NOTE_CHF)
    report = result.get("report", {})
    pc = report.get("patient_card", {})
    warnings = report.get("drug_warnings", [])
    meds = pc.get("medications", [])

    if len(meds) >= 3:
        ok(f"{len(meds)} medications found")
    else:
        fail(f"Only {len(meds)} medications — expected 4+")

    if warnings:
        ok(f"{len(warnings)} drug warnings")
        for w in warnings:
            sev = w.get('severity', '?')
            desc = (w.get('interactions') or w.get('warning') or '')[:60]
            print(f"         → {w.get('drug')} [{sev}] — {desc}...")
    else:
        fail("No drug warnings for warfarin/furosemide/lisinopril combo")

    age = pc.get("age")
    if age == 82:
        ok("Age=82 correct")
    else:
        fail(f"Age wrong — got {age}")

except Exception as e:
    fail("CHF test crashed", str(e))

# ═══════════════════════════════════════════════════════════════════════════
section("TEST 5: Minimal Note — No Crash on Empty Data")
# ═══════════════════════════════════════════════════════════════════════════
try:
    print(f"  Sending minimal note...")
    result = analyze(NOTE_MINIMAL)
    report = result.get("report", {})
    pc = report.get("patient_card", {})

    # Should not crash — just return empty/null fields gracefully
    ok("Minimal note did not crash")

    summary = report.get("plain_english_summary", "")
    if summary:
        ok(f"Summary generated even for minimal note")
        print(f"         → {summary[:80]}")
    else:
        fail("No summary for minimal note")

    # Medications should be [] not crash
    meds = pc.get("medications", [])
    if isinstance(meds, list):
        ok(f"Medications is list (not crash) — {len(meds)} items")
    else:
        fail("Medications field wrong type", str(type(meds)))

except Exception as e:
    fail("Minimal note crashed the pipeline", str(e))

# ═══════════════════════════════════════════════════════════════════════════
section("TEST 6: Thyroid Note — ICD Code for Hypothyroidism")
# ═══════════════════════════════════════════════════════════════════════════
try:
    print(f"  Sending thyroid note...")
    result = analyze(NOTE_THYROID)
    report = result.get("report", {})
    pc = report.get("patient_card", {})
    cwc = pc.get("conditions_with_codes", [])

    if cwc:
        hypo = next((c for c in cwc if "hypo" in c.get("condition","").lower() or "thyroid" in c.get("condition","").lower()), None)
        if hypo and hypo.get("icd_code"):
            ok(f"Hypothyroidism ICD code: {hypo['icd_code']} (expected E03.9)")
        elif hypo:
            fail("Hypothyroidism found but no ICD code", str(hypo))
        else:
            warn("Hypothyroidism not in conditions_with_codes")
    else:
        fail("No conditions_with_codes in thyroid note")

    meds = pc.get("medications", [])
    levo = next((m for m in meds if "levothyroxine" in str(m.get("name","")).lower() or "levo" in str(m.get("name","")).lower()), None)
    if levo:
        ok(f"Levothyroxine found — {levo}")
    else:
        warn(f"Levothyroxine not found in meds: {meds}")

except Exception as e:
    fail("Thyroid test crashed", str(e))

# ═══════════════════════════════════════════════════════════════════════════
section("TEST 7: Fine-tuned Model Endpoint")
time.sleep(10)  # wait for Groq rate limit to reset

# ═══════════════════════════════════════════════════════════════════════════
try:
    print(f"  Testing /analyze/finetuned...")
    r = requests.post(f"{BASE_URL}/analyze/finetuned", json={"text": NOTE_DIABETES}, timeout=30)
    if r.status_code == 200:
        data = r.json()
        if data.get("result"):
            ok("Fine-tuned endpoint returns result")
            print(f"         → model: {data.get('model', 'unknown')}")
        else:
            fail("Fine-tuned endpoint returned empty result", str(data))
    else:
        fail(f"Fine-tuned endpoint status {r.status_code}", r.text[:100])
except Exception as e:
    fail("Fine-tuned endpoint crashed", str(e))

# ═══════════════════════════════════════════════════════════════════════════
section("FINAL RESULTS")
# ═══════════════════════════════════════════════════════════════════════════
total = len(passed) + len(failed)
print(f"\n  {GREEN}Passed: {len(passed)}/{total}{RESET}")
print(f"  {RED}Failed: {len(failed)}/{total}{RESET}")

if failed:
    print(f"\n  {RED}{BOLD}Failed tests:{RESET}")
    for f in failed:
        print(f"  {RED}  • {f}{RESET}")

if not failed:
    print(f"\n  {GREEN}{BOLD}🎉 All tests passed! TriageAI is working correctly.{RESET}")
else:
    print(f"\n  {YELLOW}Fix the failed tests above, then re-run this script.{RESET}")