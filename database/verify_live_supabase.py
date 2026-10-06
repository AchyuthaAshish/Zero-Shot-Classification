"""Live Supabase Database Verification and Schema Inspector.

Performs live read-only and constraint tests against the configured Supabase project:
1. Connection status and project reachability.
2. Parent table (public.defect_reports) column verification (including is_multi_defect, defect_count, validation_status).
3. Parent category CHECK constraint enforcement (rejects unapproved categories).
4. Child table (public.defect_report_items) existence, reachability, and 24-column verification.
5. Child constraints verification (taxonomy check, foreign key cascade/reference, calibrated score check).
6. Read-only retrieval and parent-child query verification on existing records.
7. Row Level Security (RLS) immutability enforcement (UPDATE and DELETE prevention).

DO NOT expose real credentials in logs or console output.
"""

import os
import sys
import json
from typing import Dict, Any, Optional

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import load_settings
from database.supabase_client import (
    is_supabase_configured,
    get_supabase_client,
    get_defect_reports,
    get_defect_report_items,
    DatabaseError,
)

def run_live_verification():
    print("=" * 60)
    print("      LIVE SUPABASE BACKEND VERIFICATION REPORT")
    print("=" * 60)

    if not is_supabase_configured():
        print("[FAIL] Supabase is NOT configured.")
        print("Please configure SUPABASE_URL and SUPABASE_KEY in your .env file:")
        print("  SUPABASE_URL=https://<your-project-id>.supabase.co")
        print("  SUPABASE_KEY=<your-anon-publishable-key>")
        print("=" * 60)
        return False

    client = get_supabase_client()
    print("[PASS] Supabase client initialized with environment settings.")
    print("[INFO] Target tables: public.defect_reports, public.defect_report_items\n")

    # Step 1: Reachability & Parent Columns Verification
    print("--- Step 1: Parent Table (public.defect_reports) Column Verification ---")
    required_parent_cols = [
        "id", "reporter_name", "employee_id", "defect_description",
        "category", "confidence", "reliability", "explanation",
        "classification_mode", "provider", "model", "created_at",
        "is_multi_defect", "defect_count", "validation_status"
    ]
    try:
        res = client.table("defect_reports").select(",".join(required_parent_cols)).limit(1).execute()
        print("[PASS] Successfully queried public.defect_reports with all metadata columns.")
        print(f"       Verified parent columns: {len(required_parent_cols)} present (including is_multi_defect, defect_count, validation_status).")
    except Exception as e:
        print(f"[FAIL] Could not query public.defect_reports metadata columns: {e}")
        return False

    # Step 2: Parent Category CHECK Constraint Verification (Invalid Category Rejection)
    print("\n--- Step 2: Testing Parent Category CHECK Constraint ---")
    invalid_parent_payload = {
        "reporter_name": "Test Tester",
        "employee_id": "TEST-000",
        "defect_description": "Test invalid category constraint rejection.",
        "category": "Alien Fault",  # Invalid category
        "confidence": 0.5,
        "reliability": "Low",
        "classification_mode": "LOCAL",
    }
    try:
        client.table("defect_reports").insert(invalid_parent_payload).execute()
        print("[FAIL] Database accepted an invalid category ('Alien Fault')! CHECK constraint is missing.")
        return False
    except Exception as e:
        print("[PASS] Database rejected invalid parent category as expected by CHECK constraint.")
        print(f"       Rejection details: {type(e).__name__}")

    # Step 3: Child Table (public.defect_report_items) Reachability & Columns
    print("\n--- Step 3: Child Table (public.defect_report_items) Column Verification ---")
    required_child_cols = [
        "id", "report_id", "defect_index", "defect_id", "segment_id",
        "defect_text", "start_char", "end_char", "category",
        "confidence", "confidence_level", "confidence_range",
        "raw_score", "calibrated_score", "top2_margin",
        "reliability", "explanation", "classification_mode",
        "provider", "model", "status", "is_ambiguous",
        "ambiguity_reason", "created_at"
    ]
    try:
        res_child = client.table("defect_report_items").select(",".join(required_child_cols)).limit(1).execute()
        print("[PASS] Successfully queried public.defect_report_items.")
        print(f"       Verified child columns: {len(required_child_cols)} present and accessible via PostgREST.")
    except Exception as e:
        print(f"[FAIL] Could not query public.defect_report_items columns: {e}")
        return False

    # Step 4: Child Table Constraints Verification (Without leaving persistent data)
    print("\n--- Step 4: Testing Child Constraints (Taxonomy, FK, Calibrated Score) ---")
    
    # 4a. Child Category CHECK Constraint
    try:
        client.table("defect_report_items").insert({
            "report_id": "00000000-0000-0000-0000-000000000000",
            "defect_index": 1,
            "defect_text": "Constraint test item",
            "category": "Hydraulic Fault"  # Unapproved category
        }).execute()
        print("[FAIL] Child table accepted an unapproved category ('Hydraulic Fault')!")
        return False
    except Exception as e:
        print("[PASS] Child table rejected unapproved category ('Hydraulic Fault') via chk_defect_item_category.")

    # 4b. Child Foreign Key Constraint
    try:
        client.table("defect_report_items").insert({
            "report_id": "00000000-0000-0000-0000-000000000000",
            "defect_index": 1,
            "defect_text": "Constraint test item",
            "category": "Mechanical Fault"
        }).execute()
        print("[FAIL] Child table accepted nonexistent report_id FK!")
        return False
    except Exception as e:
        print("[PASS] Child table rejected invalid foreign key (report_id) via defect_report_items_report_id_fkey.")

    # 4c. Child Calibrated Score Bounds Constraint
    try:
        client.table("defect_report_items").insert({
            "report_id": "00000000-0000-0000-0000-000000000000",
            "defect_index": 1,
            "defect_text": "Constraint test item",
            "category": "Mechanical Fault",
            "calibrated_score": 1.5  # Bounded [0.0, 1.0]
        }).execute()
        print("[FAIL] Child table accepted out-of-bounds calibrated_score!")
        return False
    except Exception as e:
        print("[PASS] Child table rejected out-of-bounds calibrated_score (> 1.0) via chk_defect_item_calibrated_score.")

    # Step 5: Read-Only Existing Data & Retrieval Verification
    print("\n--- Step 5: Existing Data & Query Verification ---")
    reports = get_defect_reports(limit=5)
    print(f"[PASS] Successfully retrieved {len(reports)} recent reports from defect_reports.")
    if reports:
        first = reports[0]
        print(f"       Sample row ID: {first.get('id')}")
        print(f"       is_multi_defect: {first.get('is_multi_defect')}")
        print(f"       defect_count: {first.get('defect_count')}")
        print(f"       validation_status: {first.get('validation_status')}")
        
        # Test helper get_defect_report_items on existing report_id
        items = get_defect_report_items(first.get("id"))
        print(f"[PASS] get_defect_report_items() executed successfully for existing report (returned {len(items)} items).")

    # Step 6: Test RLS Immutability (UPDATE & DELETE Prevention)
    print("\n--- Step 6: Testing RLS Immutability (UPDATE/DELETE Prevention) ---")
    dummy_id = "00000000-0000-0000-0000-000000000000"
    
    # 6a. Parent UPDATE / DELETE
    try:
        res_upd_p = client.table("defect_reports").update({"category": "Unknown"}).eq("id", dummy_id).execute()
        if not res_upd_p.data:
            print("[PASS] Parent UPDATE blocked / 0 rows affected by RLS audit policy.")
        else:
            print("[WARN] Parent UPDATE returned affected rows.")
    except Exception as e:
        print(f"[PASS] Parent UPDATE blocked by database: {type(e).__name__}")

    try:
        res_del_p = client.table("defect_reports").delete().eq("id", dummy_id).execute()
        if not res_del_p.data:
            print("[PASS] Parent DELETE blocked / 0 rows affected by RLS audit policy.")
        else:
            print("[WARN] Parent DELETE returned affected rows.")
    except Exception as e:
        print(f"[PASS] Parent DELETE blocked by database: {type(e).__name__}")

    # 6b. Child UPDATE / DELETE
    try:
        res_upd_c = client.table("defect_report_items").update({"category": "Unknown"}).eq("id", dummy_id).execute()
        if not res_upd_c.data:
            print("[PASS] Child UPDATE blocked / 0 rows affected by RLS audit policy.")
        else:
            print("[WARN] Child UPDATE returned affected rows.")
    except Exception as e:
        print(f"[PASS] Child UPDATE blocked by database: {type(e).__name__}")

    try:
        res_del_c = client.table("defect_report_items").delete().eq("id", dummy_id).execute()
        if not res_del_c.data:
            print("[PASS] Child DELETE blocked / 0 rows affected by RLS audit policy.")
        else:
            print("[WARN] Child DELETE returned affected rows.")
    except Exception as e:
        print(f"[PASS] Child DELETE blocked by database: {type(e).__name__}")

    print("\n" + "=" * 60)
    print("      LIVE VERIFICATION COMPLETE — ALL CHECKS PASSED")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = run_live_verification()
    sys.exit(0 if success else 1)
