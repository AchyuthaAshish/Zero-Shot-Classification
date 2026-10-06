"""Database package for industrial defect classification storage."""
from database.supabase_client import (
    save_defect_report,
    save_multi_defect_report,
    validate_defect_item_payload,
    get_defect_reports,
    get_defect_report_items,
    get_reports_by_employee,
    get_reports_by_category,
    is_supabase_configured,
    DatabaseError,
    MultiDefectPersistenceError
)

__all__ = [
    "save_defect_report",
    "save_multi_defect_report",
    "validate_defect_item_payload",
    "get_defect_reports",
    "get_defect_report_items",
    "get_reports_by_employee",
    "get_reports_by_category",
    "is_supabase_configured",
    "DatabaseError",
    "MultiDefectPersistenceError"
]
