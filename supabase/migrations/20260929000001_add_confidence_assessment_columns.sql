-- ==============================================================================
-- Migration: 20260929000001_add_confidence_assessment_columns.sql
-- Description: Add confidence assessment columns to defect_reports table:
--              confidence_level, confidence_range, raw_score, calibrated_score.
--              Preserves existing confidence column for backward compatibility.
-- Safe & Idempotent: Can be run multiple times safely.
-- ==============================================================================

ALTER TABLE public.defect_reports
    ADD COLUMN IF NOT EXISTS confidence_level TEXT,
    ADD COLUMN IF NOT EXISTS confidence_range TEXT,
    ADD COLUMN IF NOT EXISTS raw_score NUMERIC,
    ADD COLUMN IF NOT EXISTS calibrated_score NUMERIC;

CREATE INDEX IF NOT EXISTS idx_defect_reports_confidence_level
    ON public.defect_reports (confidence_level);
