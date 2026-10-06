-- ==============================================================================
-- Migration: 20260927000001_create_defect_reports.sql
-- Description: Create defect_reports table with strict category CHECK constraint,
--              performance indexes, and Row Level Security (RLS) policies.
-- Safe & Idempotent: Can be run multiple times in Supabase SQL Editor safely.
-- ==============================================================================

-- 1. Create defect_reports table if it does not already exist
CREATE TABLE IF NOT EXISTS public.defect_reports (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    reporter_name TEXT NOT NULL CHECK (length(trim(reporter_name)) > 0),
    employee_id TEXT NOT NULL CHECK (length(trim(employee_id)) > 0),
    defect_description TEXT NOT NULL CHECK (length(trim(defect_description)) > 0),
    category TEXT NOT NULL CONSTRAINT chk_defect_category CHECK (category IN (
        'Mechanical Fault',
        'Electrical Fault',
        'Sensor Fault',
        'Temperature Fault',
        'Software Fault',
        'Power Supply Fault',
        'Communication Fault',
        'Unknown'
    )),
    confidence NUMERIC,
    reliability TEXT,
    explanation TEXT,
    classification_mode TEXT,
    provider TEXT,
    model TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 2. Performance Indexes
CREATE INDEX IF NOT EXISTS idx_defect_reports_created_at 
    ON public.defect_reports (created_at DESC);

CREATE INDEX IF NOT EXISTS idx_defect_reports_employee_id 
    ON public.defect_reports (employee_id);

CREATE INDEX IF NOT EXISTS idx_defect_reports_category 
    ON public.defect_reports (category);

-- 3. Enable Row Level Security (RLS)
ALTER TABLE public.defect_reports ENABLE ROW LEVEL SECURITY;

-- 4. Minimal Application RLS Policies
-- Allow application clients (anon / authenticated) to insert completed defect reports
DROP POLICY IF EXISTS "defect_reports_insert_policy" ON public.defect_reports;
CREATE POLICY "defect_reports_insert_policy"
    ON public.defect_reports
    FOR INSERT
    TO public
    WITH CHECK (true);

-- Allow application clients to query/read reports for the History dashboard
DROP POLICY IF EXISTS "defect_reports_select_policy" ON public.defect_reports;
CREATE POLICY "defect_reports_select_policy"
    ON public.defect_reports
    FOR SELECT
    TO public
    USING (true);

-- Disallow UPDATE and DELETE for public clients to maintain an immutable audit log.
