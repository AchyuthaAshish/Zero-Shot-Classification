-- ==============================================================================
-- Migration: 20260930000001_create_defect_report_items.sql
-- Description: Multi-defect database schema support:
--              1. Adds parent metadata columns to public.defect_reports
--                 (is_multi_defect, defect_count, validation_status).
--              2. Creates child table public.defect_report_items with 1-to-many
--                 foreign key linkage, strict category taxonomy, confidence/span
--                 constraints, performance indexes, and Row Level Security (RLS).
-- Safe & Idempotent: Can be run multiple times in Supabase SQL Editor safely.
-- ==============================================================================

-- 1. Add parent-level metadata columns to public.defect_reports (non-breaking)
ALTER TABLE public.defect_reports
    ADD COLUMN IF NOT EXISTS is_multi_defect BOOLEAN NOT NULL DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS defect_count INTEGER NOT NULL DEFAULT 1,
    ADD COLUMN IF NOT EXISTS validation_status TEXT;

-- 2. Add CHECK constraints to public.defect_reports metadata columns
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'chk_defect_reports_defect_count'
    ) THEN
        ALTER TABLE public.defect_reports
            ADD CONSTRAINT chk_defect_reports_defect_count CHECK (defect_count >= 0);
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'chk_defect_reports_validation_status'
    ) THEN
        ALTER TABLE public.defect_reports
            ADD CONSTRAINT chk_defect_reports_validation_status CHECK (
                validation_status IS NULL OR validation_status IN ('VALID', 'PARTIAL', 'INVALID', 'UNKNOWN')
            );
    END IF;
END $$;

-- 3. Add performance index on parent multi-defect indicator
CREATE INDEX IF NOT EXISTS idx_defect_reports_is_multi_defect
    ON public.defect_reports (is_multi_defect);


-- 4. Create child table public.defect_report_items
CREATE TABLE IF NOT EXISTS public.defect_report_items (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    report_id UUID NOT NULL REFERENCES public.defect_reports(id) ON DELETE CASCADE,
    defect_index INTEGER NOT NULL CHECK (defect_index >= 1),
    defect_id INTEGER,
    segment_id INTEGER,
    defect_text TEXT NOT NULL CHECK (length(trim(defect_text)) > 0),
    start_char INTEGER,
    end_char INTEGER,
    category TEXT NOT NULL CONSTRAINT chk_defect_item_category CHECK (category IN (
        'Mechanical Fault',
        'Electrical Fault',
        'Sensor Fault',
        'Temperature Fault',
        'Software Fault',
        'Power Supply Fault',
        'Communication Fault',
        'Unknown'
    )),
    confidence NUMERIC CONSTRAINT chk_defect_item_confidence CHECK (
        confidence IS NULL OR (confidence >= 0.0 AND confidence <= 1.0)
    ),
    confidence_level TEXT,
    confidence_range TEXT,
    raw_score NUMERIC,
    calibrated_score NUMERIC CONSTRAINT chk_defect_item_calibrated_score CHECK (
        calibrated_score IS NULL OR (calibrated_score >= 0.0 AND calibrated_score <= 1.0)
    ),
    top2_margin NUMERIC,
    reliability TEXT,
    explanation TEXT,
    classification_mode TEXT,
    provider TEXT,
    model TEXT,
    status TEXT CONSTRAINT chk_defect_item_status CHECK (
        status IS NULL OR status IN ('success', 'unknown', 'model_error', 'validation_error', 'system_error', 'low_confidence')
    ),
    is_ambiguous BOOLEAN NOT NULL DEFAULT FALSE,
    ambiguity_reason TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Constraints
    CONSTRAINT chk_defect_item_spans CHECK (
        (start_char IS NULL OR start_char >= 0) AND
        (end_char IS NULL OR end_char >= 0) AND
        (start_char IS NULL OR end_char IS NULL OR end_char > start_char)
    ),
    CONSTRAINT uq_defect_report_items_report_defect_index UNIQUE (report_id, defect_index)
);


-- 5. Performance Indexes for defect_report_items
CREATE INDEX IF NOT EXISTS idx_defect_report_items_report_id
    ON public.defect_report_items (report_id);

CREATE INDEX IF NOT EXISTS idx_defect_report_items_created_at
    ON public.defect_report_items (created_at DESC);

CREATE INDEX IF NOT EXISTS idx_defect_report_items_category
    ON public.defect_report_items (category);

CREATE INDEX IF NOT EXISTS idx_defect_report_items_report_defect_index
    ON public.defect_report_items (report_id, defect_index);


-- 6. Enable Row Level Security (RLS)
ALTER TABLE public.defect_report_items ENABLE ROW LEVEL SECURITY;


-- 7. Application RLS Policies (mirrors defect_reports access model)
-- Allow application clients (anon / authenticated) to insert child defect items
DROP POLICY IF EXISTS "defect_report_items_insert_policy" ON public.defect_report_items;
CREATE POLICY "defect_report_items_insert_policy"
    ON public.defect_report_items
    FOR INSERT
    TO public
    WITH CHECK (true);

-- Allow application clients to query/read child defect items for drill-down and history views
DROP POLICY IF EXISTS "defect_report_items_select_policy" ON public.defect_report_items;
CREATE POLICY "defect_report_items_select_policy"
    ON public.defect_report_items
    FOR SELECT
    TO public
    USING (true);

-- Disallow UPDATE and DELETE for public clients to maintain an immutable audit log.
