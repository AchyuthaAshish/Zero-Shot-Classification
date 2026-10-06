-- ==============================================================================
-- Migration: 20261006000003_rls_redesign.sql
-- Description: Phase 5 Step 5.6: Row Level Security (RLS) Redesign
--              1. Adds user_id UUID column to public.defect_reports referencing auth.users(id).
--              2. Establishes ON DELETE SET NULL to preserve historical industrial records
--                 if an operator account is deactivated or deleted.
--              3. Creates performance index on public.defect_reports(user_id).
--              4. Preserves legacy rows (user_id IS NULL) without fabricating synthetic ownership.
--              5. Hardens public.profiles RLS and adds BEFORE UPDATE trigger preventing
--                 unauthorized modification of the server-controlled 'role' column.
--              6. Restricts public.defect_reports RLS:
--                 - Authenticated employees can INSERT reports for themselves (user_id = auth.uid()).
--                 - Authenticated employees can SELECT their own reports (user_id = auth.uid()).
--                 - UPDATE and DELETE are blocked for client roles (immutable audit trail).
--                 - Anonymous access is denied.
--              7. Restricts public.defect_report_items RLS:
--                 - Authenticated employees can SELECT child items only if they own the parent report.
--                 - Authenticated employees can INSERT child items only for parent reports they own.
--                 - UPDATE and DELETE are blocked.
--                 - Anonymous access is denied.
-- Safe & Idempotent: Can be run multiple times in Supabase SQL Editor safely.
-- ==============================================================================

-- 1. Add user_id column to public.defect_reports if not present
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = 'defect_reports'
          AND column_name = 'user_id'
    ) THEN
        ALTER TABLE public.defect_reports
            ADD COLUMN user_id UUID REFERENCES auth.users(id) ON DELETE SET NULL;
    END IF;
END $$;

-- 2. Performance Index for user_id on defect_reports
CREATE INDEX IF NOT EXISTS idx_defect_reports_user_id
    ON public.defect_reports (user_id);

-- 3. Hardened Profile Role Protection Trigger
-- Prevents client-side modification of 'role' via direct PostgREST or SQL updates
CREATE OR REPLACE FUNCTION public.protect_profile_role()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
BEGIN
    IF NEW.role IS DISTINCT FROM OLD.role THEN
        RAISE EXCEPTION 'Modifying profile role is not permitted.';
    END IF;
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_protect_profile_role ON public.profiles;
CREATE TRIGGER trg_protect_profile_role
    BEFORE UPDATE ON public.profiles
    FOR EACH ROW
    EXECUTE FUNCTION public.protect_profile_role();

-- 4. Enable RLS on all tables
ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.defect_reports ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.defect_report_items ENABLE ROW LEVEL SECURITY;

-- 5. Profiles RLS Policies (Idempotent Refresh)
DROP POLICY IF EXISTS "Users can view own profile" ON public.profiles;
CREATE POLICY "Users can view own profile"
    ON public.profiles
    FOR SELECT
    TO authenticated
    USING ((SELECT auth.uid()) = id);

DROP POLICY IF EXISTS "Users can update own profile" ON public.profiles;
CREATE POLICY "Users can update own profile"
    ON public.profiles
    FOR UPDATE
    TO authenticated
    USING ((SELECT auth.uid()) = id)
    WITH CHECK ((SELECT auth.uid()) = id);

DROP POLICY IF EXISTS "Users can insert own profile" ON public.profiles;
CREATE POLICY "Users can insert own profile"
    ON public.profiles
    FOR INSERT
    TO authenticated
    WITH CHECK ((SELECT auth.uid()) = id);

-- 6. Defect Reports RLS Policies
-- Drop old permissive / legacy policies
DROP POLICY IF EXISTS "defect_reports_insert_policy" ON public.defect_reports;
DROP POLICY IF EXISTS "defect_reports_select_policy" ON public.defect_reports;
DROP POLICY IF EXISTS "Users can insert own reports" ON public.defect_reports;
DROP POLICY IF EXISTS "Users can view own reports" ON public.defect_reports;

-- Policy A: Authenticated employees can INSERT reports for themselves
CREATE POLICY "Users can insert own reports"
    ON public.defect_reports
    FOR INSERT
    TO authenticated
    WITH CHECK ((SELECT auth.uid()) = user_id);

-- Policy B: Authenticated employees can SELECT only their own reports
CREATE POLICY "Users can view own reports"
    ON public.defect_reports
    FOR SELECT
    TO authenticated
    USING ((SELECT auth.uid()) = user_id);

-- Note: UPDATE and DELETE have NO policies for public/authenticated, blocking them by default.

-- 7. Defect Report Items RLS Policies
-- Drop old permissive / legacy policies
DROP POLICY IF EXISTS "defect_report_items_insert_policy" ON public.defect_report_items;
DROP POLICY IF EXISTS "defect_report_items_select_policy" ON public.defect_report_items;
DROP POLICY IF EXISTS "Users can view own report items" ON public.defect_report_items;
DROP POLICY IF EXISTS "Users can insert items for own reports" ON public.defect_report_items;

-- Policy A: Authenticated employees can SELECT child items only if they own the parent defect report
CREATE POLICY "Users can view own report items"
    ON public.defect_report_items
    FOR SELECT
    TO authenticated
    USING (
        EXISTS (
            SELECT 1
            FROM public.defect_reports dr
            WHERE dr.id = defect_report_items.report_id
              AND dr.user_id = (SELECT auth.uid())
        )
    );

-- Policy B: Authenticated employees can INSERT child items only for parent defect reports they own
CREATE POLICY "Users can insert items for own reports"
    ON public.defect_report_items
    FOR INSERT
    TO authenticated
    WITH CHECK (
        EXISTS (
            SELECT 1
            FROM public.defect_reports dr
            WHERE dr.id = defect_report_items.report_id
              AND dr.user_id = (SELECT auth.uid())
        )
    );

-- Note: UPDATE and DELETE have NO policies, blocking them by default.
