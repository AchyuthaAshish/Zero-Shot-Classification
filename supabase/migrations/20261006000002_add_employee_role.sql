-- ==============================================================================
-- Migration: 20261006000002_add_employee_role.sql
-- Description: Phase 5 Step 5.3: Employee Role Enactment on public.profiles
--              1. Adds role column with default 'employee' and NOT NULL.
--              2. Enforces CHECK constraint: role IN ('employee').
--                 (Extensible design: allows future addition of 'reviewer', 'admin').
--              3. Backfills any existing profile records to 'employee'.
--              4. Updates public.handle_new_user() trigger function to explicitly
--                 initialize role = 'employee' on auth.users registration.
--              5. Preserves all existing RLS policies and table constraints.
-- Safe & Idempotent: Can be run multiple times in Supabase SQL Editor safely.
-- ==============================================================================

-- 1. Add role column to public.profiles if not present
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = 'profiles'
          AND column_name = 'role'
    ) THEN
        ALTER TABLE public.profiles
            ADD COLUMN role TEXT NOT NULL DEFAULT 'employee';
    END IF;
END $$;

-- 2. Add or update CHECK constraint on allowed roles
-- Restricts roles to 'employee' exclusively during Step 5.3
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'chk_profiles_role'
    ) THEN
        ALTER TABLE public.profiles
            ADD CONSTRAINT chk_profiles_role
            CHECK (role IN ('employee'));
    END IF;
END $$;

-- 3. Idempotently backfill any existing profiles to 'employee'
UPDATE public.profiles
SET role = 'employee'
WHERE role IS NULL OR role != 'employee';

-- 4. Update handle_new_user trigger function to explicitly populate role = 'employee'
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public, auth
AS $$
BEGIN
    INSERT INTO public.profiles (id, display_name, email, role, created_at, updated_at)
    VALUES (
        NEW.id,
        COALESCE(
            NEW.raw_user_meta_data->>'full_name',
            NEW.raw_user_meta_data->>'display_name',
            NEW.raw_user_meta_data->>'name',
            split_part(NEW.email, '@', 1),
            'Operator'
        ),
        NEW.email,
        'employee',
        NOW(),
        NOW()
    )
    ON CONFLICT (id) DO NOTHING;
    RETURN NEW;
END;
$$;

-- 5. Ensure trigger on auth.users remains attached
DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
CREATE TRIGGER on_auth_user_created
    AFTER INSERT ON auth.users
    FOR EACH ROW
    EXECUTE FUNCTION public.handle_new_user();
