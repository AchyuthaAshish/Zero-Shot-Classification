-- ==============================================================================
-- Migration: 20261006000001_create_profiles.sql
-- Description: User profiles table linked to Supabase Auth:
--              1. Creates public.profiles with 1:1 foreign key linkage to auth.users(id).
--              2. Enforces non-empty display_name constraints and audit timestamps.
--              3. Enables Row Level Security (RLS) with restrictive policies:
--                 - Authenticated users can view their own profile.
--                 - Authenticated users can update their own profile.
--                 - Authenticated users can insert their own profile.
--                 - Anonymous users have zero read/write access.
--              4. Implements idempotent handle_new_user trigger function for automated
--                 synchronization on auth.users registration (SECURITY DEFINER with safe search_path).
-- Safe & Idempotent: Can be run multiple times in Supabase SQL Editor safely.
-- ==============================================================================

-- 1. Create public.profiles table
CREATE TABLE IF NOT EXISTS public.profiles (
    id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    display_name TEXT NOT NULL CHECK (length(trim(display_name)) > 0),
    email TEXT CHECK (email IS NULL OR length(trim(email)) > 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT chk_profiles_display_name_len CHECK (length(trim(display_name)) <= 100)
);

-- 2. Performance Indexes
CREATE INDEX IF NOT EXISTS idx_profiles_created_at
    ON public.profiles (created_at DESC);

-- 3. Enable Row Level Security (RLS)
ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;

-- 4. RLS Policies
-- Policy A: Authenticated users can view only their own profile
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_policies
        WHERE tablename = 'profiles' AND policyname = 'Users can view own profile'
    ) THEN
        CREATE POLICY "Users can view own profile"
            ON public.profiles
            FOR SELECT
            TO authenticated
            USING ((SELECT auth.uid()) = id);
    END IF;
END $$;

-- Policy B: Authenticated users can update only their own profile
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_policies
        WHERE tablename = 'profiles' AND policyname = 'Users can update own profile'
    ) THEN
        CREATE POLICY "Users can update own profile"
            ON public.profiles
            FOR UPDATE
            TO authenticated
            USING ((SELECT auth.uid()) = id)
            WITH CHECK ((SELECT auth.uid()) = id);
    END IF;
END $$;

-- Policy C: Authenticated users can insert their own profile
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_policies
        WHERE tablename = 'profiles' AND policyname = 'Users can insert own profile'
    ) THEN
        CREATE POLICY "Users can insert own profile"
            ON public.profiles
            FOR INSERT
            TO authenticated
            WITH CHECK ((SELECT auth.uid()) = id);
    END IF;
END $$;

-- 5. Profile Synchronization Trigger Function
-- Automatically creates a public.profiles row when a new user registers in auth.users
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public, auth
AS $$
BEGIN
    INSERT INTO public.profiles (id, display_name, email, created_at, updated_at)
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
        NOW(),
        NOW()
    )
    ON CONFLICT (id) DO NOTHING;
    RETURN NEW;
END;
$$;

-- 6. Attach Trigger to auth.users (idempotent)
DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
CREATE TRIGGER on_auth_user_created
    AFTER INSERT ON auth.users
    FOR EACH ROW
    EXECUTE FUNCTION public.handle_new_user();
