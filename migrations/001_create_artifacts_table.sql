-- Migration: Create artifacts table for JSONB artifact store
-- Run this via Supabase SQL Editor (Dashboard → SQL Editor → New Query → Paste → Run)

-- Create artifacts table
CREATE TABLE IF NOT EXISTS artifacts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    artifact_type VARCHAR(50) NOT NULL,
    artifact_key VARCHAR(255) NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,
    run_id UUID,
    data JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Unique constraint: one version per artifact
    UNIQUE(artifact_type, artifact_key, version)
);

-- Create indexes for query performance
CREATE INDEX IF NOT EXISTS idx_artifacts_type_key
    ON artifacts(artifact_type, artifact_key);

CREATE INDEX IF NOT EXISTS idx_artifacts_run_id
    ON artifacts(run_id)
    WHERE run_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_artifacts_created_at
    ON artifacts(created_at DESC);

-- Optional: GIN index for JSONB queries (add later if needed)
-- CREATE INDEX IF NOT EXISTS idx_artifacts_data_gin
--     ON artifacts USING GIN (data);

-- Add trigger to auto-update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_artifacts_updated_at
    BEFORE UPDATE ON artifacts
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Disable RLS for now (private app, trusted environment)
ALTER TABLE artifacts DISABLE ROW LEVEL SECURITY;

-- Grant permissions to authenticated and anon roles
GRANT ALL ON artifacts TO authenticated;
GRANT ALL ON artifacts TO anon;
GRANT ALL ON artifacts TO service_role;

-- Verify table was created
SELECT
    'artifacts table created successfully' AS status,
    COUNT(*) AS row_count
FROM artifacts;
