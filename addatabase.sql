-- 1. EXTENSIONS
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "lo";

-- 2. MASTER TABLE (Partitioned by Date for Maximum Performance)
CREATE TABLE universal_vault (
    entry_id UUID DEFAULT uuid_generate_v4(),
    file_name TEXT NOT NULL,
    file_category VARCHAR(50), -- 'XRAY', 'PYTHON_CODE', 'PATIENT_DOC'
    mime_type VARCHAR(100),    -- 'image/jpeg', 'text/plain'
    
    -- STORAGE A: Small data (Code snippets, thumbnails)
    raw_content BYTEA,
    
    -- STORAGE B: Large data (High-res X-rays, 4TB limit)
    large_object_oid OID,
    
    -- THE DYNAMIC CORE: Stores any names, AI boxes, or custom tags
    metadata JSONB DEFAULT '{}'::jsonb,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    PRIMARY KEY (entry_id, created_at)
) PARTITION BY RANGE (created_at);

-- 3. AUTOMATIC PARTITIONS (Divide data by Year)
CREATE TABLE vault_2025 PARTITION OF universal_vault 
    FOR VALUES FROM ('2025-01-01') TO ('2026-01-01');

CREATE TABLE vault_2026 PARTITION OF universal_vault 
    FOR VALUES FROM ('2026-01-01') TO ('2027-01-01');

-- 4. ADVANCED INDEXING
-- GIN Index allows searching "Inside" the dynamic metadata
CREATE INDEX idx_vault_metadata ON universal_vault USING GIN (metadata);
CREATE INDEX idx_vault_cat_date ON universal_vault (file_category, created_at DESC);

-- 5. TRIGGER: PREVENT GHOST DATA
-- Deletes the Large Object from system memory when a row is deleted
CREATE OR REPLACE FUNCTION clean_large_files() RETURNS TRIGGER AS $$
BEGIN
    IF OLD.large_object_oid IS NOT NULL THEN
        PERFORM lo_unlink(OLD.large_object_oid);
    END IF;
    RETURN OLD;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_delete_large_files
BEFORE DELETE ON universal_vault
FOR EACH ROW EXECUTE FUNCTION clean_large_files();

-- 6. EASY ACCESS PROCEDURE: SAVE ANYTHING
-- Use this to insert any file type with a single command
CREATE OR REPLACE PROCEDURE vault_insert(
    p_name TEXT, p_cat TEXT, p_meta JSONB, p_raw BYTEA DEFAULT NULL
) LANGUAGE plpgsql AS $$
BEGIN
    INSERT INTO universal_vault (file_name, file_category, metadata, raw_content)
    VALUES (p_name, p_cat, p_meta, p_raw);
END;
$$;

-- 7. EASY ACCESS VIEW: DATA DASHBOARD
-- Turns complex JSON into a readable spreadsheet format
CREATE OR REPLACE VIEW vault_summary AS
SELECT 
    entry_id,
    file_name,
    file_category,
    metadata->>'patient_name' AS patient,
    metadata->>'fracture_status' AS status,
    (metadata->>'confidence')::FLOAT AS ai_score,
    created_at
FROM universal_vault;

-- 8. ANALYTICS FUNCTION
-- Get a report of what is in your vault
CREATE OR REPLACE FUNCTION vault_stats()
RETURNS TABLE(cat TEXT, total_files BIGINT, last_update TIMESTAMP) AS $$
BEGIN
    RETURN QUERY
    SELECT file_category, COUNT(*), MAX(created_at)
    FROM universal_vault
    GROUP BY file_category;
END;
$$ LANGUAGE plpgsql;

-- 9. EXAMPLE USAGE:
-- CALL vault_insert('fracture.jpg', 'XRAY', '{"patient_name": "N.K. Hazarika", "fracture_status": "positive", "confidence": 0.95}');
