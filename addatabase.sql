-- 1. EXTENSI
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "lo";
CREATE TABLE universal_vault (
    entry_id UUID DEFAULT uuid_generate_v4(),
    file_name TEXT NOT NULL,
    file_category VARCHAR(50), 
    mime_type VARCHAR(100),    
    raw_content BYTEA,
    large_object_oid OID,
    metadata JSONB DEFAULT '{}'::jsonb,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    PRIMARY KEY (entry_id, created_at)
) PARTITION BY RANGE (created_at);

-- 3. AUTOMATIC PARTITIONS (Divide data by Year)
CREATE TABLE vault_2025 PARTITION OF universal_vault 
    FOR VALUES FROM ('2025-01-01') TO ('2026-01-01');

CREATE TABLE vault_2026 PARTITION OF universal_vault 
    FOR VALUES FROM ('2026-01-01') TO ('2027-01-01');

CREATE INDEX idx_vault_metadata ON universal_vault USING GIN (metadata);
CREATE INDEX idx_vault_cat_date ON universal_vault (file_category, created_at DESC);

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

CREATE OR REPLACE PROCEDURE vault_insert(
    p_name TEXT, p_cat TEXT, p_meta JSONB, p_raw BYTEA DEFAULT NULL
) LANGUAGE plpgsql AS $$
BEGIN
    INSERT INTO universal_vault (file_name, file_category, metadata, raw_content)
    VALUES (p_name, p_cat, p_meta, p_raw);
END;
$$;

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

CREATE OR REPLACE FUNCTION vault_stats()
RETURNS TABLE(cat TEXT, total_files BIGINT, last_update TIMESTAMP) AS $$
BEGIN
    RETURN QUERY
    SELECT file_category, COUNT(*), MAX(created_at)
    FROM universal_vault
    GROUP BY file_category;
END;
$$ LANGUAGE plpgsql;
