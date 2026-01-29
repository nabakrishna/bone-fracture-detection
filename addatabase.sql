
-- 1. INFRASTRUCTURE SETUP
CREATE EXTENSION IF NOT EXISTS "uuid-ossp"; -- For unique IDs
CREATE EXTENSION IF NOT EXISTS "lo";        -- For Large Object (TB-sized files)

-- 2. THE MASTER VAULT TABLE
CREATE TABLE universal_vault (
    entry_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    file_name TEXT NOT NULL,
    file_category VARCHAR(50) NOT NULL, -- e.g., 'bone_xray', 'python_script', 'patient_data'
    mime_type VARCHAR(100),             -- e.g., 'image/jpeg', 'text/x-python'
    
    -- STORAGE 1: Small content (Code, Text, Small Thumbnails)
    raw_binary BYTEA,
    
    -- STORAGE 2: Heavy content (Large X-rays, Video, Datasets)
    large_file_oid OID,
    
    -- STORAGE 3: Dynamic Data (Everything else: Names, Dates, AI Bounding Boxes)
    -- This makes the table "Full Dynamic"
    metadata JSONB DEFAULT '{}'::jsonb,
    
    -- SYSTEM TRACKING
    checksum TEXT, -- For data integrity (MD5/SHA)
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 3. ADVANCED INDEXING (For Lightning Speed)
CREATE INDEX idx_vault_metadata_search ON universal_vault USING GIN (metadata);
CREATE INDEX idx_vault_category ON universal_vault (file_category);

-- 4. TRIGGER: AUTO-CLEANUP LARGE OBJECTS
-- Prevents "Ghost Files" from taking up space when a row is deleted.
CREATE OR REPLACE FUNCTION clean_large_objects()
RETURNS TRIGGER AS $$
BEGIN
    IF OLD.large_file_oid IS NOT NULL THEN
        PERFORM lo_unlink(OLD.large_file_oid);
    END IF;
    RETURN OLD;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_vault_cleanup
BEFORE DELETE ON universal_vault
FOR EACH ROW EXECUTE FUNCTION clean_large_objects();

-- 5. THE "EASY ACCESS" PROCEDURE: SAVE DATA
-- This allows you to insert anything with ONE command.
CREATE OR REPLACE PROCEDURE save_to_vault(
    p_name TEXT, 
    p_cat TEXT, 
    p_mime TEXT, 
    p_meta JSONB,
    p_content BYTEA DEFAULT NULL
)
LANGUAGE plpgsql AS $$
BEGIN
    INSERT INTO universal_vault (file_name, file_category, mime_type, metadata, raw_binary)
    VALUES (p_name, p_cat, p_mime, p_meta, p_content);
END;
$$;

-- 6. THE "EASY ACCESS" FUNCTION: GET DATA BY KEY
-- Example: Get all scans for a specific Patient Name hidden inside the JSON.
CREATE OR REPLACE FUNCTION find_by_meta_key(search_key TEXT, search_val TEXT)
RETURNS TABLE(entry_id UUID, file_name TEXT, metadata JSONB) AS $$
BEGIN
    RETURN QUERY
    SELECT v.entry_id, v.file_name, v.metadata
    FROM universal_vault v
    WHERE v.metadata->>search_key = search_val;
END;
$$ LANGUAGE plpgsql;

-- 7. THE "EASY ACCESS" VIEW: DATA DASHBOARD
-- Flattens the dynamic JSON so you can read it like a normal Excel sheet.
CREATE OR REPLACE VIEW vault_dashboard AS
SELECT 
    entry_id,
    file_name,
    file_category,
    metadata->>'patient_name' AS patient,
    metadata->>'fracture_detected' AS has_fracture,
    metadata->>'confidence' AS ai_score,
    PG_SIZE_PRETTY(OCTET_LENGTH(raw_binary)) AS size_on_disk,
    created_at
FROM universal_vault;

-- 8. SECURITY POLICY (Optional: Protect Data)
ALTER TABLE universal_vault ENABLE ROW LEVEL SECURITY;

-- Allow the owner to see everything, restrict others (Basic Template)
CREATE POLICY medical_staff_access ON universal_vault
    FOR SELECT TO public
    USING (file_category != 'restricted');

-- 9. ANALYTICS FUNCTION
-- Counts how many files of each type exist.
CREATE OR REPLACE FUNCTION get_vault_stats()
RETURNS TABLE(category TEXT, total_files BIGINT) AS $$
BEGIN
    RETURN QUERY
    SELECT file_category, COUNT(*) 
    FROM universal_vault 
    GROUP BY file_category;
END;
$$ LANGUAGE plpgsql;
