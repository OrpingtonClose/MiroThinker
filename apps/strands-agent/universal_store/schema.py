"""Unified schema for the Universal Store Architecture.

ALL schema additions are additive only. No columns are dropped.
The append-only contract is preserved across all tables.
"""

# ---------------------------------------------------------------------------
# Core: conditions (base table + extensions)
# ---------------------------------------------------------------------------

# Base table — created here so universal_store can boot without a pre-existing
# ConditionStore database.  All columns use IF NOT EXISTS / DEFAULT so this is
# safe to run against an already-populated database too.
CONDITIONS_BASE_TABLE = """
CREATE TABLE IF NOT EXISTS conditions (
    id INTEGER PRIMARY KEY,
    fact TEXT NOT NULL,
    source_url TEXT DEFAULT '',
    source_type TEXT DEFAULT '',
    source_ref TEXT DEFAULT '',
    row_type TEXT DEFAULT 'finding',
    parent_id INTEGER,
    related_id INTEGER,
    consider_for_use BOOLEAN DEFAULT TRUE,
    obsolete_reason TEXT DEFAULT '',
    angle TEXT DEFAULT '',
    strategy TEXT DEFAULT '',
    expansion_depth INTEGER DEFAULT 0,
    created_at TEXT DEFAULT '',
    iteration INTEGER DEFAULT 0,
    confidence FLOAT DEFAULT 0.5,
    trust_score FLOAT DEFAULT 0.5,
    novelty_score FLOAT DEFAULT 0.5,
    specificity_score FLOAT DEFAULT 0.5,
    relevance_score FLOAT DEFAULT 0.5,
    actionability_score FLOAT DEFAULT 0.5,
    duplication_score FLOAT DEFAULT -1.0,
    fabrication_risk FLOAT DEFAULT 0.0,
    verification_status TEXT DEFAULT '',
    scored_at TEXT DEFAULT '',
    score_version INTEGER DEFAULT 0,
    composite_quality FLOAT DEFAULT -1.0,
    information_density FLOAT DEFAULT -1.0,
    cross_ref_boost FLOAT DEFAULT 0.0,
    processing_status TEXT DEFAULT 'raw',
    expansion_tool TEXT DEFAULT 'none',
    expansion_hint TEXT DEFAULT '',
    expansion_fulfilled BOOLEAN DEFAULT FALSE,
    expansion_gap TEXT DEFAULT '',
    expansion_priority FLOAT DEFAULT 0.0,
    cluster_id INTEGER DEFAULT -1,
    cluster_rank INTEGER DEFAULT 0,
    contradiction_flag BOOLEAN DEFAULT FALSE,
    contradiction_partner INTEGER DEFAULT -1,
    staleness_penalty FLOAT DEFAULT 0.0,
    relationship_score FLOAT DEFAULT 0.0,
    phase TEXT DEFAULT '',
    parent_ids TEXT DEFAULT '',
    source_model TEXT DEFAULT '',
    source_run TEXT DEFAULT '',
    evaluation_count INTEGER DEFAULT 0,
    last_evaluated_at TEXT DEFAULT '',
    evaluator_angles TEXT DEFAULT '',
    mcp_research_status TEXT DEFAULT '',
    information_gain FLOAT DEFAULT 0.0
);
"""

CONDITIONS_EXTENDED_COLUMNS = """
ALTER TABLE conditions ADD COLUMN IF NOT EXISTS run_number INTEGER DEFAULT 0;
ALTER TABLE conditions ADD COLUMN IF NOT EXISTS run_id TEXT DEFAULT '';
ALTER TABLE conditions ADD COLUMN IF NOT EXISTS diffusion_pass INTEGER DEFAULT 0;
ALTER TABLE conditions ADD COLUMN IF NOT EXISTS diffusion_phase TEXT DEFAULT '';
ALTER TABLE conditions ADD COLUMN IF NOT EXISTS section_angle TEXT DEFAULT '';
ALTER TABLE conditions ADD COLUMN IF NOT EXISTS convergence_status TEXT DEFAULT 'pending';
ALTER TABLE conditions ADD COLUMN IF NOT EXISTS critique_target_id INTEGER;
ALTER TABLE conditions ADD COLUMN IF NOT EXISTS diffusion_report_id INTEGER;
ALTER TABLE conditions ADD COLUMN IF NOT EXISTS scores_json TEXT DEFAULT '{}';
ALTER TABLE conditions ADD COLUMN IF NOT EXISTS provenance_system TEXT DEFAULT '';
ALTER TABLE conditions ADD COLUMN IF NOT EXISTS query_type TEXT DEFAULT '';
ALTER TABLE conditions ADD COLUMN IF NOT EXISTS depth_evaluated BOOLEAN DEFAULT FALSE;
ALTER TABLE conditions ADD COLUMN IF NOT EXISTS understandability_evaluated BOOLEAN DEFAULT FALSE;
"""

# Ghost columns that MUST be populated
GHOST_COLUMN_FIXES = """
-- cluster_id: union-find clustering on semantic similarity
UPDATE conditions SET cluster_id = 0 WHERE cluster_id IS NULL;

-- contradiction_flag: contradiction detection scoring
UPDATE conditions SET contradiction_flag = FALSE WHERE contradiction_flag IS NULL;

-- trust_score: activated by SourceQualityCurator
UPDATE conditions SET trust_score = 0.5 WHERE trust_score IS NULL;
"""

# ---------------------------------------------------------------------------
# New tables
# ---------------------------------------------------------------------------

RUNS_TABLE = """
CREATE TABLE IF NOT EXISTS runs (
    run_number INTEGER PRIMARY KEY,
    run_id TEXT UNIQUE NOT NULL,
    started_at TEXT,
    ended_at TEXT,
    convergence_reason TEXT,
    total_queries INTEGER DEFAULT 0,
    total_cost_usd FLOAT DEFAULT 0.0,
    source_query TEXT DEFAULT '',
    vm_id TEXT DEFAULT '',
    status TEXT DEFAULT 'running'
);
"""

SCORE_HISTORY_TABLE = """
CREATE TABLE IF NOT EXISTS score_history (
    id INTEGER PRIMARY KEY,
    condition_id INTEGER NOT NULL,
    run_number INTEGER,
    evaluator_angle TEXT DEFAULT '',
    query_type TEXT DEFAULT '',
    old_confidence FLOAT, new_confidence FLOAT,
    old_novelty_score FLOAT, new_novelty_score FLOAT,
    old_specificity_score FLOAT, new_specificity_score FLOAT,
    old_relevance_score FLOAT, new_relevance_score FLOAT,
    old_actionability_score FLOAT, new_actionability_score FLOAT,
    old_fabrication_risk FLOAT, new_fabrication_risk FLOAT,
    magnitude FLOAT,
    evaluated_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

LESSONS_TABLE = """
CREATE TABLE IF NOT EXISTS lessons (
    id INTEGER PRIMARY KEY,
    lesson_type TEXT NOT NULL,
    fact TEXT NOT NULL,
    run_id TEXT NOT NULL,
    run_number INTEGER,
    angle TEXT DEFAULT '',
    query_type TEXT DEFAULT '',
    source_url TEXT DEFAULT '',
    source_type TEXT DEFAULT '',
    relevance_score FLOAT DEFAULT 0.5,
    confidence FLOAT DEFAULT 0.5,
    metadata JSON,
    halflife_runs INTEGER DEFAULT 3,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

LESSON_APPLICATIONS_TABLE = """
CREATE TABLE IF NOT EXISTS lesson_applications (
    id INTEGER PRIMARY KEY,
    lesson_id INTEGER NOT NULL,
    target_run_id TEXT NOT NULL,
    target_actor TEXT DEFAULT '',
    application_method TEXT DEFAULT '',
    applied_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

SEMANTIC_CONNECTIONS_TABLE = """
CREATE TABLE IF NOT EXISTS semantic_connections (
    id INTEGER PRIMARY KEY,
    source_condition_id INTEGER NOT NULL,
    target_condition_id INTEGER NOT NULL,
    connection_type TEXT NOT NULL,
    directionality TEXT DEFAULT 'symmetric',
    detection_stage TEXT DEFAULT 'heuristic',
    confidence FLOAT DEFAULT 0.0,
    evidence_text TEXT DEFAULT '',
    evaluator_angle TEXT DEFAULT '',
    run_number INTEGER,
    embedding_similarity FLOAT,
    llm_verdict_json TEXT DEFAULT '{}',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    consider_for_use BOOLEAN DEFAULT TRUE
);
"""

SOURCE_FINGERPRINTS_TABLE = """
CREATE TABLE IF NOT EXISTS source_fingerprints (
    id INTEGER PRIMARY KEY,
    source_url TEXT NOT NULL,
    source_type TEXT NOT NULL,
    byte_sha256 TEXT,
    text_sha256 TEXT,
    text_simhash TEXT,
    isbn TEXT,
    doi TEXT,
    ol_key TEXT,
    content_hash TEXT,
    ingested_at TEXT,
    last_accessed_at TEXT,
    extraction_status TEXT DEFAULT 'pending',
    download_method TEXT,
    legal_tier TEXT,
    metadata_json TEXT DEFAULT '{}'
);
"""

CHUNKS_TABLE = """
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    source_fingerprint_id INTEGER NOT NULL,
    parent_condition_id INTEGER,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    chapter_title TEXT DEFAULT '',
    section_title TEXT DEFAULT '',
    page_start INTEGER,
    page_end INTEGER,
    embedding FLOAT[768],
    token_count INTEGER,
    char_count INTEGER,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

SOURCE_UTILITY_LOG_TABLE = """
CREATE TABLE IF NOT EXISTS source_utility_log (
    id INTEGER PRIMARY KEY,
    source_fingerprint_id INTEGER NOT NULL,
    query_embedding FLOAT[768],
    query_text TEXT,
    angle TEXT,
    times_queried INTEGER DEFAULT 0,
    chunks_retrieved INTEGER DEFAULT 0,
    findings_produced INTEGER DEFAULT 0,
    avg_chunk_relevance FLOAT DEFAULT 0.0,
    utility_score FLOAT DEFAULT 0.5,
    utility_verdict TEXT DEFAULT 'pending',
    last_queried_at TEXT,
    last_verdict_at TEXT,
    block_future_downloads BOOLEAN DEFAULT FALSE,
    block_reason TEXT
);
"""

SOURCE_QUALITY_REGISTRY_TABLE = """
CREATE TABLE IF NOT EXISTS source_quality_registry (
    id INTEGER PRIMARY KEY,
    domain TEXT NOT NULL,
    source_type TEXT NOT NULL,
    authority_score FLOAT DEFAULT 0.5,
    avg_recency_score FLOAT DEFAULT 0.5,
    avg_finding_confidence FLOAT DEFAULT 0.5,
    fetch_count INTEGER DEFAULT 0,
    successful_fetch_count INTEGER DEFAULT 0,
    total_cost_usd FLOAT DEFAULT 0.0,
    total_info_gain_generated FLOAT DEFAULT 0.0,
    first_seen_at TEXT,
    last_seen_at TEXT,
    UNIQUE(domain, source_type)
);
"""

CONDITION_SOURCES_TABLE = """
CREATE TABLE IF NOT EXISTS condition_sources (
    condition_id INTEGER NOT NULL,
    source_registry_id INTEGER NOT NULL,
    source_url TEXT,
    extracted_fact TEXT,
    confidence_at_extraction FLOAT,
    PRIMARY KEY (condition_id, source_registry_id)
);
"""

CONDITION_EMBEDDINGS_TABLE = """
CREATE TABLE IF NOT EXISTS condition_embeddings (
    condition_id INTEGER PRIMARY KEY,
    embedding FLOAT[768],
    embedding_model TEXT DEFAULT '',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

TRACE_TABLE = """
CREATE TABLE IF NOT EXISTS trace_records (
    id INTEGER PRIMARY KEY,
    trace_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    actor_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    phase TEXT DEFAULT '',
    payload_json TEXT DEFAULT '{}',
    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
    latency_ms FLOAT DEFAULT 0.0,
    error TEXT DEFAULT '',
    stack_trace TEXT DEFAULT ''
);
"""

# ---------------------------------------------------------------------------
# Indices — without these the store is unusable at scale
# ---------------------------------------------------------------------------

INDICES = """
-- Covering index for Flock queries (PARAMOUNT)
CREATE INDEX IF NOT EXISTS idx_findings_covering ON conditions (
    consider_for_use, row_type, score_version,
    confidence DESC, novelty_score, fabrication_risk,
    specificity_score, relevance_score, actionability_score
) INCLUDE (id, fact, angle, information_gain, evaluation_count, cluster_id);

-- Semantic connections
CREATE INDEX IF NOT EXISTS idx_semconn_source ON semantic_connections(source_condition_id, connection_type);
CREATE INDEX IF NOT EXISTS idx_semconn_target ON semantic_connections(target_condition_id, connection_type);
CREATE INDEX IF NOT EXISTS idx_semconn_type ON semantic_connections(connection_type, confidence DESC);
CREATE INDEX IF NOT EXISTS idx_semconn_run ON semantic_connections(run_number);
CREATE INDEX IF NOT EXISTS idx_semconn_symmetric ON semantic_connections(
    LEAST(source_condition_id, target_condition_id),
    GREATEST(source_condition_id, target_condition_id)
);

-- Lessons
CREATE INDEX IF NOT EXISTS idx_lessons_actor_query ON lessons (
    lesson_type, angle, query_type, confidence DESC, relevance_score DESC
) INCLUDE (id, fact, metadata, run_id);
CREATE INDEX IF NOT EXISTS idx_lessons_temporal ON lessons (created_at DESC, lesson_type);
CREATE INDEX IF NOT EXISTS idx_lessons_run ON lessons (run_id, run_number);

-- Runs
CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status, started_at DESC);

-- Score history
CREATE INDEX IF NOT EXISTS idx_score_history_condition ON score_history(condition_id, evaluated_at DESC);
CREATE INDEX IF NOT EXISTS idx_score_history_run ON score_history(run_number, query_type);

-- Source tracking
CREATE INDEX IF NOT EXISTS idx_fp_text_simhash ON source_fingerprints(text_simhash);
CREATE INDEX IF NOT EXISTS idx_fp_isbn ON source_fingerprints(isbn);
CREATE INDEX IF NOT EXISTS idx_fp_content_hash ON source_fingerprints(content_hash);
CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source_fingerprint_id);
CREATE INDEX IF NOT EXISTS idx_chunks_parent ON chunks(parent_condition_id);
CREATE INDEX IF NOT EXISTS idx_sul_verdict ON source_utility_log(utility_verdict, block_future_downloads);
CREATE INDEX IF NOT EXISTS idx_sul_source ON source_utility_log(source_fingerprint_id);

-- Trace
CREATE INDEX IF NOT EXISTS idx_trace_run ON trace_records(run_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_trace_actor ON trace_records(actor_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_trace_type ON trace_records(event_type, timestamp DESC);

-- Cross-run queries
CREATE INDEX IF NOT EXISTS idx_conditions_run ON conditions(run_number, row_type, angle);
CREATE INDEX IF NOT EXISTS idx_conditions_cluster ON conditions(cluster_id, confidence DESC);
CREATE INDEX IF NOT EXISTS idx_conditions_contradiction ON conditions(contradiction_flag, confidence DESC);
"""

# ---------------------------------------------------------------------------
# DuckDB VSS vector indices (requires duckdb-vss extension)
# ---------------------------------------------------------------------------

VECTOR_INDICES = """
CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks USING HNSW (embedding) WITH (metric = 'cosine');
CREATE INDEX IF NOT EXISTS idx_condition_embeddings ON condition_embeddings USING HNSW (embedding) WITH (metric = 'cosine');
"""

# ---------------------------------------------------------------------------
# All DDL in execution order
# ---------------------------------------------------------------------------

ALL_DDL = [
    CONDITIONS_BASE_TABLE,
    CONDITIONS_EXTENDED_COLUMNS,
    RUNS_TABLE,
    SCORE_HISTORY_TABLE,
    LESSONS_TABLE,
    LESSON_APPLICATIONS_TABLE,
    SEMANTIC_CONNECTIONS_TABLE,
    SOURCE_FINGERPRINTS_TABLE,
    CHUNKS_TABLE,
    SOURCE_UTILITY_LOG_TABLE,
    SOURCE_QUALITY_REGISTRY_TABLE,
    CONDITION_SOURCES_TABLE,
    CONDITION_EMBEDDINGS_TABLE,
    TRACE_TABLE,
    INDICES,
    # VECTOR_INDICES,  -- uncomment after installing duckdb-vss
]


def _supports_index_include() -> bool:
    """Return True if the running DuckDB build supports INCLUDE in CREATE INDEX."""
    import duckdb as _duckdb
    try:
        conn = _duckdb.connect()
        conn.execute("CREATE TABLE _inc_probe (a INT, b INT)")
        conn.execute("CREATE INDEX _inc_probe_idx ON _inc_probe (a) INCLUDE (b)")
        conn.close()
        return True
    except Exception:
        return False


def get_all_ddl() -> str:
    import re
    ddl = "\n".join(ALL_DDL)
    if not _supports_index_include():
        # Strip INCLUDE (...) clauses from index definitions for older DuckDB builds
        ddl = re.sub(r"\)\s*INCLUDE\s*\([^)]*\)", ")", ddl)
    return ddl
