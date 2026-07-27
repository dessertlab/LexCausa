// ============================================================================
// LexCausa — Neo4j knowledge-graph showcase queries
// ----------------------------------------------------------------------------
// Read-only queries to inspect and visualize the legal knowledge graph, meant
// to replace the static schema figure with live, screenshottable output.
//
// How to run:
//   * Neo4j Browser  ->  http://localhost:7474  (run one statement at a time;
//                        the graph-returning ones render as an interactive graph)
//   * cypher-shell   ->  cypher-shell -u <NEO4J_USER> -p <NEO4J_PASSWORD> \
//                          -f queries/neo4j_schema_showcase.cypher
//
// SAFE to run while a DoE campaign is in progress: every statement here is
// read-only (MATCH / RETURN / SHOW / CALL db.schema...). Do NOT stop or restart
// Neo4j while runs are executing.
//
// Schema (from src/db/db_orchestrator.py):
//   Nodes
//     Codice   {name, description}
//     Libro    {name, codice, description}
//     Statute  {statute_id, articolo, titolo, testo, libro, source,
//               full_text, embedding, reference, external_reference}
//     Precedent{precedent_id, title, summary, year, court, court_level,
//               materia, source, url}
//   Relationships
//     (Codice)-[:CONTAINS]->(Libro)
//     (Statute)-[:BELONGS_TO]->(Libro)
//     (Statute)-[:CITES {kind}]->(Statute)
// ============================================================================


// ============================================================================
// A. SCHEMA MODEL  — the direct replacement for the static figure (renders as a graph)
// ============================================================================

// A1 — native data model (labels + relationships): the real alternative to the figure
CALL db.schema.visualization();

// A2 — APOC meta-graph (often cleaner). Requires the APOC plugin; if it is not
// installed you get "ProcedureNotFound" — just use A1 instead.
CALL apoc.meta.graph();


// ============================================================================
// B. CORPUS COMPOSITION  — exact numbers for the text / appendix (tables)
// ============================================================================

// B1 — node count per label
MATCH (n) RETURN labels(n)[0] AS label, count(*) AS n ORDER BY n DESC;

// B2 — relationship count per type
MATCH ()-[r]->() RETURN type(r) AS relationship, count(*) AS n ORDER BY n DESC;

// B3 — statute articles per code
MATCH (s:Statute) RETURN s.source AS codice, count(*) AS articoli ORDER BY articoli DESC;

// B4 — books per code
MATCH (c:Codice)-[:CONTAINS]->(l:Libro)
RETURN c.name AS codice, count(l) AS libri ORDER BY libri DESC;

// B5 — precedents by court level and by subject matter
MATCH (p:Precedent) RETURN p.court_level AS grado, count(*) AS n ORDER BY n DESC;
MATCH (p:Precedent) RETURN p.materia AS materia, count(*) AS n ORDER BY n DESC;

// B6 — internal-citation edges
MATCH (:Statute)-[:CITES]->(:Statute) RETURN count(*) AS cites_edges;


// ============================================================================
// C. SAMPLE SUBGRAPHS  — concrete, live examples (render as a graph: screenshot these)
// ============================================================================

// C1 — hierarchy Codice -> Libro <- Statute (a limited example)
MATCH path=(c:Codice)-[:CONTAINS]->(l:Libro)<-[:BELONGS_TO]-(s:Statute)
RETURN path LIMIT 25;

// C2 — citation graph between articles
MATCH path=(s:Statute)-[:CITES]->(t:Statute)
RETURN path LIMIT 40;

// C3 — one article with its citation neighbourhood. Robust version: pick any
// statute that actually has outgoing CITES edges (the exact `articolo` string
// format may vary, so we do not hard-code an article number here).
MATCH (s:Statute)-[:CITES]->()
WITH s LIMIT 1
MATCH path=(s)-[:CITES*1..2]-(n)
RETURN path LIMIT 50;

// C3b — if you want a specific article, inspect the format first, then match:
//   MATCH (s:Statute) WHERE s.articolo CONTAINS '1490' RETURN s.articolo, s.titolo LIMIT 5;

// C4 — most-cited articles (in-degree) — table
MATCH (s:Statute)<-[:CITES]-()
RETURN s.source AS codice, s.articolo AS articolo, s.titolo AS titolo, count(*) AS citato_da
ORDER BY citato_da DESC LIMIT 15;


// ============================================================================
// D. INDEXES  — the vector + full-text indexes described in Chapter 5
// ============================================================================

// D1 — all indexes (shows the vector index on Statute.embedding and the full-text ones)
SHOW INDEXES YIELD name, type, labelsOrTypes, properties;

// D2 — stored embedding dimension (confirms the Legal-BERT vector)
MATCH (s:Statute) WHERE s.embedding IS NOT NULL
RETURN size(s.embedding) AS embedding_dim LIMIT 1;

// D3 — a sample Statute node, non-vector properties
MATCH (s:Statute)
RETURN s.statute_id, s.source, s.libro, s.articolo, s.titolo, left(s.testo, 200) AS testo_preview
LIMIT 3;
