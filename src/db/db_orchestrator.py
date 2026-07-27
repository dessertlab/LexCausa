#!/usr/bin/env python3
"""
LexCausa Database Orchestrator.

Performs all necessary operations in order to initialize the database:
1. Verify Neo4j connection
2. Clean database (optional)
3. Create schema (indexes, constraints, graph structure)
4. Load statutes (Penal + Civil + Administrative Code) with embeddings
5. Load precedents (itacasehold) without embeddings

Usage:
    python db_orchestrator.py                    # Full setup
    python db_orchestrator.py --clean            # Cleans and reinitializes everything
    python db_orchestrator.py --check            # Only checks DB status
    python db_orchestrator.py --statutes         # Reloads only statutes
    python db_orchestrator.py --precedents       # Reloads only precedents
    python db_orchestrator.py --clean --statutes # Cleans and loads only statutes
"""

import argparse
import ast
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent))  # src/db
sys.path.insert(0, str(Path(__file__).parent.parent))  # src

from dotenv import load_dotenv  # noqa: E402
from neo4j import GraphDatabase  # noqa: E402

# Load environment
project_root = Path(__file__).parent.parent.parent
load_dotenv(project_root / ".env")

URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
USER = os.getenv("NEO4J_USER", "neo4j")
PWD = os.getenv("NEO4J_PASSWORD", "password")

EMBEDDING_DIM = 768
BATCH_SIZE = 100


class DatabaseOrchestrator:
    """Manages the complete initialization of the Neo4j database."""

    def __init__(self):
        self.driver: Optional[GraphDatabase.driver] = None
        self._connect()

    def _connect(self) -> bool:
        """Connect to Neo4j."""
        try:
            self.driver = GraphDatabase.driver(URI, auth=(USER, PWD))
            self.driver.verify_connectivity()
            print(f"✅ Connected to Neo4j: {URI}")
            return True
        except Exception as e:
            print(f"❌ Neo4j connection error: {e}")
            return False

    def close(self):
        """Closes the connection."""
        if self.driver:
            self.driver.close()

    # =========================================================================
    # STEP 1: VERIFICA E PULIZIA
    # =========================================================================

    def check_status(self) -> dict:
        """Checks the current status of the database."""
        status = {
            "connected": False,
            "nodes": {},
            "indexes": [],
            "constraints": [],
        }

        if not self.driver:
            return status

        status["connected"] = True

        with self.driver.session() as session:
            # Conta nodi per label
            for label in ["Statute", "Precedent", "Libro", "Codice"]:
                result = session.run(f"MATCH (n:{label}) RETURN count(n) as count")
                status["nodes"][label] = result.single()["count"]

            # Lista indici
            result = session.run("SHOW INDEXES YIELD name, type, state")
            for record in result:
                status["indexes"].append(
                    {
                        "name": record["name"],
                        "type": record["type"],
                        "state": record["state"],
                    }
                )

            # Lista constraint
            result = session.run("SHOW CONSTRAINTS YIELD name, type")
            for record in result:
                status["constraints"].append(
                    {
                        "name": record["name"],
                        "type": record["type"],
                    }
                )

        return status

    def print_status(self):
        """Prints the database status."""
        status = self.check_status()

        print("\n" + "=" * 60)
        print("📊 DATABASE STATUS")
        print("=" * 60)

        if not status["connected"]:
            print("❌ Not connected to Neo4j")
            return

        print("\n📦 Nodes:")
        for label, count in status["nodes"].items():
            emoji = "✅" if count > 0 else "⚪"
            print(f"   {emoji} {label}: {count}")

        print("\n🔍 Indexes:")
        for idx in status["indexes"]:
            state_emoji = "✅" if idx["state"] == "ONLINE" else "⏳"
            print(f"   {state_emoji} {idx['name']} ({idx['type']}) - {idx['state']}")

        if not status["indexes"]:
            print("   ⚪ No indexes")

        print("\n🔒 Constraints:")
        for c in status["constraints"]:
            print(f"   ✅ {c['name']} ({c['type']})")

        if not status["constraints"]:
            print("   ⚪ No constraints")

        print("=" * 60)

    def clean_database(self):
        """Completely cleans the database."""
        print("\n🗑️ Cleaning database...")

        with self.driver.session() as session:
            # Drop tutti gli indici
            result = session.run("SHOW INDEXES YIELD name")
            for record in result:
                idx_name = record["name"]
                try:
                    session.run(f"DROP INDEX {idx_name} IF EXISTS")
                    print(f"   Dropped index: {idx_name}")
                except Exception:
                    pass

            # Drop tutti i constraint
            result = session.run("SHOW CONSTRAINTS YIELD name")
            for record in result:
                const_name = record["name"]
                try:
                    session.run(f"DROP CONSTRAINT {const_name} IF EXISTS")
                    print(f"   Dropped constraint: {const_name}")
                except Exception:
                    pass

            # Delete all nodes
            session.run("MATCH (n) DETACH DELETE n")
            print("   Deleted all nodes")

        print("✅ Database cleaned")

    # =========================================================================
    # STEP 2: CREAZIONE SCHEMA
    # =========================================================================

    def create_schema(self):
        """Creates indexes, constraints, and graph structure."""
        print("\n🏗️ Creating schema...")

        with self.driver.session() as session:
            # -----------------------------------------------------------------
            # CONSTRAINT
            # -----------------------------------------------------------------
            print("\n   📋 Creating constraints...")

            constraints = [
                ("statute_unique_id", "Statute", "statute_id"),
                ("codice_unique_name", "Codice", "name"),
                ("precedent_unique_id", "Precedent", "precedent_id"),
            ]

            for name, label, prop in constraints:
                try:
                    session.run(
                        f"""
                        CREATE CONSTRAINT {name} IF NOT EXISTS
                        FOR (n:{label})
                        REQUIRE n.{prop} IS UNIQUE
                    """
                    )
                    print(f"      ✅ {name}")
                except Exception as e:
                    print(f"      ⚠️ {name}: {e}")

            # Constraint composto per Libro (name + codice)
            try:
                session.run(
                    """
                    CREATE CONSTRAINT libro_unique_name_codice IF NOT EXISTS
                    FOR (n:Libro)
                    REQUIRE (n.name, n.codice) IS UNIQUE
                """
                )
                print("      ✅ libro_unique_name_codice")
            except Exception as e:
                print(f"      ⚠️ libro_unique_name_codice: {e}")

            # -----------------------------------------------------------------
            # VECTOR INDEXES
            # -----------------------------------------------------------------
            print("\n   🔍 Creating vector indexes...")

            vector_indexes = [
                ("statutes_idx", "Statute", "embedding"),
            ]

            # Ensure deprecated precedent vector index is removed.
            session.run("DROP INDEX precedents_idx IF EXISTS")
            print("      ✅ precedents_idx removed (precedents without embeddings)")

            for name, label, prop in vector_indexes:
                # Drop e ricrea per assicurare dimensione corretta
                session.run(f"DROP INDEX {name} IF EXISTS")
                session.run(
                    f"""
                    CREATE VECTOR INDEX {name} IF NOT EXISTS
                    FOR (n:{label}) ON (n.{prop})
                    OPTIONS {{
                        indexConfig: {{
                            `vector.dimensions`: {EMBEDDING_DIM},
                            `vector.similarity_function`: 'COSINE',
                            `vector.quantization.enabled`: true
                        }}
                    }}
                """
                )
                print(f"      ✅ {name} ({EMBEDDING_DIM} dim)")

            # -----------------------------------------------------------------
            # FULLTEXT INDEX
            # -----------------------------------------------------------------
            print("\n   📝 Creating fulltext indexes...")

            session.run(
                """
                CREATE FULLTEXT INDEX statutes_fulltext_idx IF NOT EXISTS
                FOR (n:Statute)
                ON EACH [n.titolo, n.testo, n.full_text]
            """
            )
            print("      ✅ statutes_fulltext_idx")

            session.run(
                """
                CREATE FULLTEXT INDEX precedents_fulltext_idx IF NOT EXISTS
                FOR (n:Precedent)
                ON EACH [n.title, n.summary]
            """
            )
            print("      ✅ precedents_fulltext_idx")

            session.run(
                """
                CREATE INDEX statute_source_articolo_idx IF NOT EXISTS
                FOR (n:Statute)
                ON (n.source, n.articolo)
            """
            )
            print("      ✅ statute_source_articolo_idx")

            # -----------------------------------------------------------------
            # STRUTTURA GRAFO (Codice -> Libro)
            # -----------------------------------------------------------------
            print("\n   🏛️ Creating graph structure...")

            # Codici
            session.run(
                """
                MERGE (c:Codice {name: 'Codice Penale'})
                SET c.description = 'Codice Penale Italiano'
            """
            )
            session.run(
                """
                MERGE (c:Codice {name: 'Codice Civile'})
                SET c.description = 'Codice Civile Italiano'
            """
            )
            session.run(
                """
                MERGE (c:Codice {name: 'Codice Amministrativo'})
                SET c.description = 'Legge 7 agosto 1990, n. 241'
            """
            )

            # Libri Codice Penale (prefisso CP per distinguerli)
            libri_penale = [
                ("CP Libro I", "Dei reati in generale"),
                ("CP Libro II", "Dei delitti in particolare"),
                ("CP Libro III", "Delle contravvenzioni in particolare"),
            ]
            for libro_name, libro_desc in libri_penale:
                session.run(
                    """
                    MERGE (l:Libro {name: $name, codice: 'codice_penale'})
                    SET l.description = $description
                    WITH l
                    MATCH (c:Codice {name: 'Codice Penale'})
                    MERGE (c)-[:CONTAINS]->(l)
                """,
                    name=libro_name,
                    description=libro_desc,
                )

            # Libri Codice Civile (prefisso CC per distinguerli)
            libri_civile = [
                ("CC Libro I", "Delle persone e della famiglia"),
                ("CC Libro II", "Delle successioni"),
                ("CC Libro III", "Della proprietà"),
                ("CC Libro IV", "Delle obbligazioni"),
                ("CC Libro V", "Del lavoro"),
                ("CC Libro VI", "Della tutela dei diritti"),
                ("CC Fuori range", "Articoli fuori dalla numerazione standard"),
            ]
            for libro_name, libro_desc in libri_civile:
                session.run(
                    """
                    MERGE (l:Libro {name: $name, codice: 'codice_civile'})
                    SET l.description = $description
                    WITH l
                    MATCH (c:Codice {name: 'Codice Civile'})
                    MERGE (c)-[:CONTAINS]->(l)
                """,
                    name=libro_name,
                    description=libro_desc,
                )

            print("      ✅ Penal Code (3 books)")
            print("      ✅ Civil Code (7 books)")
            print("      ✅ Administrative Code (no books)")

        print("\n✅ Schema created")

    # =========================================================================
    # STEP 3: CARICAMENTO STATUTI
    # =========================================================================

    def load_statutes(self):
        """Loads Penal, Civil, and Administrative Codes with embeddings."""
        from data_loader import (
            load_codice_amministrativo_with_embeddings,
            load_codice_civile_with_embeddings,
            load_codice_penale_with_embeddings,
        )

        print("\n📖 Loading statutes...")

        # Replace mode: avoid duplicates on repeated --statutes runs.
        with self.driver.session() as session:
            session.run("MATCH (s:Statute) DETACH DELETE s")
        print("   🧹 Existing Statute nodes removed")

        # Codice Penale
        print("\n   📕 Penal Code...")
        df_penale, emb_penale = load_codice_penale_with_embeddings()
        self._ingest_statutes(df_penale, "codice_penale", emb_penale)

        # Codice Civile
        print("\n   📗 Civil Code...")
        df_civile, emb_civile = load_codice_civile_with_embeddings()
        self._ingest_statutes(df_civile, "codice_civile", emb_civile)

        # Codice Amministrativo
        print("\n   📘 Administrative Code (L. 241/1990)...")
        df_amm, emb_amm = load_codice_amministrativo_with_embeddings()
        self._ingest_statutes(df_amm, "codice_amministrativo", emb_amm)

        print("\n   🔗 Creating CITES relationships...")
        cites_count = self._create_cites_relationships()
        print(f"      ✅ CITES relationships created: {cites_count}")

        print("\n✅ Statutes loaded")

    @staticmethod
    def _parse_list_field(value) -> list[str]:
        """
        Parse a list-like CSV field.

        Accepted formats:
        - python list string: "['40', '41']"
        - json list string: ["40", "41"]
        - single string token
        """
        if value is None:
            return []
        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()]

        raw = str(value).strip()
        if not raw or raw.lower() == "nan":
            return []

        parsed = None
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(raw)
                break
            except Exception:
                continue

        if isinstance(parsed, list):
            return [str(v).strip() for v in parsed if str(v).strip()]

        # Fallback: comma-separated or plain string.
        if "," in raw:
            return [x.strip() for x in raw.split(",") if x.strip()]
        return [raw]

    @staticmethod
    def _normalize_internal_reference(ref: str, source: str) -> str:
        """Normalize internal article references to match stored `Statute.articolo`."""
        if not ref:
            return ""
        token = ref.strip().lower()
        token = token.replace("‑", "-").replace("–", "-").replace("—", "-")
        token = re.sub(r"^art(?:t|icolo)?\.?\s*", "", token)
        m = re.search(r"(\d+(?:-[a-z]+)*(?:\.\d+)?)", token)
        if not m:
            return ""
        normalized = m.group(1)
        # Codice civile statute ids are stored as article_id (e.g. art2043).
        if source == "codice_civile" and not normalized.startswith("art"):
            normalized = f"art{normalized}"
        return normalized

    def _extract_row_references(self, row, source: str) -> tuple[list[str], list[str]]:
        """
        Extract and normalize internal and external references from heterogeneous CSV schemas.
        """
        internal_raw = (
            row.get("reference")
            or row.get("references")
            or row.get("article_references")
            or row.get("article_reference")
            or "[]"
        )
        external_raw = (
            row.get("external_reference") or row.get("external_references") or "[]"
        )

        internal_refs = [
            self._normalize_internal_reference(ref, source)
            for ref in self._parse_list_field(internal_raw)
        ]
        internal_refs = sorted({r for r in internal_refs if r})

        external_refs = sorted(
            {ref for ref in self._parse_list_field(external_raw) if ref}
        )
        return internal_refs, external_refs

    def _create_cites_relationships(self) -> int:
        """Create intra-source CITES relationships from Statute.reference arrays."""
        with self.driver.session() as session:
            session.run("MATCH (:Statute)-[r:CITES]->(:Statute) DELETE r")

            result = session.run(
                """
                MATCH (s:Statute)
                UNWIND coalesce(s.reference, []) AS ref
                WITH s, toLower(trim(toString(ref))) AS ref
                WHERE ref <> ''
                MATCH (t:Statute)
                WHERE t.source = s.source
                  AND toLower(t.articolo) = ref
                  AND s.statute_id <> t.statute_id
                MERGE (s)-[r:CITES {kind: 'internal_reference'}]->(t)
                RETURN count(r) AS rel_count
            """
            )
            record = result.single()
            return int(record["rel_count"] if record else 0)

    def _ingest_statutes(self, df, source: str, embeddings):
        """Inserts statutes into the database."""
        if embeddings is None:
            print(f"      ⚠️ No embeddings for {source}")
            return

        if len(embeddings) != len(df):
            print(
                f"      ⚠️ Mismatch: {len(df)} articles vs {len(embeddings)} embeddings"
            )
            return

        with self.driver.session() as session:
            for i in range(0, len(df), BATCH_SIZE):
                batch = df.iloc[i : i + BATCH_SIZE]
                batch_emb = embeddings[i : i + BATCH_SIZE]

                records = []
                for idx, (_, row) in enumerate(batch.iterrows()):
                    global_idx = i + idx

                    # Normalizza campi
                    if source == "codice_civile":
                        articolo = str(row.get("article_id", ""))
                        titolo = str(row.get("article_title", ""))
                        testo = str(row.get("article_text", ""))
                    elif source == "codice_amministrativo":
                        articolo = str(row.get("numero", ""))
                        titolo = str(row.get("titolo", ""))
                        testo = str(row.get("contenuto", ""))
                    else:
                        articolo = str(row.get("articolo", ""))
                        titolo = str(row.get("titolo", ""))
                        testo = str(row.get("testo", ""))

                    def _clean_text(value) -> str:
                        if value is None:
                            return ""
                        as_str = str(value).strip()
                        return "" if as_str.lower() == "nan" else as_str

                    articolo = _clean_text(articolo)
                    titolo = _clean_text(titolo)
                    testo = _clean_text(testo)

                    libro = _clean_text(row.get("libro", ""))
                    if not libro:
                        libro = _clean_text(row.get("libro_codice_penale", ""))
                    if not libro:
                        libro = _clean_text(row.get("libro_codice_civile", ""))

                    reference, external_reference = self._extract_row_references(
                        row, source
                    )

                    records.append(
                        {
                            "statute_id": f"{source}_{global_idx}_{articolo}",
                            "articolo": articolo,
                            "titolo": titolo,
                            "testo": testo,
                            "libro": libro,
                            "source": source,
                            "full_text": f"Art. {articolo} - {titolo}: {testo}",
                            "embedding": batch_emb[idx].tolist(),
                            "reference": reference,
                            "external_reference": external_reference,
                        }
                    )

                session.run(
                    """
                    UNWIND $records AS record
                    CREATE (s:Statute {
                        statute_id: record.statute_id,
                        articolo: record.articolo,
                        titolo: record.titolo,
                        testo: record.testo,
                        libro: record.libro,
                        source: record.source,
                        full_text: record.full_text,
                        embedding: record.embedding,
                        reference: record.reference,
                        external_reference: record.external_reference
                    })
                    WITH s, record
                    FOREACH (_ IN CASE WHEN record.libro <> '' THEN [1] ELSE [] END |
                        MERGE (l:Libro {name: record.libro, codice: record.source})
                        MERGE (s)-[:BELONGS_TO]->(l)
                    )
                """,
                    records=records,
                )

                inserted = min(i + BATCH_SIZE, len(df))
                print(f"      Inserted: {inserted}/{len(df)}", end="\r")

            print(f"      ✅ {len(df)} articles with embeddings")

    # =========================================================================
    # STEP 4: CARICAMENTO PRECEDENTI (itacasehold)
    # =========================================================================

    def load_precedents(self):
        """Loads itacasehold precedents without embeddings (fulltext only)."""
        from data_loader import load_itacasehold_metadata

        print("\n⚖️ Loading precedents (itacasehold, no-embedding)...")

        try:
            metadata = load_itacasehold_metadata()
        except FileNotFoundError as e:
            print(f"      ⚠️ {e}")
            return

        with self.driver.session() as session:
            for i in range(0, len(metadata), BATCH_SIZE):
                batch_meta = metadata[i : i + BATCH_SIZE]

                records = []
                for idx, meta in enumerate(batch_meta):
                    global_idx = i + idx
                    records.append(
                        {
                            "precedent_id": f"prec_{global_idx}",
                            "title": str(meta.get("title", ""))[:500],
                            "summary": str(meta.get("summary", ""))[:5000],
                            "url": str(meta.get("url", "")),
                            "materia": str(meta.get("materia", ""))[:200],
                            "year": meta.get("year"),
                            "court": str(meta.get("court", ""))[:200],
                            "court_level": str(meta.get("court_level", ""))[:80],
                            "source": "itacasehold",
                        }
                    )

                session.run(
                    """
                    UNWIND $records AS record
                    MERGE (p:Precedent {precedent_id: record.precedent_id})
                    SET p.title = record.title,
                        p.summary = record.summary,
                        p.url = record.url,
                        p.materia = record.materia,
                        p.year = record.year,
                        p.court = record.court,
                        p.court_level = record.court_level,
                        p.source = record.source
                """,
                    records=records,
                )

                inserted = min(i + BATCH_SIZE, len(metadata))
                print(f"      Inserted: {inserted}/{len(metadata)}", end="\r")

            print(f"      ✅ {len(metadata)} precedents (without embeddings)")

        print("\n✅ Precedents loaded")

    # =========================================================================
    # STEP 5: VERIFICA INDICI
    # =========================================================================

    def wait_for_indexes(self, timeout: int = 120):
        """Waits for all indexes to be ONLINE."""
        print("\n⏳ Waiting for indexes to be online...")

        start = time.time()
        while time.time() - start < timeout:
            with self.driver.session() as session:
                result = session.run(
                    """
                    SHOW INDEXES YIELD name, state
                    WHERE state <> 'ONLINE'
                    RETURN count(*) as pending
                """
                )
                pending = result.single()["pending"]

                if pending == 0:
                    print("   ✅ All indexes are ONLINE")
                    return True

                print(f"   ⏳ {pending} indexes pending...", end="\r")
                time.sleep(2)

        print("   ⚠️ Timeout - some indexes may not be ready")
        return False

    # =========================================================================
    # ORCHESTRAZIONE PRINCIPALE
    # =========================================================================

    def run_full_setup(
        self,
        clean: bool = False,
        load_statutes: bool = True,
        load_precedents: bool = True,
    ):
        """Performs the complete database setup.

        Args:
            clean: If True, cleans the database before initializing
            load_statutes: If True, loads the statutes (Penal + Civil Code)
            load_precedents: If True, loads the precedents (itacasehold)
        """
        print("\n" + "=" * 60)
        print("🚀 LexCausa Database Orchestrator")
        print("=" * 60)

        start_time = time.time()

        # Step 0: Pulizia (opzionale)
        if clean:
            self.clean_database()

        # Step 1: Stato iniziale
        print("\n📊 Initial status:")
        self.print_status()

        # Step 2: Schema (sempre necessario)
        self.create_schema()

        # Step 3: Statuti (opzionale)
        if load_statutes:
            self.load_statutes()
        else:
            print("\n📖 Loading statutes: SKIP")

        # Step 4: Precedenti (opzionale)
        if load_precedents:
            self.load_precedents()
        else:
            print("\n⚖️ Loading precedents: SKIP")

        # Step 5: Attendi indici
        self.wait_for_indexes()

        # Stato finale
        elapsed = time.time() - start_time
        print("\n" + "=" * 60)
        print("📊 Final status:")
        self.print_status()

        print(f"\n⏱️ Total time: {elapsed:.1f}s")
        print("✅ Setup complete!")


def main():
    parser = argparse.ArgumentParser(
        description="LexCausa Database Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python db_orchestrator.py                    # Full setup
  python db_orchestrator.py --clean            # Cleans and reinitializes everything
  python db_orchestrator.py --check            # Only checks DB status
  python db_orchestrator.py --statutes         # Reloads only statutes
  python db_orchestrator.py --precedents       # Reloads only precedents
  python db_orchestrator.py --clean --statutes # Cleans and loads only statutes
        """,
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Cleans the database before initializing",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only checks database status",
    )
    parser.add_argument(
        "--statutes",
        action="store_true",
        help="Loads only statutes (Penal + Civil Code)",
    )
    parser.add_argument(
        "--precedents",
        action="store_true",
        help="Loads only precedents (itacasehold)",
    )
    args = parser.parse_args()

    orchestrator = DatabaseOrchestrator()

    try:
        if args.check:
            orchestrator.print_status()
        else:
            # Se nessun flag specifico, carica tutto
            # Se almeno uno è specificato, carica solo quello
            if not args.statutes and not args.precedents:
                load_statutes = True
                load_precedents = True
            else:
                load_statutes = args.statutes
                load_precedents = args.precedents

            orchestrator.run_full_setup(
                clean=args.clean,
                load_statutes=load_statutes,
                load_precedents=load_precedents,
            )
    finally:
        orchestrator.close()


if __name__ == "__main__":
    main()
