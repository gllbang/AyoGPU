"""Immutable content-addressed pgvector collections in a dedicated schema.

The namespace/corpus signature lives in the scheduler; stored document rows are
immutable and keyed by a hash of text plus vector. No pre-existing tables are
updated or dropped. Retrieval uses the same cosine metric as normalized BGE.
"""

import hashlib
import json
import os
import time

from .backends import first_output


class PgVectorWorker:
    def __init__(self, dsn=None, dimension=1024):
        import psycopg

        self.connection = psycopg.connect(
            dsn or os.environ["HERASYS_PG_DSN"], autocommit=True
        )
        self.dimension = int(dimension)
        if not 1 <= self.dimension <= 16000:
            raise ValueError("Invalid vector dimension")
        with self.connection.cursor() as cursor:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cursor.execute("CREATE SCHEMA IF NOT EXISTS herasys_reproduction")
            cursor.execute(
                f"CREATE TABLE IF NOT EXISTS herasys_reproduction.documents_{self.dimension} "
                f"(content_id text PRIMARY KEY, passage text NOT NULL, embedding vector({self.dimension}) NOT NULL)"
            )
        try:
            from pgvector.psycopg import register_vector
        except ImportError:
            self.binary_vectors = False
        else:
            register_vector(self.connection)
            self.binary_vectors = True

    def ready(self):
        return {
            "kind": "pgvector",
            "dimension": self.dimension,
            "binary_vectors": self.binary_vectors,
        }

    def execute_batch(self, requests):
        if any(r["config"].get("map_ports") for r in requests):
            from .pipeline import collect_mapped, expand_mapped

            expanded, layout = expand_mapped(requests)
            results = self.execute_batch(expanded) if expanded else []
            return collect_mapped(results, layout)
        import numpy as np

        start = time.perf_counter()
        outputs = []
        table = f"herasys_reproduction.documents_{self.dimension}"
        # The actor executes one physical batch atomically. Autocommitting every
        # INSERT turns a 16-query batch into up to hundreds of WAL fsyncs.
        with self.connection.transaction(), self.connection.cursor() as cursor:
            for request in requests:
                if request["op"] == "vectordb_ingestion":
                    values = list(request["inputs"].values())
                    documents = next(
                        v for v in values if len(v) and isinstance(v[0], str)
                    )
                    vectors = next(
                        v
                        for v in values
                        if len(v)
                        and (
                            isinstance(v[0], (list, tuple))
                            or getattr(v, "ndim", 0) == 2
                        )
                    )
                    if len(documents) != len(vectors) or any(
                        len(v) != self.dimension for v in vectors
                    ):
                        raise ValueError("Document/vector shape mismatch")
                    rows = []
                    for text, vector in zip(documents, vectors):
                        vector = np.asarray(vector, dtype="<f4")
                        if not np.isfinite(vector).all():
                            raise ValueError("Vectors must contain finite values")
                        encoded = text.encode("utf-8")
                        # Hash the exact float32 bytes stored by PostgreSQL;
                        # avoid formatting thousands of floats into JSON twice.
                        content_id = hashlib.sha256(
                            b"herasys-vector-v2\0"
                            + len(encoded).to_bytes(8, "big")
                            + encoded
                            + vector.tobytes()
                        ).hexdigest()
                        rows.append(
                            (
                                content_id,
                                text,
                                vector
                                if self.binary_vectors
                                else json.dumps(vector.tolist()),
                            )
                        )
                    value_placeholder = "%b" if self.binary_vectors else "%s::vector"
                    cursor.executemany(
                        f"INSERT INTO {table} VALUES (%s,%s,{value_placeholder}) ON CONFLICT DO NOTHING",
                        rows,
                    )
                    outputs.append(first_output(request, [row[0] for row in rows]))
                elif request["op"] == "vectordb_searching":
                    ids = request["inputs"]["index"]
                    query = request["inputs"]["query_vector"]
                    query = np.asarray(query, dtype=np.float32)
                    value_placeholder = "%b" if self.binary_vectors else "%s::vector"
                    # WITH ORDINALITY preserves input order for equal-score ties and duplicates.
                    cursor.execute(
                        f"SELECT d.passage FROM unnest(%s::text[]) WITH ORDINALITY AS i(id,pos) "
                        f"JOIN {table} d ON d.content_id=i.id "
                        f"ORDER BY d.embedding <=> {value_placeholder}, i.pos LIMIT %s",
                        (
                            ids,
                            query
                            if self.binary_vectors
                            else json.dumps(query.tolist()),
                            request["config"]["top_k"],
                        ),
                    )
                    outputs.append(
                        first_output(request, [row[0] for row in cursor.fetchall()])
                    )
                else:
                    raise ValueError(f"Unsupported pgvector operation {request['op']}")
        elapsed = time.perf_counter() - start
        return [{"output": output, "service_s": elapsed} for output in outputs]

    def release(self, outputs):
        pass
