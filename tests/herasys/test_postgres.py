import os

import pytest

from HeraSys.backends import memory_index, memory_search
from HeraSys.postgres import PgVectorWorker


@pytest.mark.skipif(
    not os.environ.get("HERASYS_PG_DSN"), reason="opt-in local pgvector integration"
)
def test_pgvector_matches_memory_and_preserves_duplicate_documents():
    import numpy as np

    worker = PgVectorWorker(dimension=3)
    ingestion = {
        "op": "vectordb_ingestion",
        "inputs": {
            "documents": ["first", "second", "first"],
            "vectors": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
        },
        "outputs": ["index"],
        "config": {},
    }
    pg_index = worker.execute_batch([ingestion])[0]["output"]["index"]
    assert worker.execute_batch([ingestion])[0]["output"]["index"] == pg_index
    search = {
        "op": "vectordb_searching",
        "inputs": {"query_vector": [1.0, 0.0, 0.0], "index": pg_index},
        "config": {"top_k": 3},
        "outputs": ["contexts"],
    }
    actual = worker.execute_batch([search])[0]["output"]
    search["inputs"]["index"] = memory_index(ingestion)["index"]
    assert actual == memory_search(search) == {"contexts": ["first", "first", "second"]}
    # Array transport and PostgreSQL binary encoding preserve both IDs and ties.
    ingestion["inputs"]["vectors"] = np.asarray(
        ingestion["inputs"]["vectors"], dtype=np.float32
    )
    assert worker.execute_batch([ingestion])[0]["output"]["index"] == pg_index
    search["inputs"] = {
        "index": pg_index,
        "query_vector": np.asarray([1, 0, 0], dtype=np.float32),
    }
    assert worker.execute_batch([search])[0]["output"] == actual
    worker.binary_vectors = False
    assert worker.execute_batch([ingestion])[0]["output"]["index"] == pg_index
    assert worker.execute_batch([search])[0]["output"] == actual
    search["inputs"]["index"] = memory_index(ingestion)["index"]
    assert memory_search(search) == actual


@pytest.mark.skipif(
    not os.environ.get("HERASYS_PG_DSN"), reason="opt-in pgvector integration"
)
def test_ingestion_batch_rolls_back_on_invalid_vector():
    import uuid

    worker = PgVectorWorker(dimension=3)
    passage = "atomic ingestion " + uuid.uuid4().hex
    good = {
        "op": "vectordb_ingestion",
        "inputs": {"documents": [passage], "vectors": [[1.0, 0.0, 0.0]]},
        "outputs": ["index"],
        "config": {},
    }
    bad = {**good, "inputs": {"documents": ["invalid"], "vectors": [[1.0, 0.0]]}}
    with pytest.raises(ValueError, match="shape mismatch"):
        worker.execute_batch([good, bad])
    count = worker.connection.execute(
        "SELECT count(*) FROM herasys_reproduction.documents_3 WHERE passage=%s",
        (passage,),
    ).fetchone()[0]
    assert count == 0
