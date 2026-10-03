"""Paper workloads expressed with HeraSys DAG/Node, including real pipeline splits."""

import math

from HeraSys.dags.dag import DAG
from HeraSys.dags.node import Node
from HeraSys.dags.node_commons import NodeAnnotation, NodeIOSchema, NodeOps, NodeType
from HeraSys.engines.engine_types import EngineType


def primitive(name, op, engine, inputs, outputs, *, ms=1, **config):
    return Node(
        name,
        NodeType.COMPUTE,
        engine_type=engine,
        op_type=op,
        io_schema=NodeIOSchema(
            dict.fromkeys(inputs, object), dict.fromkeys(outputs, object)
        ),
        anno=NodeAnnotation.BATCHABLE,
        relative_ms=ms,
        config=config,
    )


def _generation(
    nodes,
    parent,
    context_port,
    question_port,
    *,
    max_tokens,
    quantum,
    model,
    session,
    expected_prompt_tokens=256,
):
    prefill = primitive(
        f"{session}_prefill",
        NodeOps.LLM_PREFILLING,
        EngineType.LLM,
        [question_port, context_port],
        [f"{session}_state_0"],
        ms=max(1, expected_prompt_tokens * 0.05),
        model=model,
        max_tokens=max_tokens,
        question_port=question_port,
        context_port=context_port,
        _hera_session=session,
        temperature=0.0,
        prompt_template="Context: {context}\nQuestion: {question}\nAnswer:",
    )
    parent >> prefill
    nodes.append(prefill)
    previous = prefill
    for step in range(math.ceil(max_tokens / quantum)):
        final = (step + 1) * quantum >= max_tokens
        output = (
            ("answer" if session == "answer" else "search_query")
            if final
            else f"{session}_state_{step + 1}"
        )
        decode = primitive(
            f"{session}_decode_{step}",
            NodeOps.LLM_PARTIAL_DECODING,
            EngineType.LLM,
            [previous.output_names[0]],
            [output],
            ms=2 * min(quantum, max_tokens - step * quantum),
            model=model,
            max_tokens=max_tokens,
            token_budget=min(quantum, max_tokens - step * quantum),
            final=final,
            _hera_session=session,
            _hera_deterministic=True,
        )
        previous >> decode
        nodes.append(decode)
        previous = decode
    return previous


def rag(
    question,
    documents,
    *,
    rerank=True,
    splits=2,
    max_tokens=32,
    quantum=16,
    embedding_model="BAAI/bge-large-en-v1.5",
    reranker_model="BAAI/bge-reranker-large",
    llm_model="meta-llama/Meta-Llama-3-8B",
    with_llm=True,
    dynamic_pipeline=False,
):
    if not documents or not isinstance(question, str):
        raise ValueError("RAG needs a question and a nonempty document collection")
    if min(splits, max_tokens, quantum) < 1:
        raise ValueError("Splits and token budgets must be positive")
    dag, nodes, values, indexes = (
        DAG(name="herasys_rag"),
        [],
        {"question": question},
        [],
    )
    splits = min(splits, len(documents))
    for lane in range(0 if dynamic_pipeline else splits):
        start, stop = (
            lane * len(documents) // splits,
            (lane + 1) * len(documents) // splits,
        )
        port, vector, index = f"documents_{lane}", f"vectors_{lane}", f"index_{lane}"
        values[port] = list(documents[start:stop])
        embed = primitive(
            f"embed_documents_{lane}",
            NodeOps.EMBEDDING,
            EngineType.EMBEDDER,
            [port],
            [vector],
            ms=max(1, sum(map(len, values[port])) * 0.004),
            model=embedding_model,
            role="document",
        )
        ingest = primitive(
            f"ingest_{lane}",
            NodeOps.VECTORDB_INGESTION,
            EngineType.VECTOR_DB,
            [port, vector],
            [index],
            ms=max(1, len(values[port]) * 0.1),
            _hera_immutable=True,
            _hera_reuse=True,
        )
        embed >> ingest
        nodes.extend([embed, ingest])
        indexes.append(ingest)
    if dynamic_pipeline:
        from .pipeline import pipeline, stage

        values["documents"] = list(documents)
        embed = primitive(
            "embed_documents",
            NodeOps.EMBEDDING,
            EngineType.EMBEDDER,
            ["documents"],
            ["vectors"],
            ms=max(1, sum(map(len, documents)) * 0.004),
            model=embedding_model,
            role="document",
        )
        ingest = primitive(
            "ingest",
            NodeOps.VECTORDB_INGESTION,
            EngineType.VECTOR_DB,
            ["documents", "vectors"],
            ["index_0"],
            ms=max(1, len(documents) * 0.1),
            _hera_immutable=True,
            _hera_reuse=True,
        )
        chain = pipeline(
            "index_documents",
            [stage(embed), stage(ingest)],
            inputs=["documents"],
            split_ports=["documents"],
            max_splits=splits,
        )
        nodes.append(chain)
        indexes.append(chain)
    gather = primitive(
        "gather_indexes",
        NodeOps.AGGREGATOR,
        EngineType.AGGREGATOR,
        [n.output_names[0] for n in indexes],
        ["index"],
        mode="indexes",
    )
    for index in indexes:
        index >> gather
    query = primitive(
        "embed_question",
        NodeOps.EMBEDDING,
        EngineType.EMBEDDER,
        ["question"],
        ["query_vector"],
        ms=max(1, len(question) * 0.004),
        model=embedding_model,
        role="query",
    )
    search = primitive(
        "retrieve",
        NodeOps.VECTORDB_SEARCHING,
        EngineType.VECTOR_DB,
        ["query_vector", "index"],
        ["contexts"],
        ms=max(1, len(documents) * 0.02),
        top_k=12,
    )
    gather >> search
    query >> search
    nodes.extend([gather, query, search])
    tail, context = search, "contexts"
    if rerank:
        tail = primitive(
            "rerank",
            NodeOps.RERANKING,
            EngineType.RERANKER,
            ["question", "contexts"],
            ["reranked_contexts"],
            ms=max(1, sum(len(d) for d in documents[:12]) * 0.005),
            model=reranker_model,
            top_k=8,
        )
        search >> tail
        nodes.append(tail)
        context = "reranked_contexts"
    if with_llm:
        _generation(
            nodes,
            tail,
            context,
            "question",
            max_tokens=max_tokens,
            quantum=quantum,
            model=llm_model,
            session="answer",
            expected_prompt_tokens=(
                len(question)
                + sum(sorted(map(len, documents), reverse=True)[: 8 if rerank else 12])
            )
            / 4
            + 16,
        )
    dag.register_nodes(*nodes)
    dag.metadata.update(
        workload="rag_rerank" if rerank else "rag",
        splits=splits,
        dynamic_pipeline=dynamic_pipeline,
    )
    return dag, values


def web_search(
    question,
    *,
    llm_model="meta-llama/Meta-Llama-3-8B",
    max_tokens=32,
    quantum=16,
    search_provider="fixture",
):
    dag, nodes = DAG(name="herasys_web_search"), []
    instruction = primitive(
        "rewrite_instruction",
        NodeOps.AGGREGATOR,
        EngineType.AGGREGATOR,
        ["instruction"],
        ["rewrite_context"],
        mode="identity",
    )
    nodes.append(instruction)
    rewrite = _generation(
        nodes,
        instruction,
        "rewrite_context",
        "question",
        max_tokens=16,
        quantum=quantum,
        model=llm_model,
        session="rewrite",
        expected_prompt_tokens=len(question) / 4 + 32,
    )
    search = primitive(
        "web_search",
        "web_search",
        EngineType.DUMMY,
        ["search_query"],
        ["web_context"],
        ms=100,
        provider=search_provider,
        top_k=5,
        _hera_reuse=True,
    )
    rewrite >> search
    nodes.append(search)
    _generation(
        nodes,
        search,
        "web_context",
        "question",
        max_tokens=max_tokens,
        quantum=quantum,
        model=llm_model,
        session="answer",
        # The fixture contains only a short prefix plus a <=16-token rewrite.
        # Applying a five-snippet Google estimate here overstates its prompt
        # by an order of magnitude and biases both profiling and priority.
        expected_prompt_tokens=len(question) / 4
        + (
            16 + 16 + len("Offline fixture for ") / 4
            if search_provider == "fixture"
            else 640
        ),
    )
    dag.register_nodes(*nodes)
    dag.metadata.update(workload="web_search", splits=1)
    return dag, {
        "question": question,
        "instruction": "Rewrite the question as a concise web search query.",
    }
