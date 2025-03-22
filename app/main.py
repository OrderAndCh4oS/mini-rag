# Todo: extract entities (for low-level and high-level)
# Todo: extract entity relations (for low-level and high-level)
# Todo: create embeddings for descriptions of the relations
# Todo: dedupe entities and relations (against current data and cache)
# Todo: create caches for embeddings/entities/relations (do not create new embeddings if hashes of strings match)
# Todo: clean up any text, trim excess white space at end of text and between paragraphs
# Todo: store entities/relations in networkx (allow various kg store managers)
# Todo: store data in nano-vectordb (allow for various vector store managers)
# Todo: implement naive/local/global/hybrid queries
# Todo: implement delete doc
# Todo: explore kag patterns/strategies
import os
import time

import networkx as nx
import numpy as np

from nano_vectordb import NanoVectorDB

from app.llm import get_embedding, get_completion
from app.logger import logger, set_logger

from app.definitions import INPUT_DOCS_DIR, SOURCE_TO_DOC_ID_MAP, DOC_ID_TO_SOURCE_MAP, EMBEDDINGS_DB, \
    EXCERPT_DB, DOC_ID_TO_EXCERPT_IDS, KG_DB
from app.prompts import get_query_system_prompt, excerpt_summary_prompt, get_extract_entities_prompt

from app.utilities import get_json, remove_from_json, read_file, get_docs, make_hash, add_to_json, \
    create_file_if_not_exists, split_string_by_multi_markers, clean_str

dim = 1536
embeddings_db = NanoVectorDB(dim, storage_file=EMBEDDINGS_DB)


def remove_document_by_id(doc_id):
    doc_id_to_excerpt_ids = get_json(DOC_ID_TO_EXCERPT_IDS)
    doc_id_to_source_map = get_json(DOC_ID_TO_SOURCE_MAP)
    if doc_id in doc_id_to_source_map:
        source = doc_id_to_source_map[doc_id]
        remove_from_json(DOC_ID_TO_SOURCE_MAP, doc_id)
        remove_from_json(SOURCE_TO_DOC_ID_MAP, source)
    if doc_id in doc_id_to_excerpt_ids:
        excerpt_ids = doc_id_to_excerpt_ids[doc_id]
        for excerpt_id in excerpt_ids:
            remove_from_json(EXCERPT_DB, excerpt_id)
        remove_from_json(DOC_ID_TO_EXCERPT_IDS, doc_id)
        embeddings_db.delete(excerpt_ids)
        embeddings_db.save()


def import_documents():
    sources = get_docs(INPUT_DOCS_DIR)
    for source in sources:
        content = read_file(source)
        doc_id = make_hash(content, "doc_")

        source_to_doc_id_map = get_json(SOURCE_TO_DOC_ID_MAP)

        if source not in source_to_doc_id_map:
            logger.info(f"importing new document {source} with id {doc_id}")
            add_document_maps(source, content)
            embed_document(content, doc_id)
            extract_entities(content, doc_id)
        elif source_to_doc_id_map[source] != doc_id:
            logger.info(f"updating existing document {source} with id {doc_id}")
            old_doc_id = source_to_doc_id_map[source]
            remove_document_by_id(old_doc_id)
            add_document_maps(source, content)
            embed_document(content, doc_id)
            extract_entities(content, doc_id)
        else:
            logger.info(f"no changes, skipping document {source} with id {doc_id}")


def add_document_maps(source, content):
    doc_id = make_hash(content, "doc_")
    add_to_json(SOURCE_TO_DOC_ID_MAP, source, doc_id)
    add_to_json(DOC_ID_TO_SOURCE_MAP, doc_id, source)


def embed_document(content, doc_id):
    excerpts = get_excerpts(content)
    excerpt_ids = []
    for i, excerpt in enumerate(excerpts):
        excerpt_id = make_hash(excerpt, "excerpt_id_")
        excerpt_ids.append(excerpt_id)
        summary = get_excerpt_summary(content, excerpt)
        embedding_content = f"{excerpt}\n\n{summary}"
        embedding_result = get_embedding(embedding_content)
        vector = np.array(embedding_result, dtype=np.float32)
        embeddings_db.upsert(
            [{"__id__": excerpt_id, "__vector__": vector, "__doc_id__": doc_id, "__inserted_at__": time.time()}])
        add_to_json(EXCERPT_DB, excerpt_id, {
            "doc_id": doc_id,
            "doc_order_index": i,
            "excerpt": excerpt,
            "summary": summary,
            "indexed_at": time.time()
        })
        logger.info(f"created embedding for {excerpt_id} — {embedding_result}")

    embeddings_db.save()
    add_to_json(DOC_ID_TO_EXCERPT_IDS, doc_id, excerpt_ids)


def get_excerpts(content, n=2000, overlap=200):
    excerpts = []
    step = n - overlap
    for i in range(0, len(content), step):
        excerpts.append(content[i:i + n])
    return excerpts


def get_excerpt_summary(full_doc, excerpt):
    prompt = excerpt_summary_prompt(full_doc, excerpt)
    summary = get_completion(prompt)

    logger.info(f"Excerpt:\n{excerpt}\n\nSummary:\n{summary}")

    return summary


def extract_entities(content, doc_id):
    excerpts = get_excerpts(content)
    graph = None
    if os.path.exists(KG_DB):
        try:
            graph = nx.read_graphml(KG_DB)
            logger.info(f"Loaded existing graph from {KG_DB}")
        except Exception as e:
            logger.error(f"Error loading graph from {KG_DB}: {e}")
            graph = nx.DiGraph()
    else:
        graph = nx.DiGraph()
        logger.info("No existing graph found. Creating a new graph.")

    for excerpt in excerpts:
        result = get_completion(get_extract_entities_prompt(excerpt))
        logger.info(result)
        # --- Data Cleaning ---
        # Remove the trailing '<|COMPLETE|>' marker
        data_str = result.replace('<|COMPLETE|>', '').strip()

        # --- Split the Data into Records ---
        # Split the data using the "+|+" marker
        records = split_string_by_multi_markers(data_str, ['+|+'])

        # Remove surrounding parentheses from each record if present
        clean_records = []
        for record in records:
            if record.startswith('(') and record.endswith(')'):
                record = record[1:-1]
            clean_records.append(clean_str(record))
        records = clean_records

        for record in records:
            fields = split_string_by_multi_markers(record, ['<|>'])
            if not fields:
                continue
            record_type = fields[0].lower()
            logger.info(f"{record_type} {len(fields)}")
            if record_type == '"entity"':
                if len(fields) >= 4:
                    _, name, category, description = fields[:4]
                    logger.info(f"Entity - Name: {name}, Category: {category}, Description: {description}")
                    graph.add_node(name, category=category, description=description)
            elif record_type == '"relationship"':
                if len(fields) >= 6:
                    _, source, target, description, keywords, weight = fields[:6]
                    logger.info(f"Relationship - Source: {source}, Target: {target}, Description: {description}, Keywords: {keywords}, Weight: {weight}")
                    graph.add_edge(source, target, description=description, keywords=keywords, weight=weight)
            elif record_type == '"content_keywords"':
                if len(fields) >= 2:
                    logger.info(f"Content Keywords: {fields[1]}")
                    graph.graph['content_keywords'] = fields[1]

    nx.write_graphml(graph, KG_DB)
    # --- Verification: Print the Graph Contents ---
    print("Nodes:")
    for node, data in graph.nodes(data=True):
        print(f"{node}: {data}")

    print("\nEdges:")
    for src, tgt, data in graph.edges(data=True):
        print(f"{src} -> {tgt}: {data}")

    if 'content_keywords' in graph.graph:
        print("\nGraph Metadata:")
        print("content_keywords:", graph.graph['content_keywords'])


def query(text):
    logger.info(f"Received Query:\n{text}")
    embedding = get_embedding(text)
    embedding_array = np.array(embedding)
    results = embeddings_db.query(query=embedding_array, top_k=5, better_than_threshold=0.02)
    excerpt_db = get_json(EXCERPT_DB)
    system_prompt = get_query_system_prompt(excerpt_db, results)

    return get_completion(text, context=system_prompt.strip())


if __name__ == '__main__':
    set_logger("main.log")

    create_file_if_not_exists(SOURCE_TO_DOC_ID_MAP, "{}")
    create_file_if_not_exists(DOC_ID_TO_SOURCE_MAP, "{}")
    create_file_if_not_exists(DOC_ID_TO_EXCERPT_IDS, "{}")
    create_file_if_not_exists(EXCERPT_DB, "{}")

    import_documents()

    # print(query("what do rabbits eat?"))  # Should answer
    # print(query("what do cats eat?"))  # Should reject

    # remove_document_by_id("doc_4c3f8100da0b90c1a44c94e6b4ffa041")
