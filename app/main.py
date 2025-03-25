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
    EXCERPT_DB, DOC_ID_TO_EXCERPT_IDS, KG_DB, ENTITIES_DB, RELATIONSHIPS_DB, KG_SEP, TUPLE_SEP, REC_SEP, COMPLETE_TAG, \
    QUERY_CACHE, EMBEDDING_CACHE
from app.prompts import get_query_system_prompt, excerpt_summary_prompt, get_extract_entities_prompt, \
    get_high_low_level_keywords_prompt

from app.utilities import get_json, remove_from_json, read_file, get_docs, make_hash, add_to_json, \
    create_file_if_not_exists, split_string_by_multi_markers, clean_str, extract_json_from_text

dim = 1536
embeddings_db = NanoVectorDB(dim, storage_file=EMBEDDINGS_DB)
entities_db = NanoVectorDB(dim, storage_file=ENTITIES_DB)
relationships_db = NanoVectorDB(dim, storage_file=RELATIONSHIPS_DB)


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
        embeddings_db.upsert([
            {"__id__": excerpt_id, "__vector__": vector, "__doc_id__": doc_id, "__inserted_at__": time.time()}
        ])
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
    if os.path.exists(KG_DB):
        try:
            graph = nx.read_graphml(KG_DB)
            logger.info(f"Loaded existing graph from {KG_DB}")
        except Exception as e:
            logger.error(f"Error loading graph from {KG_DB}: {e}")
            graph = nx.Graph()
    else:
        graph = nx.Graph()
        logger.info("No existing graph found. Creating a new graph.")

    for excerpt in excerpts:
        excerpt_id = make_hash(excerpt, "excerpt_id_")
        result = get_completion(get_extract_entities_prompt(excerpt))
        logger.info(result)

        data_str = result.replace(COMPLETE_TAG, '').strip()

        records = split_string_by_multi_markers(data_str, [REC_SEP])

        clean_records = []
        for record in records:
            if record.startswith('(') and record.endswith(')'):
                record = record[1:-1]
            clean_records.append(clean_str(record))
        records = clean_records

        for record in records:
            fields = split_string_by_multi_markers(record, [TUPLE_SEP])
            print("FIELDS", fields)
            if not fields:
                continue
            fields = [field[1:-1] if field.startswith('"') and field.endswith('"') else field for field in fields]
            print("FIELDS TWO", fields)
            record_type = fields[0].lower()
            logger.info(f"{record_type} {len(fields)}")
            if record_type == 'entity':
                if len(fields) >= 4:
                    _, name, category, description = fields[:4]
                    logger.info(f"Entity - Name: {name}, Category: {category}, Description: {description}")
                    # Todo: implement upsert for node, if node exists combine data with separators
                    existing_node = graph.nodes.get(name)
                    if existing_node:
                        print("NODE", existing_node)
                        existing_categories = split_string_by_multi_markers(existing_node["category"], KG_SEP)
                        categories = KG_SEP.join(set(list(existing_categories) + [category]))
                        existing_descriptions = split_string_by_multi_markers(existing_node["description"], KG_SEP)
                        descriptions = KG_SEP.join(set(list(existing_descriptions) + [description]))
                        existing_excerpt_ids = split_string_by_multi_markers(existing_node["excerpt_id"], KG_SEP)
                        excerpt_ids = KG_SEP.join(set(list(existing_excerpt_ids) + [excerpt_id]))
                        graph.add_node(
                            name,
                            category=categories,
                            description=descriptions,
                            excerpt_id=excerpt_ids
                        )
                    else:
                        graph.add_node(name, category=category, description=description, excerpt_id=excerpt_id)
                    entity_id = make_hash(name, prefix="ent-")
                    embedding_content = f"{name} {description}"
                    embedding_result = get_embedding(embedding_content)
                    vector = np.array(embedding_result, dtype=np.float32)
                    entities_db.upsert([
                        {
                            "__id__": entity_id,
                            "__vector__": vector,
                            "__entity_name__": name,
                            "__inserted_at__": time.time(),
                        }
                    ])
            elif record_type == 'relationship':
                if len(fields) >= 6:
                    _, source, target, description, keywords, weight = fields[:6]
                    logger.info(
                        f"Relationship - Source: {source}, Target: {target}, Description: {description}, Keywords: {keywords}, Weight: {weight}"
                    )
                    # Todo: implement upsert for edge, if edge exists combine data with separators
                    graph.add_edge(source, target, description=description, keywords=keywords, weight=weight)
                    relationship_id = make_hash(f"{source}_{target}", prefix="ent-")
                    embedding_content = f"{keywords} {source} {target} {description}"
                    embedding_result = get_embedding(embedding_content)
                    vector = np.array(embedding_result, dtype=np.float32)
                    relationships_db.upsert([
                        {
                            "__id__": relationship_id,
                            "__vector__": vector,
                            "__source__": source,
                            "__target__": target,
                            "__inserted_at__": time.time(),
                        }
                    ])
            elif record_type == 'content_keywords':
                if len(fields) >= 2:
                    logger.info(f"Content Keywords: {fields[1]}")
                    graph.graph['content_keywords'] = fields[1]

    entities_db.save()
    relationships_db.save()

    nx.write_graphml(graph, KG_DB)
    # # --- Verification: Print the Graph Contents ---
    # print("Nodes:")
    # for node, data in graph.nodes(data=True):
    #     print(f"{node}: {data}")
    #
    # print("\nEdges:")
    # for src, tgt, data in graph.edges(data=True):
    #     print(f"{src} -> {tgt}: {data}")
    #
    # if 'content_keywords' in graph.graph:
    #     print("\nGraph Metadata:")
    #     print("content_keywords:", graph.graph['content_keywords'])


def query(text):
    logger.info(f"Received Query:\n{text}")
    embedding = get_embedding(text)
    embedding_array = np.array(embedding)
    results = embeddings_db.query(query=embedding_array, top_k=5, better_than_threshold=0.02)
    excerpt_db = get_json(EXCERPT_DB)
    system_prompt = get_query_system_prompt(excerpt_db, results)

    return get_completion(text, context=system_prompt.strip())


def kg_query(text):
    prompt = get_high_low_level_keywords_prompt(text)
    result = get_completion(prompt)
    keyword_data = extract_json_from_text(result)
    print(keyword_data)

    ll_keywords = keyword_data.get("low_level_keywords", [])
    print(ll_keywords)
    if len(ll_keywords):
        ll_embedding = get_embedding(ll_keywords)
        ll_embedding_array = np.array(ll_embedding)
        ll_results = entities_db.query(query=ll_embedding_array, top_k=5, better_than_threshold=0.02)
        print(ll_results)
    graph = nx.read_graphml(KG_DB)
    ll_data = [graph.nodes.get(r["__entity_name__"]) for r in ll_results]
    ll_degrees = [graph.degree(r["__entity_name__"]) for r in ll_results]
    print(ll_degrees)
    print(ll_data)
    ll_dataset = [
        {**n, "entity_name": k["__entity_name__"], "rank": d}
        for k, n, d in zip(ll_results, ll_data, ll_degrees)
    ]
    print(ll_dataset)
    sort_kg_data_set_by_relation_size(graph, ll_dataset)

    # # Todo: remove duplication of functionality here
    # hl_keywords = keyword_data.get("high_level_keywords", [])
    # print(hl_keywords)
    # if len(hl_keywords):
    #     hl_embedding = get_embedding(hl_keywords)
    #     hl_embedding_array = np.array(hl_embedding)
    #     hl_results = entities_db.query(query=hl_embedding_array, top_k=5, better_than_threshold=0.02)
    #     print(hl_results)
    # hl_data = [graph.nodes.get(r["__entity_name__"]) for r in hl_results]
    # hl_degrees = [graph.degree(r["__entity_name__"]) for r in hl_results]
    # print(hl_data)
    # print(hl_degrees)
    # hl_dataset = [
    #     {**n, "entity_name": k["__entity_name__"], "rank": d}
    #     for k, n, d in zip(hl_results, hl_data, hl_degrees)
    # ]
    # print(hl_dataset)

    # Todo: get text units
    # Todo: get relations


def sort_kg_data_set_by_relation_size(graph, kg_dataset):
    excerpt_ids = [row["excerpt_id"] for row in kg_dataset]
    print(excerpt_ids)
    # Todo: create set of excerpt ids
    # Todo: edges for each node in data set by entity name
    edges = [graph.edges(row["entity_name"]) for row in kg_dataset]
    print(edges)
    sibling_node_names = set()
    for edge in edges:
        if not edge:
            continue
        sibling_node_names.update([e[1] for e in edge])
    print(sibling_node_names)
    sibling_nodes = [graph.nodes.get(name) for name in list(sibling_node_names)]
    print(sibling_nodes)


if __name__ == '__main__':
    set_logger("main.log")

    create_file_if_not_exists(SOURCE_TO_DOC_ID_MAP, "{}")
    create_file_if_not_exists(DOC_ID_TO_SOURCE_MAP, "{}")
    create_file_if_not_exists(DOC_ID_TO_EXCERPT_IDS, "{}")
    create_file_if_not_exists(EXCERPT_DB, "{}")
    create_file_if_not_exists(QUERY_CACHE, "{}")
    create_file_if_not_exists(EMBEDDING_CACHE, "{}")

    import_documents()

    # print(query("what do rabbits eat?"))  # Should answer
    # print(query("what do cats eat?"))  # Should reject

    # kg_query("what do rabbits eat?")  # Should answer

    # remove_document_by_id("doc_4c3f8100da0b90c1a44c94e6b4ffa041")
