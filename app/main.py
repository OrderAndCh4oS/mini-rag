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
import inspect
import time

import numpy as np
from textwrap import wrap

from nano_vectordb import NanoVectorDB

from app.llm import get_embedding, get_completion
from app.logger import logger, set_logger

from app.definitions import INPUT_DOCS_DIR, SOURCE_TO_DOC_ID_MAP, DOC_ID_TO_SOURCE_MAP, EMBEDDINGS_DB, \
    EXCERPT_DB, DOC_ID_TO_EXCERPT_IDS

from app.utilities import get_json, remove_from_json, read_file, get_docs, make_hash, add_to_json, \
    create_file_if_not_exists

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
        elif source_to_doc_id_map[source] != doc_id:
            logger.info(f"updating existing document {source} with id {doc_id}")
            old_doc_id = source_to_doc_id_map[source]
            remove_document_by_id(old_doc_id)
            add_document_maps(source, content)
            embed_document(content, doc_id)
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


def get_excerpts(content, n=2000):
    return wrap(content, n)


def get_excerpt_summary(content, excerpt):
    summary = get_completion(inspect.cleandoc(f"""
        Create a concise, short one sentence summary of how the <excerpt> relates to the broader context of the <full-document> and surrounding content.  

        <full-document>
        {content}
        </full-document>
        
        <excerpt>
        {excerpt}
        </excerpt>
        
        Respond with the summary only.
    """))

    logger.info(f"excerpt {excerpt}, summary {summary}")

    return summary


def query(text):
    embedding = get_embedding(text)
    embedding_array = np.array(embedding)
    results = embeddings_db.query(query=embedding_array, top_k=5, better_than_threshold=0.02)
    excerpt_db = get_json(EXCERPT_DB)
    system_prompt = inspect.cleandoc("""
        You are a professional assistant responsible for answering questions related the to following information in the sources. Each source will contain an excerpt which has been pulled from accurate source material and a summary which outlines the broader context from which the excerpt was taken.

        The excerpts are the source of truth, However, the summaries will contain additional contextual information that may have been lost when extracting the excerpt from the real-life document. Favour information from excerpts where possible.
        
        If you don't know the answer, just say so. Do not make anything up or include information where the supporting evidence is not provided.
        
        Response parameters:
         - Answers must be in en_GB english
         - Use markdown formatting with appropriate section headings
         - Each section should focus on one main point or aspect of the answer
         - Use clear and descriptive section titles that reflect the content
    """)
    system_prompt += "\n\n"
    for i, result in enumerate(results):
        excerpt_data = excerpt_db[result["__id__"]]
        system_prompt += inspect.cleandoc(f"""
            # Source {i + 1}.
    
            ## Excerpt
            
            {excerpt_data["excerpt"]}
            
            ## Summary
            
            {excerpt_data["summary"]} 
        """)
        system_prompt += "\n\n"

    return get_completion(text, context=system_prompt.strip())


if __name__ == '__main__':
    # create_file_if_not_exists(SOURCE_TO_DOC_ID_MAP, "{}")
    # create_file_if_not_exists(DOC_ID_TO_SOURCE_MAP, "{}")
    # create_file_if_not_exists(DOC_ID_TO_EXCERPT_IDS, "{}")
    # create_file_if_not_exists(EXCERPT_DB, "{}")
    # set_logger("main.log")
    #
    # import_documents()

    print(query("what do rabbits eat?"))  # Should answer
    print(query("what do cats eat?"))  # Should reject

    # remove_document_by_id("doc_4c3f8100da0b90c1a44c94e6b4ffa041")
