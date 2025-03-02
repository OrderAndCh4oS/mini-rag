# Mini RAG

This project ingests text documents, creates smart excerpts with concise summaries, and generates vector embeddings
using the OpenAI API. The embeddings are stored in a local vector database to enable fast semantic querying.

## Overview

- **Document Ingestion:** Reads and hashes text files to ensure unique document identification.
- **Smart Excerpts:** Extracts text snippets and produces brief contextual summaries.
- **Embedding Generation:** Uses the OpenAI API to create vector embeddings from the excerpts.
- **Caching & Deduplication:** Avoids redundant processing by caching embeddings and tracking document changes.
- **Semantic Querying:** Retrieves relevant document content based on semantic similarity.

Simply place your documents in the input directory, configure your OpenAI API key in the `.env` file, and run the main
script to start the pipeline.

## Document Ingestion & Hashing

- **File Discovery:**  
  The system scans the document input directory for text files (e.g., `.txt`, `.md`) using recursive file search.

- **Unique Identification:**  
  Each document's content is read and hashed (using MD5 with a prefix) to generate a unique document ID.  
  Mappings between file sources and document IDs are stored in JSON files to track changes.

### Smart Excerpts & Summaries

- **Splitting Content:**  
  Documents are divided into manageable excerpts (default size: 2000 characters).

- **Contextual Summaries:**  
  A one-sentence summary is generated for each excerpt to capture its relationship to the full document.

- **Combined Input for Embeddings:**  
  Each excerpt is paired with its summary to ensure that both literal text and context are represented.

### Embedding Generation & Caching

- **Vector Creation:**  
  The combined excerpt and summary are passed to the OpenAI API to generate a 1536-dimensional embedding.

- **Local Storage & Caching:**  
  Generated embeddings are stored in a local vector database (NanoVectorDB) along with metadata.  
  The system checks cached hashes to avoid redundant processing.

### Semantic Indexing for Fast Queries

- **Persistent Index:**  
  All mappings, excerpts, summaries, and embeddings are saved to reflect the current state of the documents.

- **Efficient Querying:**  
  When a query is made, its embedding is compared against the stored embeddings, and the system retrieves the most
  relevant excerpts and summaries to form a context-rich response.

## Document Update Process

When a document is ingested, its full content is hashed to create a unique document ID. This hash acts as a fingerprint
for the file. The update process works as follows:

1. **Hash Generation:**
    - The content of each file is read and passed through a hash function (using MD5 with a prefix) to create a unique
      document ID.

2. **Checking for Changes:**
    - The system maintains a mapping (in a JSON file) between source file paths and their document IDs.
    - When a file is processed in the `import_documents()` function, it checks if:
        - The file is new (not in the mapping), or
        - The existing hash does not match the new hash (indicating that the file has been updated).

3. **Handling Updates:**
    - If the document has been updated (i.e. the hash has changed), the system first **removes** the old version by
      calling the `remove_document_by_id(doc_id)` function.
    - After deletion, the updated file is re-ingested:
        - New mappings are created.
        - The file is split into excerpts.
        - Excerpts are summarized.
        - New vector embeddings are generated and stored.

This approach ensures that only the current content of a document is stored and queried, preventing outdated information
from persisting in the system.l

## Document Deletion Process

The deletion of a document, triggered by an update or a manual removal, is performed by the
`remove_document_by_id(doc_id)` function. Here’s how it works:

1. **Mapping Cleanup:**
    - The function retrieves JSON mappings:
        - **DOC_ID_TO_SOURCE_MAP:** Links document IDs to their source files.
        - **SOURCE_TO_DOC_ID_MAP:** Links source files back to document IDs.
        - **DOC_ID_TO_EXCERPT_IDS:** Keeps track of all excerpt IDs associated with a document.
    - It removes the document’s ID from these mappings, ensuring that the system no longer associates the file with any
      stored data.

2. **Excerpt and Embedding Removal:**
    - For each excerpt ID related to the document:
        - The corresponding entry is deleted from the excerpt database (`EXCERPT_DB`).
        - The associated vector embeddings are deleted from the local vector database (using `embeddings_db.delete()`).
    - After removing these entries, the vector database is saved to persist the changes.

This deletion process guarantees that all traces of the old document—including its excerpts, summaries, and
embeddings—are completely removed from the system.
