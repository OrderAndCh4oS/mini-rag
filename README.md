# Mini RAG

This project ingests text documents, creates smart excerpts with concise summaries, and generates vector embeddings using the OpenAI API. The embeddings are stored in a local vector database to enable fast semantic querying.

## Overview

- **Document Ingestion:** Reads and hashes text files to ensure unique document identification.
- **Smart Excerpts:** Extracts text snippets and produces brief contextual summaries.
- **Embedding Generation:** Uses the OpenAI API to create vector embeddings from the excerpts.
- **Caching & Deduplication:** Avoids redundant processing by caching embeddings and tracking document changes.
- **Semantic Querying:** Retrieves relevant document content based on semantic similarity.

Simply place your documents in the input directory, configure your OpenAI API key in the `.env` file, and run the main script to start the pipeline.
