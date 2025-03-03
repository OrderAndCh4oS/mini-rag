import os
from pipes import SOURCE

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
LOG_DIR = os.path.join(ROOT_DIR, "logs")
INPUT_DOCS_DIR = os.path.join(ROOT_DIR, "input_docs")

SOURCE_TO_DOC_ID_MAP = os.path.join(DATA_DIR, "source_to_doc_id_map.json")
DOC_ID_TO_SOURCE_MAP = os.path.join(DATA_DIR, "doc_id_to_source_map.json")
DOC_ID_TO_EXCERPT_IDS = os.path.join(DATA_DIR, "doc_id_to_excerpt_ids.json")
EVALUATION_DATA_SET = os.path.join(DATA_DIR, "evaluation_data_set.json")
EXCERPT_DB = os.path.join(DATA_DIR, "excerpt_db.json")
EMBEDDINGS_DB = os.path.join(DATA_DIR, "embeddings_db.json")
