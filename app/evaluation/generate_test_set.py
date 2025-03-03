# Todo: Extract snippets, generate query and responses. Write to json.
import inspect
import re

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score

# Note: Make sure to download the NLTK packages, if you haven't already
# nltk.download('punkt')
# nltk.download('punkt_tab')
# nltk.download('wordnet')

from app.definitions import INPUT_DOCS_DIR, EVALUATION_DATA_SET
from app.llm import get_completion
from app.utilities import get_docs, read_file, create_file_if_not_exists, write_json


def calculate_bleu(reference_text: str, candidate_text: str) -> float:
    """
    Calculates the BLEU score between a reference text and a candidate text.

    Args:
        reference_text (str): The reference text (e.g., the pull_quote).
        candidate_text (str): The candidate text (e.g., the response).

    Returns:
        float: The BLEU score.
    """
    smoothing = SmoothingFunction().method1

    reference_tokens = nltk.word_tokenize(reference_text.lower())
    candidate_tokens = nltk.word_tokenize(candidate_text.lower())

    score = sentence_bleu([reference_tokens], candidate_tokens, smoothing_function=smoothing)

    return score


def calculate_meteor(reference_text: str, candidate_text: str) -> float:
    """
    Calculates the METEOR score between a reference text and a candidate text.

    Args:
        reference_text (str): The reference text.
        candidate_text (str): The candidate text.

    Returns:
        float: The METEOR score.
    """
    reference_tokens = nltk.word_tokenize(reference_text.lower())
    candidate_tokens = nltk.word_tokenize(candidate_text.lower())

    score = meteor_score([reference_tokens], candidate_tokens)

    return score


if __name__ == '__main__':
    create_file_if_not_exists(EVALUATION_DATA_SET, "")
    docs = get_docs(INPUT_DOCS_DIR)
    data_set = []
    for doc in docs:
        content = read_file(doc)
        raw_pull_quotes = get_completion(inspect.cleandoc(f"""
            Extract up to ten pull quote facts from this document, you must pull the exact copy used in the document.
            
            Remove all markdown formatting
            Use no markdown formatting in your response
            Each pull quote should be on a single line
            
            <content>
            {content}
            </content>
        """))
        pull_quotes = [pull_quote.strip() for pull_quote in re.sub(r'\n+', '\n', raw_pull_quotes).strip().split('\n')]
        for pull_quote in pull_quotes:
            raw_query_response = get_completion(inspect.cleandoc(f"""
                Ask a question that relates to the pull quote and an expected response that you might expect from a rag that contains this information

                The question and response must directly target this pull quote.

                Use no markdown formatting in your response
                The question and response should each be on a single line: 
                 - the question must be on the first line
                 - the response must be on the second line
                respond with only the question and response

                <content>
                {content}
                </content>

                <pull-quote>
                {pull_quote}
                </pull-quote>
            """))
            clean_result = [line.strip() for line in re.sub(r'\n+', '\n', raw_query_response).strip().split('\n')]
            if len(clean_result) != 2:
                continue
            (query, response) = clean_result
            bleu = calculate_bleu(pull_quote, response)
            meteor = calculate_meteor(pull_quote, response)
            # Note: chuck out results that match too closely they're likely identical to the initial query
            if bleu < 0.85:
                data_set.append({
                    "pull_quote": pull_quote,
                    "query": query,
                    "response": response,
                    "bleu": bleu,
                    "meteor": meteor
                })

    write_json(EVALUATION_DATA_SET, data_set)
