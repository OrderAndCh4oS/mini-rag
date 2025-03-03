import inspect

from app.definitions import EVALUATION_DATA_SET
from app.llm import get_completion
from app.main import query
from app.utilities import get_json


def accuracy(query, response, source):
    prompt = inspect.cleandoc(f"""
        Does the <response> accurately answer the <query> in relation to the <source>?
        
        Response parameters:
        You must answer with either "yes" or "no"
        There must be no other content in your response
        
        <query>
        {query}
        </query>
        
        <response>
        {response}
        </response>
        
        <source>
        {source}
        </source>
    """)

    responses = []
    for _ in range(3):
        response = get_completion(prompt)
        if response not in ['yes', 'no']:
            print("failed to respond with yes or no")
            continue
        responses.append(1 if response == 'yes' else 0)

    return 1 if sum(responses) / len(responses) > 0.5 else 0


if __name__ == '__main__':
    data_set = get_json(EVALUATION_DATA_SET)
    results = []

    for row in data_set:
        # result = accuracy(row["query"], row["response"], row["pull_quote"])
        response = query(row["query"])
        result = accuracy(row["query"], response, row["pull_quote"])
        results.append(result)

    score = sum(results) / len(results)
    print(score) # 1 for data set
