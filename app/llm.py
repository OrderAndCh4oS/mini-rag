import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_completion(query, model="gpt-4o-mini", context=""):
    system_message = [{"role": "developer", "content": context}] if context else []
    messages = [{"role": "user", "content": query}]

    response = client.chat.completions.create(
        model=model,
        store=True,
        messages=system_message + messages
    )

    return response.choices[0].message.content


def get_chat_completion(query, model="gpt-4o-mini", context="", chat_history=[]):
    system_message = [{"role": "developer", "content": context}] if context else []
    messages = chat_history + [{"role": "user", "content": query}]

    response = client.chat.completions.create(
        model=model,
        store=True,
        messages=system_message + messages
    )

    return messages + [{"role": "assistant", "content": response.choices[0].message.content}]


def get_embedding(content, model="text-embedding-3-small"):
    response = client.embeddings.create(
        model=model,
        input=content,
    )

    return response.data[0].embedding
