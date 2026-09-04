"""The application under evaluation: a RAG answerer.

Its prompt lives in prompts/answerer.md so that a pull request weakening it is a
visible one-file diff — which is exactly the demo the CI gate exists to reject.
"""
import os
import pathlib

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

PROMPT = pathlib.Path(__file__).resolve().parent.parent / "prompts" / "answerer.md"
TEMPLATE = """{system}

QUESTION: {question}

RETRIEVED CONTEXT:
{context}"""


def answer(question, context, model=None, api=None):
    api = api or Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    model = model or os.environ.get("ANSWERER_MODEL", "claude-sonnet-5")
    msg = api.messages.create(
        model=model, max_tokens=400, temperature=0,
        messages=[{"role": "user", "content": TEMPLATE.format(
            system=PROMPT.read_text().strip(), question=question, context=context)}],
    )
    return msg.content[0].text.strip()
