"""
Example of streaming tokens from Google Gemini model
using asynchronous programming.
"""

import asyncio
import os
from typing import TypedDict

from google import genai
from langgraph.config import get_stream_writer
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
MODEL_ID = "models/gemini-3-flash-preview"


class GraphState(TypedDict):
    """Graph State."""

    prompt: str
    answer: str
    has_context: bool
    has_documents: bool


def save_graph(app: CompiledStateGraph):
    """Save the graph as a mermaid PNG image."""

    png = app.get_graph().draw_mermaid_png()

    path = "graph.png"
    with open(path, "wb") as f:
        f.write(png)


async def stream_tokens(prompt: str):
    """Stream tokens from Gemini model."""

    async for chunk in await client.aio.models.generate_content_stream(
        model=MODEL_ID,
        contents=prompt,
    ):
        yield chunk.text


async def answer_tool(prompt: str):
    """Generate answer using the provided prompt."""

    writer = get_stream_writer()
    response = ""
    async for chunk in stream_tokens(prompt):
        response += chunk
        writer({"answer": chunk})

    return response


async def generate_answer_node(state: GraphState):
    """Generate answer based on the prompt in the state."""

    prompt = state["prompt"]

    function_response = await answer_tool(prompt)
    return {"answer": function_response}


def retrieve_rag_node(state: GraphState):
    """Simulates a RAG retrieval process."""

    mock_knowledge_base = {
        "langgraph": (
            "LangGraph is a library for building stateful, "
            "multi-actor applications with LLMs."
        ),
        "python": (
            "Python is a high-level, interpreted "
            "programming language known for readability."
        ),
        "gemini": (
            "Gemini is Google's most capable AI model, "
            "built to be natively multimodal."
        ),
    }

    query = state["prompt"].lower()
    documents = [v for k, v in mock_knowledge_base.items() if k in query]

    return {"has_documents": bool(documents)}


def check_context_node(state: GraphState):
    """Evaluates if the question provides sufficient context."""

    prompt = state["prompt"]

    prompt = (
        f"Does this input contain a clear topic or question "
        f"with enough context to answer? "
        f"Input: '{prompt}'. "
        f"Answer YES or NO."
    )

    response = client.models.generate_content(
        model=MODEL_ID,
        contents=prompt,
    )
    has_context = "YES" in response.text.upper()

    return {
        "has_context": has_context,
    }


def cannot_answer_node(_: GraphState):
    """Handles cases where the question cannot be answered."""

    return {
        "answer": (
            "I'm sorry, but I cannot provide "
            "an answer based on the given input."
        )
    }


async def main():
    """Main function to demonstrate token streaming."""

    workflow = StateGraph(GraphState)

    workflow.add_node("check_context_node", check_context_node)
    workflow.add_node("retrieve_rag_node", retrieve_rag_node)
    workflow.add_node("generate_answer_node", generate_answer_node)
    workflow.add_node("cannot_answer_node", cannot_answer_node)

    workflow.add_conditional_edges(
        "check_context_node",
        lambda s: (
            "retrieve_rag_node" if s["has_context"] else "cannot_answer_node"
        ),
    )

    # workflow.add_edge("retrieve_rag_node", "generate_answer_node")

    workflow.add_conditional_edges(
        "retrieve_rag_node",
        lambda s: (
            "generate_answer_node"
            if s["has_documents"]
            else "cannot_answer_node"
        ),
    )

    workflow.add_edge("generate_answer_node", END)

    workflow.set_entry_point("check_context_node")

    app = workflow.compile()

    save_graph(app)

    prompts = [
        "Hello, how are you? Please write a short poem about the sea.",
        "What is LangGraph?",
    ]
    for prompt in prompts:
        print(f"Prompt: {prompt}\n")
        async for stream_mode, chunk in app.astream(
            input={"prompt": prompt},
            stream_mode=[
                "values",
                "updates",
                "custom",
                "messages",
                "checkpoints",
                "tasks",
            ],
        ):
            if stream_mode in ["custom", "values"]:
                if chunk.get("answer"):
                    print(chunk.get("answer"), end="|", flush=True)
                    print()
                    print()


if __name__ == "__main__":
    asyncio.run(main())
