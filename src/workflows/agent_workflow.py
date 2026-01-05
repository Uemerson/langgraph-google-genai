"""Workflow Agent using LangGraph and Google GenAI."""

from typing import Dict, TypedDict

from google import genai
from langgraph.config import get_stream_writer
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langsmith import get_current_run_tree
from langsmith.schemas import UsageMetadata


class GraphState(TypedDict):
    """Graph State."""

    prompt: str
    answer: str
    has_context: bool
    has_documents: bool


class AgentWorkFlow:
    """Agent Workflow to manage a LangGraph execution."""

    __client: genai.Client
    __model_id: str

    def __init__(self, client: genai.Client, model_id: str):
        self.__client = client
        self.__model_id = model_id
        self.__app = self.__build_graph()

    def __build_graph(self) -> CompiledStateGraph:
        """Graph construction."""

        workflow = StateGraph(GraphState)

        workflow.add_node("check_context", self.check_context_node)
        workflow.add_node("retrieve_rag", self.retrieve_rag_node)
        workflow.add_node("generate_answer", self.generate_answer_node)
        workflow.add_node("cannot_answer", self.cannot_answer_node)

        workflow.add_conditional_edges(
            "check_context",
            self.check_context_condition,
            {
                "has_context": "retrieve_rag",
                "no_context": "cannot_answer",
            },
        )

        workflow.add_conditional_edges(
            "retrieve_rag",
            self.retrieve_rag_condition,
            {
                "has_documents": "generate_answer",
                "no_documents": "cannot_answer",
            },
        )

        workflow.add_edge("generate_answer", END)
        workflow.set_entry_point("check_context")

        return workflow.compile()

    def check_context_node(self, state: GraphState) -> Dict:
        """Checks whether the prompt has enough context."""

        prompt = (
            "Does the following input contain a clear question "
            "or topic with enough context to answer?\n\n"
            f"Input: {state['prompt']}\n\n"
            "Answer YES or NO."
        )

        response = self.__client.models.generate_content(
            model=self.__model_id,
            contents=prompt,
        )

        input_tokens = response.usage_metadata.prompt_token_count
        output_tokens = response.usage_metadata.candidates_token_count

        self.__usage_metadata(input_tokens, output_tokens)

        return {"has_context": "YES" in response.text.upper()}

    def retrieve_rag_node(self, state: GraphState) -> Dict:
        """Mock RAG retrieval."""

        mock_knowledge_base = {
            "langgraph": (
                "LangGraph is a library for building stateful, "
                "multi-actor applications with LLMs."
            ),
            "python": (
                "Python is a high-level, interpreted programming "
                "language known for readability."
            ),
            "gemini": (
                "Gemini is Google's most capable AI model, "
                "built to be natively multimodal."
            ),
        }

        query = state["prompt"].lower()
        documents = [v for k, v in mock_knowledge_base.items() if k in query]

        return {"has_documents": bool(documents)}

    async def generate_answer_node(self, state: GraphState) -> Dict:
        """Generates the final answer."""

        writer = get_stream_writer()
        response_text = ""

        input_tokens = 0
        output_tokens = 0

        async for (
            chunk
        ) in await self.__client.aio.models.generate_content_stream(
            model=self.__model_id,
            contents=state["prompt"],
        ):
            if chunk.usage_metadata:
                input_tokens = chunk.usage_metadata.prompt_token_count
                output_tokens = chunk.usage_metadata.candidates_token_count

            if chunk.text:
                response_text += chunk.text
                writer({"answer": chunk.text})

        self.__usage_metadata(input_tokens, output_tokens)

        return {"answer": response_text}

    def __usage_metadata(self, input_tokens: int, output_tokens: int) -> None:
        """Send usage metadata to LangSmith."""

        run = get_current_run_tree()

        if run:
            run.add_metadata(
                metadata={
                    "ls_model_name": self.__model_id,
                    "ls_model_type": "llm",
                    "ls_provider": "google_genai",
                    "ls_run_depth": 0,
                    "ls_temperature": 0.7,
                    "usage_metadata": UsageMetadata(
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        total_tokens=input_tokens + output_tokens,
                    ),
                    "invocation_params": {
                        "_type": "google_gemini",
                        "candidate_count": 1,
                        "image_config": None,
                        "max_output_tokens": None,
                        "model": self.__model_id,
                        "stop": None,
                        "temperature": 0.7,
                        "top_k": None,
                        "top_p": None,
                    },
                    "options": {"streaming": True, "stop": None},
                },
            )

    def cannot_answer_node(self, _: GraphState) -> Dict:
        """Fallback node when the question cannot be answered."""

        return {
            "answer": (
                "I'm sorry, but I cannot provide an answer "
                "based on the given input."
            )
        }

    def check_context_condition(self, state: GraphState) -> str:
        """Determines the condition based on context check."""

        return "has_context" if state["has_context"] else "no_context"

    def retrieve_rag_condition(self, state: GraphState) -> str:
        """Determines the condition based on RAG retrieval."""

        return "has_documents" if state["has_documents"] else "no_documents"

    async def stream(self, prompt: str):
        """Streams workflow execution output."""

        async for stream_mode, chunk in self.__app.astream(
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
                    yield chunk["answer"]

    def save_graph(self, path: str = "graph.png") -> None:
        """Saves the workflow graph as a Mermaid PNG."""

        png = self.__app.get_graph().draw_mermaid_png()
        with open(path, "wb") as f:
            f.write(png)
