import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


def get_llm(temperature: float = 0.0) -> ChatOpenAI:
    """Single place that decides which LLM the whole project uses.

    Works with any OpenAI-compatible server: Hugging Face router, Ollama, OpenAI, etc.
    """
    return ChatOpenAI(
        base_url=os.environ.get("LLM_BASE_URL", "https://router.huggingface.co/v1"),
        api_key=os.environ.get("LLM_API_KEY") or os.environ.get("HF_TOKEN", "none"),
        model=os.environ.get("LLM_MODEL") or os.environ.get("HF_MODEL", "meta-llama/Llama-3.3-70B-Instruct"),
        temperature=temperature,
    )