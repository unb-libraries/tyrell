"""Provides the builders for the core components of Tyrell."""
from langchain_community.llms import LlamaCpp
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from tyrell.llm import get_summarize_oneshot, get_summarize_chunk, get_summarize_final, get_compress_result

def build_summarizer_chains(llm: LlamaCpp) -> list[LLMChain]:
    """Builds the LLM chains from the LLM.

    Args:
        llm (LlamaCpp): The LLM to build the chains from.

    Returns:
        list[LLMChain]: The LLM chains.
    """
    chains = {}

    oneshot_prompt = PromptTemplate(
        input_variables=["date", "document"],
        template=get_summarize_oneshot(),
    )
    chains['summarize_oneshot'] = oneshot_prompt | llm

    chunk_prompt = PromptTemplate(
        input_variables=["date", "chunk", "prev_summary", "cur_chunk_no", "total_chunk_no"],
        template=get_summarize_chunk(),
    )
    chains['summarize_chunk'] = chunk_prompt | llm

    chunk_prompt = PromptTemplate(
        input_variables=["date", "original"],
        template=get_compress_result(),
    )
    chains['compress'] = chunk_prompt | llm

    final_prompt = PromptTemplate(
        input_variables=["date", "summary"],
        template=get_summarize_final(),
    )
    chains['summarize_final'] = final_prompt | llm

    return chains
