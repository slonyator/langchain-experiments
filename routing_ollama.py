"""
routing_ollama.py

This module demonstrates the routing concept in LangChain using Ollama
"""

import os
from typing import Literal

from dotenv import find_dotenv, load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from loguru import logger
from pyprojroot import here

from ollama_functions import OllamaFunctions


class ChainNavigator(BaseModel):
    """Route a user query to the most relevant Chain to process a claim."""

    claim_type: Literal[
        "storm", "tap-water", "fire", "theft", "elementary", "motor"
    ] = Field(
        ...,
        title="Type of claim",
        description=(
            "Given a user document choose which category "
            "fits best to classify the given input document."
        ),
    )


class StormCauseInspector(BaseModel):
    """Inspect the cause of the storm damage."""

    cause: Literal["hail", "hurricane", "other"] = Field(
        ...,
        title="Storm damage",
        description="The description of the storm damage.",
    )


if __name__ == "__main__":
    _ = load_dotenv(find_dotenv())
    api_key = os.getenv("OPENAI_API_KEY")

    llm = OllamaFunctions(model="llama3", temperature=0, format="json")

    ollama_model = llm.with_structured_output(ChainNavigator)

    SYSTEM = """
    You are a dedicated assistant for insurance claims handling.
    You help process relevant information and classify claims to the
    relevant type. However, you do not have access to current events.
    Process the document and classify the claim type. 
    """

    chat_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM),
            ("human", "{document}"),
        ]
    )

    prompt = PromptTemplate.from_template(
        """
        You are a dedicated assistant for insurance claims handling.
        You help process relevant information and classify claims to the
        relevant type. However, you do not have access to current events.
        Process the document and classify the claim type. 
        Document: {document}
        """
    )

    openai_model = ChatOpenAI(
        api_key="ollama", model="llama3", base_url="http://localhost:11434/"
    )

    OPENAI_LLM = openai_model.with_structured_output(ChainNavigator)

    ollama_chain = prompt | ollama_model
    openai_chain = chat_prompt | OPENAI_LLM

    data = PyPDFLoader(str(here("./schadenmeldung.pdf"))).load()
    logger.info(data)

    try:
        open_ai_response = openai_chain.invoke(
            {"document": data[0].page_content}
        )
        logger.info(open_ai_response)
    except Exception as e:
        logger.error(e)

    try:
        response = ollama_chain.invoke({"document": data[0].page_content})
        logger.info(response)
    except Exception as e:
        logger.error(e)
