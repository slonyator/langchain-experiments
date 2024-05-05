"""
routers.py

This module demonstrates the routing concept in LangChain.
It includes two examples: one for chains and another for agents.

"""
import os
from typing import Literal

from dotenv import find_dotenv, load_dotenv
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableBranch
from loguru import logger
from pydantic import BaseModel, Field
from pyprojroot import here
from langchain_experimental.llms.ollama_functions import OllamaFunctions


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
    llm = OllamaFunctions(model="llama3", temperature=0,)

    navigator_llm = llm.with_structured_output(ChainNavigator)

    SYSTEM = """
        You are a dedicated assistant for insurance claims handling. 
        You help process relevant information and classify claims to the 
        relevant type. However, you do not have access to current events. 
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM),
            ("human", "{document}"),
        ]
    )

    navigator = prompt | navigator_llm

    data = PyPDFLoader(str(here("./schadenmeldung.pdf"))).load()
    logger.info(data)

    response = navigator.invoke({"document": data[0].page_content})
    logger.info("Simple Chain successfully invoked.")
    logger.info(response.get("claim_type"))

    classification_chain = (
            prompt
            | navigator_llm
            | RunnableLambda(lambda x: x["claim_type"])
            | StrOutputParser()
    )

    logger.info(
        classification_chain.invoke({"document": data[0].page_content})
    )
    logger.info("Chain with RunnableLambda successfully invoked.")

    # Testing the StormCauseInspector
    branch = RunnableBranch(
        (
            lambda x: "storm" in x["claim_type"],
            prompt
            | llm.with_structured_output(StormCauseInspector)
            | RunnableLambda(lambda x: x["cause"])
            | StrOutputParser(),
        ),
        prompt | llm | StrOutputParser(),
    )

    complete_chain = {
                         "claim_type": classification_chain,
                         "document": lambda x: x["document"],
                     } | branch

    logger.info(complete_chain.invoke({"document": data[0].page_content}))
    logger.info(
        classification_chain.invoke({"document": data[0].page_content})
    )
    logger.info(
        "Chain with RunnableLambda and RunnableBranch successfully invoked."
    )
