"""
routing_ollama.py

This module demonstrates the routing concept in LangChain using Ollama
"""

from typing import Literal

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from loguru import logger
from pyprojroot import here

# from ollama_functions import OllamaFunctions


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

def routing(info):
    print(info)
    print(type(info))
    if "storm" in info["document"]:
        return RunnablePassthrough.assign(cause=storm_cause_chain)
    else:
        return "default-value"

def final_result_only(info):
    if "storm" in info["document"]:
        return storm_cause_chain
    else:
        return "default-value"


if __name__ == "__main__":
    llm = OllamaFunctions(model="llama3", temperature=0, format="json")

    ollama_structured_model = llm.with_structured_output(ChainNavigator)

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

    logger.info("Setting up a simple navigator chain to test Ollama")
    navigator_chain = chat_prompt | ollama_structured_model

    data = PyPDFLoader(str(here("./schadenmeldung.pdf"))).load()
    logger.info(data)

    response = navigator_chain.invoke({"document": data[0].page_content})
    logger.info(response)
    logger.success("Simple navigator chain test successful")

    type_chain = (
            chat_prompt
            | ollama_structured_model
            | RunnableLambda(lambda x: x.dict())
            | RunnableLambda(lambda x: x["claim_type"])
            | StrOutputParser()
    )
    logger.info("Type chain setup")
    _type = type_chain.invoke({"document": data[0].page_content})
    logger.info(_type)
    logger.success("Chain with RunnableLambda invoked.")

    storm_inspector_llm = llm.with_structured_output(StormCauseInspector)
    logger.info("Storm inspector Model setup successfully")

    storm_cause_chain = (
            chat_prompt
            | storm_inspector_llm
            | RunnableLambda(lambda x: x.dict())
            | RunnableLambda(lambda x: x["cause"])
            | StrOutputParser()
    )

    logger.info("Storm cause chain setup")
    cause = storm_cause_chain.invoke({"document": data[0].page_content})
    logger.success("Second Chain with RunnableLambda invoked.")

    logger.info("Routing with final results setup")
    sequential_chain = (
            {"document": type_chain} | RunnableLambda(final_result_only) | StrOutputParser()
    )

    final_result = sequential_chain.invoke({"document": data[0].page_content})
    logger.success(final_result)

    logger.info("Routing with intermediate results setup")
    seq_chain = {"document": type_chain} | RunnableLambda(routing)
    logger.info("Sequential chain setup")

    structured_response = seq_chain.invoke({"document": data[0].page_content})
    logger.success(structured_response)

    logger.info("Routing successfully completed")
