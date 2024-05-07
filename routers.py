"""
routers.py

This module demonstrates the routing concept in LangChain.
It includes two examples: one for chains and another for agents.

"""
import os
from typing import Literal

from dotenv import find_dotenv, load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableLambda,
    RunnableBranch,
    RunnablePassthrough,
)
from langchain_openai import ChatOpenAI
from loguru import logger
from pydantic import BaseModel, Field
from pyprojroot import here


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
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo", openai_api_key=api_key, temperature=0
    )

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

    type_chain = (
        prompt
        | navigator_llm
        | RunnableLambda(lambda x: x["claim_type"])
        | StrOutputParser()
    )

    logger.info(type_chain.invoke({"document": data[0].page_content}))
    logger.info("Chain with RunnableLambda successfully invoked.")

    storm_inspector_llm = llm.with_structured_output(StormCauseInspector)

    storm_cause_chain = (
        prompt
        | storm_inspector_llm
        | RunnableLambda(lambda x: x["cause"])
        | StrOutputParser()
    )

    branch = RunnableBranch(
        (
            lambda x: "storm" in x["claim_type"],
            storm_cause_chain,
        ),
        RunnableLambda(lambda _: "default-value"),
    )

    complete_chain = {
        "claim_type": type_chain,
        "document": lambda x: x["document"],
    } | branch

    final_respone = complete_chain.invoke({"document": data[0].page_content})
    logger.info(final_respone)

    logger.info("Rounting with functional approach")

    def route(info):
        if "storm" in info["document"]:
            return storm_cause_chain
        else:
            return "default-value"

    sequential_chain = (
        {"document": type_chain} | RunnableLambda(route) | StrOutputParser()
    )
    logger.info(sequential_chain.invoke({"document": data[0].page_content}))

    def routing(info):
        if "storm" in info["document"]:
            return RunnablePassthrough.assign(cause=storm_cause_chain)
        else:
            return "default-value"

    logger.info("Functional Routing with intermediate results")
    seq_chain = {"document": type_chain} | RunnableLambda(routing)

    logger.info(seq_chain.invoke({"document": data[0].page_content}))
