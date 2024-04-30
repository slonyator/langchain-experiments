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
from langchain_core.runnables import RunnableLambda, RunnableBranch
from langchain_openai import ChatOpenAI
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
    print(data)

    # Testing the navigotor
    response = navigator.invoke({"document": data[0].page_content})
    print(response.get("claim_type"))

    # Embedding the navigator in a real chain
    classification_chain = (
        prompt
        | navigator_llm
        | RunnableLambda(lambda x: x["claim_type"])
        | StrOutputParser()
    )

    print(classification_chain.invoke({"document": data[0].page_content}))

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

    print(complete_chain.invoke({"document": data[0].page_content}))
