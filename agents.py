import os

from dotenv import load_dotenv, find_dotenv
from langchain.agents import AgentExecutor
from langchain.agents import tool
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import (
    OpenAIToolsAgentOutputParser,
)
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools.ddg_search import DuckDuckGoSearchRun
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pyprojroot import here


@tool
def damage_detector(text: str) -> str:
    """Returns a summary of the damaged items"""

    template = (
        "text: {text} \n\n Instruction: Erstelle eine Liste aller "
        "beschädigten Objekte. Gib, falls vorhanden, die Höhe des "
        "Schadens für jeden Gegenstand an."
    )
    prompt = PromptTemplate(
        input_variables=["text"],
        template=template,
    )

    summary_chain = prompt | ChatOpenAI(
        openai_api_key=api_key,
        temperature=0,
        model_name="gpt-4",
    )

    summary = summary_chain.invoke({"text": text})

    return summary.content


@tool
def calculator(text: str) -> str:
    """Returns the total cost of the damaged items"""

    template = (
        "text: {text} \n\n Instruction: Berechne die Gesamtkosten "
        "der beschädigten Objekte."
    )
    prompt = PromptTemplate(
        input_variables=["text"],
        template=template,
    )

    calculator_chain = prompt | ChatOpenAI(
        openai_api_key=api_key,
        temperature=0,
        model_name="gpt-4",
    )

    cost = calculator_chain.invoke({"text": text})

    return cost.content


@tool
def online_checker(text: str) -> str:
    """Checks on the web weather the price for an item is supposed to be in a reasonable range"""

    search_engine = DuckDuckGoSearchRun()

    result = search_engine.run(text)

    return result


if __name__ == "__main__":
    load_dotenv(find_dotenv())
    api_key = os.getenv("OPENAI_API_KEY")

    data = PyPDFLoader(str(here("./schadenmeldung.pdf"))).load()
    print(data)

    print(
        "-------------------------------------------------------------------"
    )
    summary = damage_detector(data[0].page_content)
    print(summary)
    costs = calculator(summary)
    print(
        "-------------------------------------------------------------------"
    )
    print(costs)

    tools = [damage_detector, calculator, online_checker]

    llm = ChatOpenAI(model_name="gpt-4", openai_api_key=api_key, temperature=0)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are very powerful assistant, but don't know current events",
            ),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    llm_with_tools = llm.bind_tools(tools)

    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_to_openai_tool_messages(
                x["intermediate_steps"]
            ),
        }
        | prompt
        | llm_with_tools
        | OpenAIToolsAgentOutputParser()
    )

    print(
        "-------------------------------------------------------------------"
    )

    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    response = agent_executor.invoke(
        {"input": data[0].page_content}, return_only_outputs=True
    )

    print(response)
