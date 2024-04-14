import os

from dotenv import find_dotenv, load_dotenv
from langchain_openai import OpenAI
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate

if __name__ == "__main__":
    load_dotenv(find_dotenv())
    api_key = os.environ["OPENAI_API_KEY"]

    llm = OpenAI(
        model_name="gpt-3.5-turbo-instruct",
        temperature=0,
        openai_api_key=api_key,
    )

    template = """Question: {question}
    Answer: """

    prompt = PromptTemplate.from_template(template)
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    question = "What is the best insurance company in Germany? Name one!"
    print(llm_chain.invoke(question))
