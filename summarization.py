from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
import openai
from config import Config

openai.api_key = Config.OPENAI_API_KEY

# Function to summarize document using GPT-3.5 Turbo
def summarize_document(text):
    prompt = f"Summarize the following document:\n\n{text}"

    # Using OpenAI API directly to summarize
    response = openai.Completion.create(
        model="gpt-4",
        prompt=prompt,
        max_tokens=150
    )

    return response.choices[0].text.strip()

# Optionally use LangChain for more advanced chaining or processing
def langchain_summary(text):
    template = 'Summarize the following document:\n\n{text},do not mention sorry anywhere unless there is no {text}"
    prompt = PromptTemplate(input_variables=["text"], template=template)
    llm = OpenAI(model="gpt-3.5-turbo")
    chain = LLMChain(llm=llm, prompt=prompt)
    summary = chain.run({"text": text})
    return summary
