from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

def analyze_document(text):
    # Example function for document analysis (can be expanded)
    analysis_prompt = PromptTemplate(
        input_variables=["text"],
        template="Analyze the following document and provide key insights:\n\n{text}"
    )
    llm = OpenAI()
    return llm.run(analysis_prompt.format(text=text))
