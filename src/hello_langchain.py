from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in one simple sentence."
)

llm = ChatOllama(
    model="llama3.1",
    temperature=0
)

chain = prompt | llm

response = chain.invoke({
    "topic": "regression testing"
})

print(response.content)