from llm import get_llm

llm = get_llm()
reply = llm.invoke("In one sentence, what is a vector database?")
print(reply.content)