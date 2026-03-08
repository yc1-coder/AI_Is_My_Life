from langchain_community.llms import Ollama  #倒入包
llm = Ollama(model = 'qwen:1.8b')
response = llm.invoke("AI会对人类文明产生深远的影响吗")
print(response)
