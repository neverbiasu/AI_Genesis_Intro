# import os
# from typing import List
# from ChatGLM3 import ChatGLM3

# from langchain.agents import load_tools
# from langchain.agents import initialize_agent
# from langchain.agents import AgentType
# from langchain_core.prompts import ChatPromptTemplate

# # MODEL_PATH = os.environ.get('MODEL_PATH', 'THUDM/chatglm3-6b')
# MODEL_PATH = "/home/faych/.vscode-server/data/ChatGLM3/models"

# def run_tool(tools, llm, prompt_chain: List[str]):
#     loaded_tolls = []
#     for tool in tools:
#         if isinstance(tool, str):
#             loaded_tolls.append(load_tools([tool], llm=llm)[0])
#         else:
#             loaded_tolls.append(tool)
#     agent = initialize_agent(
#         loaded_tolls, llm,
#         agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
#         verbose=True,
#         handle_parsing_errors=True
#     )
#     for prompt in prompt_chain:
#         agent.run(prompt)



                                       

# if __name__ == "__main__":
#     llm = ChatGLM3()
#     llm.load_model(model_name_or_path=MODEL_PATH)

#     tmp_prompt = "tell me a joke about {foo}"
#     prompt = ChatPromptTemplate.from_template(tmp_prompt)
#     chain = prompt | llm
#     chain.invoke({"foo": "bears"})
#     # arxiv: 单个工具调用示例 1
#     # run_tool(["arxiv"], llm, [
#     #     "帮我查询GLM-130B相关工作"
#     # ])

#     # '''# weather: 单个工具调用示例 2
#     # run_tool([Weather()], llm, [
#     #     "今天北京天气怎么样？",
#     #     "What's the weather like in Shanghai today",
#     # ])'''

#     # calculator: 单个工具调用示例 3
#     # run_tool([Calculator()], llm, [
#     #     "12345679乘以54等于多少？",
#     #     "3.14的3.14次方等于多少？",
#     #     "根号2加上根号三等于多少？",
#     # ]),

#     # arxiv + weather + calculator: 多个工具结合调用
#     # run_tool([Calculator(), "arxiv", Weather()], llm, [
#     #     "帮我检索GLM-130B相关论文",
#     #     "今天北京天气怎么样？",
#     #     "根号3减去根号二再加上4等于多少？",
#     # ])


from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema.messages import AIMessage
from langchain_community.llms.chatglm3 import ChatGLM3

template = """{question}"""
prompt = PromptTemplate.from_template(template)

endpoint_url = "http://127.0.0.1:8000/v1/chat/completions"

messages = [
    AIMessage(content="我将从美国到中国来旅游，出行前希望了解中国的城市"),
    AIMessage(content="欢迎问我任何问题。"),
]

llm = ChatGLM3(
    endpoint_url=endpoint_url,
    max_tokens=80000,
    prefix_messages=messages,
    top_p=0.9,
)

# llm_chain = LLMChain(prompt=prompt, llm=llm)
# question = "北京和上海两座城市有什么不同？"

# llm_chain.run(question)
chain = prompt | llm
chain.invoke({"question": "北京和上海两座城市有什么不同？"})