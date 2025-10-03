import os
from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model

from tools import internet_search
from prompts import research_agent_prompt, critique_agent_prompt, competitive_analysis_prompt
from langchain_cerebras import ChatCerebras
from dotenv import load_dotenv
load_dotenv()

llm = ChatCerebras(
    model="gpt-oss-120b",
    api_key=os.getenv("CEREBRAS_API_KEY")
)

# messages = [
#     (
#         "system",
#         "You are a helpful assistant that answers user's query perfectly",
#     ),
#     ("human", "who are you?"),
# ]
# for chunk in llm.stream(messages):
#     print(chunk.content, end="", flush=True)


research_instructions = """You are an expert researcher. Your job is to conduct thorough research, and then answer the query.

You have access to a few tools.

## `internet_search`

Use this to run an internet search for a given query. You can specify the number of results, the topic, and whether raw content should be included.
"""

agent = create_deep_agent(
    tools=[internet_search],
    instructions=research_instructions,
    model=llm,
)

result = agent.invoke({"messages": [{"role": "user", "content": "today date?"}]})
print(result)