import os
from deepagents import create_deep_agent
from tools import internet_search
from prompts import research_agent_prompt, critique_agent_prompt, competitive_analysis_prompt
from langchain_cerebras import ChatCerebras
from dotenv import load_dotenv
load_dotenv()

llm = ChatCerebras(
    model="qwen-3-235b-a22b-instruct-2507",
    api_key=os.getenv("CEREBRAS_API_KEY")
)

# llm_with_tools = llm.bind_tools([internet_search])
# print(llm_with_tools.invoke("what is ishaangupta1201?").content)
# messages = [
#     (
#         "system",
#         "You are a helpful assistant that answers user's query perfectly",
#     ),
#     ("human", "who are you?"),
# ]
# for chunk in llm.stream(messages):
#     print(chunk.content, end="", flush=True)


# research_instructions = """You are an expert researcher. Your job is to conduct thorough research, and then answer the query.
#
# You have access to a few tools.
#
# ## `internet_search`
#
# Use this to run an internet search for a given query. You can specify the number of results, the topic, and whether raw content should be included.
# """
#
# agent = create_deep_agent(
#     tools=[internet_search],
#     instructions=research_instructions,
#     model=llm,
# )
# result = agent.invoke({"messages": [{"role": "user", "content": "who is ishaangupta-yb?"}]})
# print(result)


research_sub_agent = {
    "name": "research-agent",
    "description": "Expert business intelligence researcher. Use for deep-dive research on specific aspects of companies (e.g., 'Company A pricing and packaging details', 'Company B customer reviews and satisfaction', 'recent partnerships and acquisitions for Company A'). Always call with ONE focused research topic. For multiple topics, call multiple times in parallel.",
    "prompt": research_agent_prompt,
    "tools": [internet_search],
}

critique_sub_agent = {
    "name": "critique-agent",
    "description": "Strategic report reviewer. Use after drafting company_profiles.md or competitive_analysis.md to identify critical gaps and needed improvements. Optionally specify focus areas (e.g., 'focus on strategic recommendations quality' or 'check for balance between both companies').",
    "prompt": critique_agent_prompt,
}

competitive_analysis_agent = create_deep_agent(
    tools=[internet_search],
    instructions=competitive_analysis_prompt,
    subagents=[critique_sub_agent, research_sub_agent],
    model=llm
).with_config({"recursion_limit": 1000})
result = competitive_analysis_agent.invoke({
        "messages": [{"role": "user", "content": "Compare Twitter vs Instagram as social media"}]
    })
print(result)