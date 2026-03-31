import asyncio
import os
from langchain_mcp_adapters.client import MultiServerMCPClient
# Updated import to avoid the LangGraph deprecation warning
from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

async def main():
    client_config = {
        "math": {
            "command": "python",
            "args": ["mathserver.py"],
            "transport": "stdio",
        }
    }

    client = MultiServerMCPClient(client_config)
    tools = await client.get_tools()
    model = ChatGroq(model="llama-3.3-70b-versatile") 
    
    agent = create_react_agent(model, tools)

    math_response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "What is 10 + 20?"}]}
    )
    
    print(math_response["messages"][-1].content)

if __name__ == "__main__":
    asyncio.run(main())