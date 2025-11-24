import os
import chainlit as cl
import json
from contextlib import AsyncExitStack
from openai import AsyncOpenAI
from mcp import ClientSession
from mcp.client.sse import sse_client

# Configuration
MCP_SSE_URL = "http://localhost:8000/mcp"  # FastMCP servers usually expose /mcp
OPENAI_MODEL = "gpt-4o"

client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

@cl.on_chat_start
async def start():
    # 1. Initialize the ExitStack to manage our async context managers
    # This keeps the connection alive without blocking the chat loop
    exit_stack = AsyncExitStack()
    cl.user_session.set("exit_stack", exit_stack)

    try:
        # 2. Connect to the SSE Endpoint
        # We use enter_async_context so the connection stays open after this function returns
        transport = await exit_stack.enter_async_context(sse_client(MCP_SSE_URL))
        read_stream, write_stream = transport

        # 3. Create the MCP Client Session
        session = await exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        await session.initialize()
        
        # Save session for later use
        cl.user_session.set("mcp_session", session)

        # 4. List Tools
        result = await session.list_tools()
        tools = result.tools

        # 5. Convert to OpenAI format
        openai_tools = []
        for tool in tools:
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema
                }
            })
        
        cl.user_session.set("openai_tools", openai_tools)
        
        await cl.Message(content=f"✅ Connected to FortiManager (HTTP Mode) at {MCP_SSE_URL}").send()
        
    except Exception as e:
        await cl.Message(content=f"❌ Connection Failed: {str(e)}").send()

@cl.on_message
async def main(message: cl.Message):
    openai_tools = cl.user_session.get("openai_tools")
    mcp_session = cl.user_session.get("mcp_session")
    
    if not mcp_session:
        await cl.Message(content="No active MCP connection.").send()
        return

    # 1. Call OpenAI
    response = await client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": message.content}],
        tools=openai_tools,
        tool_choice="auto"
    )
    
    msg = response.choices[0].message
    
    # 2. Handle Tool Calls
    if msg.tool_calls:
        for tool_call in msg.tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)
            
            async with cl.Step(name=tool_name, type="tool") as step:
                step.input = tool_args
                
                # Execute tool over HTTP
                result = await mcp_session.call_tool(tool_name, arguments=tool_args)
                
                step.output = result.content[0].text
                
            # Optional: Send result back to OpenAI for final summary (omitted for brevity)
            await cl.Message(content=f"Result: {result.content[0].text}").send()
    else:
        await cl.Message(content=msg.content).send()

@cl.on_chat_end
async def end():
    # Cleanup: Close the SSE connection and Session cleanly
    exit_stack = cl.user_session.get("exit_stack")
    if exit_stack:
        await exit_stack.aclose()