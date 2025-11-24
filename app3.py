import os
import chainlit as cl
import json
import asyncio
from contextlib import AsyncExitStack
from openai import AsyncOpenAI
from mcp import ClientSession
from mcp.client.sse import sse_client

# --- CONFIGURATION ---
# Updated to your specific endpoint
MCP_SSE_URL = "http://localhost:8000/mcp" 
OPENAI_MODEL = "gpt-4o"

client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

@cl.on_chat_start
async def start():
    # 1. Notify user we are starting the connection sequence
    await cl.Message(content=f"🔄 Connecting to FortiManager at {MCP_SSE_URL}...").send()
    
    exit_stack = AsyncExitStack()
    cl.user_session.set("exit_stack", exit_stack)

    try:
        # 2. Define the connection logic
        async def connect():
            # sse_client connects to /mcp and listens for the 'endpoint' event
            # to know where to send POST messages.
            transport = await exit_stack.enter_async_context(sse_client(MCP_SSE_URL))
            read_stream, write_stream = transport
            
            session = await exit_stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            await session.initialize()
            return session

        # 3. Execute with a 5-second Timeout to prevent UI freezing
        session = await asyncio.wait_for(connect(), timeout=5.0)
        cl.user_session.set("mcp_session", session)
        
        await cl.Message(content="✅ Connection established! Fetching tools...").send()

        # 4. Load Tools
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
        
        await cl.Message(content=f"🚀 System Ready. Loaded {len(tools)} tools.").send()
        
    except asyncio.TimeoutError:
        await cl.Message(content=f"❌ **Connection Timed Out**\nCould not connect to `{MCP_SSE_URL}` within 5 seconds.\nPlease verify the URL is exactly where the SSE stream is published.").send()
    except Exception as e:
        await cl.Message(content=f"❌ **Error**: {str(e)}").send()

@cl.on_message
async def main(message: cl.Message):
    openai_tools = cl.user_session.get("openai_tools")
    mcp_session = cl.user_session.get("mcp_session")
    
    # Fallback: If connection failed, chat normally without tools
    if not mcp_session or not openai_tools:
        await cl.Message(content="⚠️ *Offline Mode (No Tools)*").send()
        response = await client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": message.content}]
        )
        await cl.Message(content=response.choices[0].message.content).send()
        return

    # 1. Ask OpenAI (with tools)
    response = await client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": message.content}],
        tools=openai_tools,
        tool_choice="auto"
    )
    
    msg = response.choices[0].message
    
    # 2. Check for tool calls
    if msg.tool_calls:
        # Notify user which tool is running
        tool_names = ", ".join([t.function.name for t in msg.tool_calls])
        await cl.Message(content=f"🛠️ **Running:** `{tool_names}`").send()

        for tool_call in msg.tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)
            
            async with cl.Step(name=tool_name, type="tool") as step:
                step.input = tool_args
                
                # EXECUTE REMOTE TOOL
                try:
                    result = await mcp_session.call_tool(tool_name, arguments=tool_args)
                    step.output = result.content[0].text
                except Exception as e:
                    step.output = f"Error: {str(e)}"

            # Send the raw tool output to the chat (or feed back to OpenAI if you prefer)
            await cl.Message(content=f"**Output from {tool_name}:**\n```json\n{step.output}\n```").send()
    else:
        # Standard response
        await cl.Message(content=msg.content).send()

@cl.on_chat_end
async def end():
    exit_stack = cl.user_session.get("exit_stack")
    if exit_stack:
        await exit_stack.aclose()