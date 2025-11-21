import chainlit as cl
import json
import os
import httpx
import traceback
import asyncio
from openai import AsyncOpenAI

# MCP Imports
from mcp.client.session import ClientSession
from mcp.client.sse import sse_client
from mcp.types import CallToolResult, Tool

# Configuration
MCP_SERVER_URL = "http://localhost:8000/mcp" 
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = AsyncOpenAI(api_key=OPENAI_API_KEY)

async def check_server_health(url):
    """
    Diagnostic to check if the server is reachable.
    """
    try:
        async with httpx.AsyncClient() as client:
            headers = {"Accept": "text/event-stream"}
            # We use a short timeout here to fail fast if the server is down
            async with client.stream("GET", url, headers=headers, timeout=3.0) as response:
                if response.status_code == 200:
                    return True, "✅ Server is healthy and accepting SSE connections."
                elif response.status_code == 405:
                    return True, "⚠️ Server found (405), but continuing..."
                else:
                    return True, f"⚠️ Server reachable (Status {response.status_code}). Proceeding..."
                    
    except httpx.ConnectError:
        return False, "❌ Connection Refused. Is Docker running? Did you map ports (8000:8000)?"
    except httpx.ReadTimeout:
        # If we time out reading the stream, it usually means we connected successfully!
        return True, "✅ Server connected (stream active)."
    except Exception as e:
        return False, f"❌ Network Error: {str(e)}"

async def create_mcp_client():
    # Log progress to help debug hanging
    print("DEBUG: Connecting to SSE endpoint...")
    sse_ctx = sse_client(MCP_SERVER_URL)
    
    print("DEBUG: Establishing connection...")
    (read_stream, write_stream) = await sse_ctx.__aenter__()
    
    print("DEBUG: Creating ClientSession...")
    session = ClientSession(read_stream, write_stream)
    
    print("DEBUG: Initializing Session (Handshake)...")
    # CRITICAL FIX: Add timeout to initialization
    # If the server accepts the connection but never sends the handshake, this prevents the hang.
    await asyncio.wait_for(session.initialize(), timeout=5.0)
    
    print("DEBUG: Session Initialized successfully.")
    return session, sse_ctx

@cl.on_chat_start
async def start():
    if not OPENAI_API_KEY:
        await cl.Message("⚠️ Error: OPENAI_API_KEY is missing.").send()
        return

    # 1. Diagnostic Step
    await cl.Message(f"🔄 Attempting to connect to {MCP_SERVER_URL}...").send()
    is_alive, status_msg = await check_server_health(MCP_SERVER_URL)
    
    if not is_alive:
         await cl.Message(f"{status_msg}").send()
         return # Stop here if server is dead
    
    await cl.Message(f"{status_msg}").send()

    # 2. Connect to MCP (Wrapped in Try/Except for detailed errors)
    try:
        # This is where it was hanging before. Now it has a timeout.
        session, sse_ctx = await create_mcp_client()
        
        cl.user_session.set("mcp_session", session)
        cl.user_session.set("sse_ctx", sse_ctx)
        
        # 3. List Tools
        print("DEBUG: Listing tools...")
        tools_response = await session.list_tools()
        tools = tools_response.tools
        cl.user_session.set("mcp_tools", tools)

        tool_names = [t.name for t in tools]
        await cl.Message(
            content=f"✅ **Connected!**\n\n**Available Tools:**\n" + "\n".join([f"- `{t}`" for t in tool_names])
        ).send()

    except asyncio.TimeoutError:
        print(f"ERROR: Connection Timed Out.\n{traceback.format_exc()}")
        await cl.Message("❌ **Connection Timed Out**\nThe server accepted the connection but failed to complete the MCP handshake.\n\n**Check:**\n1. Does your MCP server logs show any errors?\n2. Is the server implementation handling the initialization request?").send()
    
    except Exception as e:
        error_msg = str(e)
        if hasattr(e, 'exceptions'):
            error_msg = " | ".join([str(sub) for sub in e.exceptions])
        
        print(f"Detailed Error:\n{traceback.format_exc()}") 
        await cl.Message(f"❌ **MCP Connection Failed**\n\nError details: {error_msg}").send()

@cl.on_chat_end
async def end():
    sse_ctx = cl.user_session.get("sse_ctx")
    if sse_ctx:
        try:
            await sse_ctx.__aexit__(None, None, None)
        except Exception:
            pass

@cl.on_message
async def main(message: cl.Message):
    session: ClientSession = cl.user_session.get("mcp_session")
    mcp_tools: list[Tool] = cl.user_session.get("mcp_tools")

    # Guard: If session failed to load, warn user but don't crash
    if not session:
        await cl.Message("⚠️ No connection to MCP server. I will try to answer without tools.").send()
        mcp_tools = [] # Empty list so logic below doesn't crash

    openai_tools = []
    if mcp_tools:
        for tool in mcp_tools:
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema
                }
            })

    messages = cl.user_session.get("messages", [])
    messages.append({"role": "user", "content": message.content})

    msg = cl.Message(content="")
    await msg.send()

    try:
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=openai_tools if openai_tools else None,
            tool_choice="auto" if openai_tools else "none"
        )

        response_message = response.choices[0].message

        if response_message.tool_calls and session:
            messages.append(response_message)

            for tool_call in response_message.tool_calls:
                tool_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)

                async with cl.Step(name=tool_name, type="tool") as step:
                    step.input = arguments
                    
                    try:
                        result: CallToolResult = await session.call_tool(tool_name, arguments)
                        output_text = ""
                        if result.content:
                            for content in result.content:
                                if content.type == "text":
                                    output_text += content.text
                                elif content.type == "image":
                                    output_text += "[Image Returned]"
                        else:
                            output_text = "Tool executed successfully."
                            
                        step.output = output_text
                    except Exception as e:
                        output_text = f"Error: {str(e)}"
                        step.output = output_text

                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": tool_name,
                    "content": output_text,
                })

            final_response = await client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
            )
            final_answer = final_response.choices[0].message.content
            msg.content = final_answer
            await msg.update()
            messages.append({"role": "assistant", "content": final_answer})
        
        else:
            final_answer = response_message.content
            msg.content = final_answer
            await msg.update()
            messages.append({"role": "assistant", "content": final_answer})

        cl.user_session.set("messages", messages)
    
    except Exception as e:
        await cl.Message(f"❌ Error during chat: {str(e)}").send()