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

async def debug_sse_stream(url):
    """
    Manually connects to the SSE stream to see if ANY data is flowing.
    This helps detect if Docker/Nginx is buffering the output.
    """
    try:
        print(f"DEBUG: Testing stream at {url}...")
        async with httpx.AsyncClient() as client:
            async with client.stream("GET", url, headers={"Accept": "text/event-stream"}, timeout=2.0) as response:
                if response.status_code != 200:
                    return False, f"Server returned status {response.status_code} (Not 200)"
                
                # Try to read ONE chunk of data
                try:
                    iterator = response.aiter_lines()
                    first_line = await asyncio.wait_for(iterator.__anext__(), timeout=2.0)
                    print(f"DEBUG: Received first line: {first_line}")
                    return True, "Stream is active"
                except asyncio.TimeoutError:
                    return False, "❌ CONNECTION OPEN BUT NO DATA. (Likely Docker Buffering Issue)"
                except StopAsyncIteration:
                    return False, "❌ Connection closed immediately by server."
    except Exception as e:
        return False, f"Connection failed: {str(e)}"

async def create_mcp_client():
    # 1. Connect to SSE Endpoint (with strict timeout)
    sse_ctx = sse_client(MCP_SERVER_URL)
    
    print("DEBUG: Waiting for SSE handshake (endpoint event)...")
    # Wrap the context manager entry in a timeout
    (read_stream, write_stream) = await asyncio.wait_for(sse_ctx.__aenter__(), timeout=5.0)
    
    # 2. Initialize Session
    print("DEBUG: Initializing ClientSession...")
    session = ClientSession(read_stream, write_stream)
    await asyncio.wait_for(session.initialize(), timeout=5.0)
    
    return session, sse_ctx

@cl.on_chat_start
async def start():
    try:
        if not OPENAI_API_KEY:
            await cl.Message("⚠️ Error: OPENAI_API_KEY is missing.").send()
            return

        # --- PHASE 1: DIAGNOSTICS ---
        msg = await cl.Message(f"🔄 Connecting to {MCP_SERVER_URL}...").send()
        
        # Check if data is actually flowing
        is_flowing, debug_msg = await debug_sse_stream(MCP_SERVER_URL)
        
        if not is_flowing:
             # FIX: Set content property first, then update()
             msg.content = f"⚠️ **Connection Warning**\n{debug_msg}\n\n**Fix:** The server is reachable (200 OK), but the data is stuck in the buffer.\n\n**Try this in your Dockerfile:**\n`ENV PYTHONUNBUFFERED=1`"
             await msg.update()
             # We try to proceed anyway, but it will likely fail below

        # --- PHASE 2: REAL CONNECTION ---
        session, sse_ctx = await create_mcp_client()
        
        # Store in session
        cl.user_session.set("mcp_session", session)
        cl.user_session.set("sse_ctx", sse_ctx)
        
        # List Tools
        tools_response = await session.list_tools()
        tools = tools_response.tools
        cl.user_session.set("mcp_tools", tools)

        tool_names = [t.name for t in tools]
        
        # FIX: Set content property first, then update()
        msg.content = f"✅ **Connected!**\n\n**Available Tools:**\n" + "\n".join([f"- `{t}`" for t in tool_names])
        await msg.update()

    except asyncio.TimeoutError:
        print(f"CRITICAL ERROR: Timed out waiting for MCP Handshake.\n{traceback.format_exc()}")
        await cl.Message("❌ **Connection Timed Out (Handshake)**\n\nThe server accepted the connection, but we never received the required 'endpoint' event.\n\n**Solution:**\nThis is almost certainly a **Buffering Issue** inside Docker. Use the 'STDIO' transport method instead of SSE, or ensure your server flushes output immediately.").send()
    
    except Exception as e:
        error_msg = str(e)
        print(f"Detailed Error:\n{traceback.format_exc()}") 
        await cl.Message(f"❌ **Connection Failed**\n\nError: {error_msg}").send()

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

    if not session:
        await cl.Message("⚠️ No connection to MCP server. Answering without tools.").send()
        mcp_tools = []

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
        # If we have tools, use them
        if openai_tools:
            response = await client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                tools=openai_tools,
                tool_choice="auto"
            )
        else:
            # Fallback for no tools
            response = await client.chat.completions.create(
                model="gpt-4o",
                messages=messages
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
        else:
            final_answer = response_message.content

        msg.content = final_answer
        await msg.update()
        messages.append({"role": "assistant", "content": final_answer})
        cl.user_session.set("messages", messages)
    
    except Exception as e:
        await cl.Message(f"❌ Error during chat: {str(e)}").send()