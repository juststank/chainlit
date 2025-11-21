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
    """
    try:
        print(f"DEBUG: Testing stream at {url}...")
        async with httpx.AsyncClient() as client:
            headers = {"Accept": "text/event-stream"}
            async with client.stream("GET", url, headers=headers, timeout=5.0) as response:
                if response.status_code != 200:
                    return False, f"Server returned status {response.status_code} (Not 200)"
                
                print(f"DEBUG: Connection Open (200 OK). Waiting for data...")
                # Try to read ONE chunk of data to verify flow
                try:
                    iterator = response.aiter_lines()
                    first_line = await asyncio.wait_for(iterator.__anext__(), timeout=5.0)
                    print(f"DEBUG: Received first line: {first_line}")
                    return True, "Stream is active"
                except asyncio.TimeoutError:
                    return False, "❌ CONNECTION OPEN BUT NO DATA. (Docker Buffering suspected)"
                except StopAsyncIteration:
                    return False, "❌ Connection closed immediately by server."
    except Exception as e:
        return False, f"Connection failed: {str(e)}"

async def create_mcp_client():
    # 1. Connect to SSE Endpoint
    sse_ctx = sse_client(MCP_SERVER_URL)
    
    print("DEBUG: Waiting for SSE handshake (endpoint event)...")
    # FIX 1: Increased timeout from 5s to 30s
    # Docker sometimes delays the first flush of data
    (read_stream, write_stream) = await asyncio.wait_for(sse_ctx.__aenter__(), timeout=30.0)
    
    # 2. Initialize Session
    print("DEBUG: Initializing ClientSession...")
    session = ClientSession(read_stream, write_stream)
    await asyncio.wait_for(session.initialize(), timeout=30.0)
    
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
             # Warning only - we try to proceed anyway
             msg.content = f"⚠️ **Connection Warning**\n{debug_msg}\n\n**Potential Fix:**\nYour Docker container is likely buffering the output. Add `ENV PYTHONUNBUFFERED=1` to your Dockerfile and rebuild."
             await msg.update()

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
        
        msg.content = f"✅ **Connected!**\n\n**Available Tools:**\n" + "\n".join([f"- `{t}`" for t in tool_names])
        await msg.update()

    except asyncio.TimeoutError:
        print(f"CRITICAL ERROR: Timed out waiting for MCP Handshake.\n{traceback.format_exc()}")
        await cl.Message("❌ **Connection Timed Out**\n\nThe server accepted the connection (200 OK) but sent no data for 30 seconds.\n\n**REQUIRED FIX:**\n1. Stop your Docker container.\n2. Add `ENV PYTHONUNBUFFERED=1` to your Dockerfile.\n3. Rebuild with `docker compose build`.\n4. Restart `docker compose up -d`.").send()
    
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