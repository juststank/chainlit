import chainlit as cl
import json
import os
from openai import AsyncOpenAI

# MCP Imports
from mcp.client.session import ClientSession
from mcp.client.sse import sse_client
from mcp.types import CallToolResult, Tool

# Configuration
MCP_SERVER_URL = "http://localhost:8000/sse" 
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI Client
client = AsyncOpenAI(api_key=OPENAI_API_KEY)

async def create_mcp_client():
    """
    Establishes a connection to the MCP server via SSE.
    Returns the session and a context manager to keep it alive.
    """
    # We use the sse_client context manager helper from the mcp SDK
    sse_ctx = sse_client(MCP_SERVER_URL)
    (read_stream, write_stream) = await sse_ctx.__aenter__()
    
    session = ClientSession(read_stream, write_stream)
    await session.initialize()
    
    return session, sse_ctx

@cl.on_chat_start
async def start():
    """
    Initializes the connection to the MCP server and lists available tools.
    """
    if not OPENAI_API_KEY:
        await cl.Message("⚠️ Error: OPENAI_API_KEY is missing from environment variables.").send()
        return

    try:
        # 1. Connect to MCP Server
        session, sse_ctx = await create_mcp_client()
        
        # 2. Store session in user_session for reuse
        cl.user_session.set("mcp_session", session)
        cl.user_session.set("sse_ctx", sse_ctx) # Keep ctx to prevent GC or close
        
        # 3. Fetch Tools
        tools_response = await session.list_tools()
        tools = tools_response.tools
        cl.user_session.set("mcp_tools", tools)

        # 4. Notify User
        tool_names = [t.name for t in tools]
        await cl.Message(
            content=f"✅ Connected to MCP Server at {MCP_SERVER_URL}\n\n**Available Tools:**\n" + "\n".join([f"- `{t}`" for t in tool_names])
        ).send()

    except Exception as e:
        await cl.Message(f"❌ Failed to connect to MCP Server: {str(e)}").send()

@cl.on_chat_end
async def end():
    """
    Clean up the SSE connection when the chat ends.
    """
    sse_ctx = cl.user_session.get("sse_ctx")
    if sse_ctx:
        await sse_ctx.__aexit__(None, None, None)

@cl.on_message
async def main(message: cl.Message):
    session: ClientSession = cl.user_session.get("mcp_session")
    mcp_tools: list[Tool] = cl.user_session.get("mcp_tools")

    if not session:
        await cl.Message("No connection to MCP server. Please restart the chat.").send()
        return

    # 1. Format Tools for OpenAI
    # OpenAI expects a specific JSON schema for function calling
    openai_tools = []
    for tool in mcp_tools:
        openai_tools.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema
            }
        })

    # 2. Prepare Conversation History
    messages = cl.user_session.get("messages", [])
    messages.append({"role": "user", "content": message.content})

    # 3. First Call to LLM (Decision Making)
    # We ask the LLM: "Given this user message and these tools, what should I do?"
    msg = cl.Message(content="")
    await msg.send()

    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=openai_tools if openai_tools else None,
        tool_choice="auto" if openai_tools else "none"
    )

    response_message = response.choices[0].message

    # 4. Handle Tool Calls
    if response_message.tool_calls:
        # The LLM wants to use a tool
        messages.append(response_message) # Add the "thought" to history

        for tool_call in response_message.tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)

            # Notify UI we are running a tool
            async with cl.Step(name=tool_name, type="tool") as step:
                step.input = arguments
                
                try:
                    # EXECUTE THE TOOL ON DOCKER CONTAINER via MCP
                    result: CallToolResult = await session.call_tool(tool_name, arguments)
                    
                    # Format result
                    output_text = ""
                    if result.content:
                        for content in result.content:
                            if content.type == "text":
                                output_text += content.text
                            elif content.type == "image":
                                output_text += "[Image Returned]"
                    else:
                        output_text = "Tool executed successfully with no output."
                        
                    step.output = output_text
                except Exception as e:
                    output_text = f"Error executing tool: {str(e)}"
                    step.output = output_text

            # Add tool result to history for the LLM to see
            messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": tool_name,
                "content": output_text,
            })

        # 5. Second Call to LLM (Final Answer)
        # Now that the LLM has the tool results, ask it to generate the final answer
        final_response = await client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
        )
        final_answer = final_response.choices[0].message.content
        msg.content = final_answer
        await msg.update()
        
        # Update history
        messages.append({"role": "assistant", "content": final_answer})
    
    else:
        # No tool needed, just standard chat
        final_answer = response_message.content
        msg.content = final_answer
        await msg.update()
        messages.append({"role": "assistant", "content": final_answer})

    # Save history
    cl.user_session.set("messages", messages)