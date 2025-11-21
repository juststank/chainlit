# app.py
import os
import json
import asyncio
import chainlit as cl
from openai import OpenAI

client = OpenAI()
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# MCP Server URL
MCP_SERVER_URL = "http://localhost:8000/mcp"

# Store MCP session globally
mcp_session = None
mcp_streams = None

async def init_mcp_session():
    """Initialize MCP session with proper imports"""
    global mcp_session, mcp_streams
    
    try:
        from mcp import ClientSession
        from mcp.client.sse import sse_client
        
        # Connect to MCP server via SSE
        read_stream, write_stream = await sse_client(MCP_SERVER_URL).__aenter__()
        mcp_streams = (read_stream, write_stream)
        
        # Create session
        session = ClientSession(read_stream, write_stream)
        await session.__aenter__()
        
        # Initialize
        init_result = await session.initialize()
        
        mcp_session = session
        return True
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please install: pip install mcp httpx-sse")
        return False
    except Exception as e:
        print(f"Connection error: {e}")
        import traceback
        traceback.print_exc()
        return False

@cl.on_chat_start
async def start():
    """Initialize MCP connection when chat starts"""
    global mcp_session
    
    try:
        await cl.Message(content="🔄 Connecting to FortiManager MCP server...").send()
        
        success = await init_mcp_session()
        
        if not success or not mcp_session:
            await cl.Message(
                content="❌ Failed to initialize MCP session. Check console for details."
            ).send()
            return
        
        # List available tools
        tools_result = await mcp_session.list_tools()
        
        if hasattr(tools_result, 'tools'):
            tool_names = [tool.name for tool in tools_result.tools]
            tool_descriptions = {tool.name: tool.description for tool in tools_result.tools}
            
            message = f"✅ Connected to FortiManager MCP server!\n\n**Available tools ({len(tool_names)}):**\n\n"
            for name in tool_names:
                desc = tool_descriptions.get(name, "No description")
                message += f"• **{name}**: {desc}\n"
            
            await cl.Message(content=message).send()
        else:
            await cl.Message(content="✅ Connected but no tools found.").send()
            
    except Exception as e:
        await cl.Message(
            content=f"❌ Error: {str(e)}\n\nMake sure to install: `pip install mcp httpx-sse`"
        ).send()
        import traceback
        traceback.print_exc()

@cl.on_message
async def on_message(message: cl.Message):
    global mcp_session
    
    if not mcp_session:
        await cl.Message(content="❌ MCP server not connected. Please restart the chat.").send()
        return
    
    try:
        # Get available tools
        tools_result = await mcp_session.list_tools()
        
        # Convert MCP tools to OpenAI format
        openai_tools = []
        for tool in tools_result.tools:
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.inputSchema if hasattr(tool, 'inputSchema') else {}
                }
            })
        
        # Initial API call
        messages = [
            {"role": "system", "content": "You are a helpful assistant with access to FortiManager tools. Use them to help manage and monitor FortiGate devices and policies."},
            {"role": "user", "content": message.content}
        ]
        
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=openai_tools if openai_tools else None,
            temperature=0.2
        )
        
        # Handle tool calls
        max_iterations = 5
        iteration = 0
        
        while response.choices[0].message.tool_calls and iteration < max_iterations:
            iteration += 1
            assistant_message = response.choices[0].message
            
            # Add assistant message to conversation
            messages.append({
                "role": "assistant",
                "content": assistant_message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    } for tc in assistant_message.tool_calls
                ]
            })
            
            for tool_call in assistant_message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)
                
                await cl.Message(
                    content=f"🔧 Calling tool: **{tool_name}**\n```json\n{json.dumps(tool_args, indent=2)}\n```"
                ).send()
                
                # Call MCP tool
                result = await mcp_session.call_tool(tool_name, tool_args)
                
                # Extract content from result
                if hasattr(result, 'content'):
                    if isinstance(result.content, list):
                        tool_response = "\n".join([
                            item.text if hasattr(item, 'text') else str(item) 
                            for item in result.content
                        ])
                    else:
                        tool_response = str(result.content)
                else:
                    tool_response = str(result)
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_response
                })
            
            # Get next response
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=openai_tools if openai_tools else None,
                temperature=0.2
            )
        
        await cl.Message(content=response.choices[0].message.content).send()
        
    except Exception as e:
        await cl.Message(content=f"❌ Error: {str(e)}").send()
        import traceback
        traceback.print_exc()

@cl.on_chat_end
async def end():
    """Cleanup MCP connection"""
    global mcp_session, mcp_streams
    try:
        if mcp_session:
            await mcp_session.__aexit__(None, None, None)
        if mcp_streams:
            # Cleanup streams if needed
            pass
    except Exception as e:
        print(f"Cleanup error: {e}")