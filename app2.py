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
connection_task = None

async def init_mcp_session():
    """Initialize MCP session with proper imports"""
    try:
        from mcp import ClientSession
        from mcp.client.sse import sse_client
        
        print(f"Attempting to connect to {MCP_SERVER_URL}...")
        
        # Connect with timeout
        async with asyncio.timeout(10):  # 10 second timeout
            sse_context = sse_client(MCP_SERVER_URL)
            read_stream, write_stream = await sse_context.__aenter__()
            
            # Create session
            session = ClientSession(read_stream, write_stream)
            await session.__aenter__()
            
            # Initialize
            await session.initialize()
            
            print("MCP session initialized successfully")
            return session
        
    except asyncio.TimeoutError:
        print("Connection timeout - server may not support SSE at /mcp endpoint")
        return None
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please install: pip install mcp httpx-sse")
        return None
    except Exception as e:
        print(f"Connection error: {e}")
        import traceback
        traceback.print_exc()
        return None

@cl.on_chat_start
async def start():
    """Initialize MCP connection when chat starts"""
    global mcp_session, connection_task
    
    # Start connection in background
    connection_task = asyncio.create_task(init_mcp_session())
    
    # Send initial message immediately (don't wait)
    await cl.Message(content="🔄 Connecting to FortiManager MCP server in background...\n\nYou can start chatting, but tools won't be available until connected.").send()
    
    # Wait for connection with timeout
    try:
        async with asyncio.timeout(5):
            mcp_session = await connection_task
    except asyncio.TimeoutError:
        await cl.Message(content="⚠️ Connection is taking longer than expected. Continuing in background...").send()
        return
    
    # Connection completed
    if mcp_session:
        try:
            tools_result = await mcp_session.list_tools()
            
            if hasattr(tools_result, 'tools'):
                tool_names = [tool.name for tool in tools_result.tools]
                
                message = f"✅ Connected to FortiManager MCP!\n\n**Available tools ({len(tool_names)}):**\n"
                for name in tool_names[:10]:  # Show first 10
                    message += f"• {name}\n"
                
                if len(tool_names) > 10:
                    message += f"... and {len(tool_names) - 10} more"
                
                await cl.Message(content=message).send()
            else:
                await cl.Message(content="✅ Connected but no tools found.").send()
        except Exception as e:
            await cl.Message(content=f"⚠️ Connected but error listing tools: {str(e)}").send()
    else:
        await cl.Message(content="❌ Failed to connect to MCP server. Check console logs.\n\nYou can still chat, but FortiManager tools won't be available.").send()

@cl.on_message
async def on_message(message: cl.Message):
    global mcp_session, connection_task
    
    # Check if connection is still in progress
    if connection_task and not connection_task.done():
        await cl.Message(content="⏳ Still connecting to MCP server... Please wait.").send()
        try:
            mcp_session = await asyncio.wait_for(connection_task, timeout=10)
        except asyncio.TimeoutError:
            await cl.Message(content="❌ Connection timeout. Proceeding without MCP tools.").send()
    
    # Continue with or without MCP
    try:
        openai_tools = []
        
        if mcp_session:
            try:
                # Get available tools
                tools_result = await mcp_session.list_tools()
                
                # Convert MCP tools to OpenAI format
                for tool in tools_result.tools:
                    openai_tools.append({
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description or "",
                            "parameters": tool.inputSchema if hasattr(tool, 'inputSchema') else {}
                        }
                    })
            except Exception as e:
                await cl.Message(content=f"⚠️ Error accessing tools: {str(e)}").send()
        
        # Initial API call
        system_content = "You are a helpful assistant"
        if openai_tools:
            system_content += " with access to FortiManager tools. Use them to help manage and monitor FortiGate devices and policies."
        
        messages = [
            {"role": "system", "content": system_content},
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
        
        while response.choices[0].message.tool_calls and iteration < max_iterations and mcp_session:
            iteration += 1
            assistant_message = response.choices[0].message
            
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
                    content=f"🔧 **{tool_name}**\n```json\n{json.dumps(tool_args, indent=2)}\n```"
                ).send()
                
                try:
                    # Call MCP tool
                    result = await mcp_session.call_tool(tool_name, tool_args)
                    
                    # Extract content
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
                except Exception as e:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": f"Error calling tool: {str(e)}"
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