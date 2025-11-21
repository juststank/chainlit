# app.py
import os
import asyncio
import chainlit as cl
from openai import OpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

client = OpenAI()
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Global MCP session
mcp_session = None

@cl.on_chat_start
async def start():
    """Initialize MCP connection when chat starts"""
    global mcp_session
    
    # Configure your FortiManager MCP server path
    server_params = StdioServerParameters(
        command="python",  # or "node" if it's a Node.js server
        args=["/path/to/your/fortimanager-mcp/server.py"],  # adjust path
        env=None
    )
    
    try:
        # Create MCP client connection
        stdio_transport = await stdio_client(server_params)
        mcp_session = ClientSession(stdio_transport[0], stdio_transport[1])
        await mcp_session.initialize()
        
        # List available tools
        tools_result = await mcp_session.list_tools()
        tool_names = [tool.name for tool in tools_result.tools]
        
        await cl.Message(
            content=f"Connected to FortiManager MCP server!\nAvailable tools: {', '.join(tool_names)}"
        ).send()
    except Exception as e:
        await cl.Message(content=f"Failed to connect to MCP server: {str(e)}").send()

@cl.on_message
async def on_message(message: cl.Message):
    if not mcp_session:
        await cl.Message(content="MCP server not connected!").send()
        return
    
    # Get available tools from MCP server
    tools_result = await mcp_session.list_tools()
    
    # Convert MCP tools to OpenAI function format
    openai_tools = []
    for tool in tools_result.tools:
        openai_tools.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema
            }
        })
    
    # Initial API call with tools
    messages = [
        {"role": "system", "content": "You are a helpful assistant with access to FortiManager tools."},
        {"role": "user", "content": message.content}
    ]
    
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=openai_tools,
        temperature=0.2
    )
    
    # Handle tool calls
    while response.choices[0].message.tool_calls:
        messages.append(response.choices[0].message)
        
        for tool_call in response.choices[0].message.tool_calls:
            tool_name = tool_call.function.name
            tool_args = eval(tool_call.function.arguments)  # Use json.loads in production
            
            # Call MCP tool
            result = await mcp_session.call_tool(tool_name, tool_args)
            
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(result.content)
            })
        
        # Get next response
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=openai_tools,
            temperature=0.2
        )
    
    await cl.Message(content=response.choices[0].message.content).send()

@cl.on_chat_end
async def end():
    """Cleanup MCP connection"""
    global mcp_session
    if mcp_session:
        await mcp_session.__aexit__(None, None, None)