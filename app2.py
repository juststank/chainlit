# app.py
import os
import json
import chainlit as cl
from openai import OpenAI
from mcp import ClientSession
from mcp.client.sse import sse_client

client = OpenAI()
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Global MCP session
mcp_session = None
mcp_context = None

@cl.on_chat_start
async def start():
    """Initialize MCP connection when chat starts"""
    global mcp_session, mcp_context
    
    # URL of your FortiManager MCP server (adjust port if needed)
    server_url = "http://localhost:3000/sse"  # Change to your server URL
    
    try:
        # Create MCP client connection using SSE
        mcp_context = sse_client(server_url)
        read_stream, write_stream = await mcp_context.__aenter__()
        
        mcp_session = ClientSession(read_stream, write_stream)
        await mcp_session.__aenter__()
        
        # Initialize the session
        await mcp_session.initialize()
        
        # List available tools
        tools_result = await mcp_session.list_tools()
        tool_names = [tool.name for tool in tools_result.tools]
        
        await cl.Message(
            content=f"✅ Connected to FortiManager MCP server!\n\n**Available tools:**\n" + 
                    "\n".join([f"- {name}" for name in tool_names])
        ).send()
    except Exception as e:
        await cl.Message(content=f"❌ Failed to connect to MCP server: {str(e)}").send()
        raise

@cl.on_message
async def on_message(message: cl.Message):
    if not mcp_session:
        await cl.Message(content="MCP server not connected!").send()
        return
    
    try:
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
            {"role": "system", "content": "You are a helpful assistant with access to FortiManager tools. Use them to help manage FortiGate devices."},
            {"role": "user", "content": message.content}
        ]
        
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=openai_tools if openai_tools else None,
            temperature=0.2
        )
        
        # Handle tool calls in a loop
        max_iterations = 5  # Prevent infinite loops
        iteration = 0
        
        while response.choices[0].message.tool_calls and iteration < max_iterations:
            iteration += 1
            assistant_message = response.choices[0].message
            messages.append(assistant_message)
            
            # Execute each tool call
            for tool_call in assistant_message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)
                
                # Show which tool is being called
                await cl.Message(
                    content=f"🔧 Calling tool: `{tool_name}`\nArguments: `{json.dumps(tool_args, indent=2)}`"
                ).send()
                
                # Call MCP tool
                result = await mcp_session.call_tool(tool_name, tool_args)
                
                # Extract content from result
                if hasattr(result, 'content'):
                    if isinstance(result.content, list):
                        tool_response = "\n".join([str(item.text) if hasattr(item, 'text') else str(item) for item in result.content])
                    else:
                        tool_response = str(result.content)
                else:
                    tool_response = str(result)
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_response
                })
            
            # Get next response from OpenAI
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=openai_tools if openai_tools else None,
                temperature=0.2
            )
        
        # Send final response
        await cl.Message(content=response.choices[0].message.content).send()
        
    except Exception as e:
        await cl.Message(content=f"❌ Error: {str(e)}").send()
        import traceback
        print(traceback.format_exc())

@cl.on_chat_end
async def end():
    """Cleanup MCP connection"""
    global mcp_session, mcp_context
    try:
        if mcp_session:
            await mcp_session.__aexit__(None, None, None)
        if mcp_context:
            await mcp_context.__aexit__(None, None, None)
    except Exception as e:
        print(f"Error during cleanup: {e}")