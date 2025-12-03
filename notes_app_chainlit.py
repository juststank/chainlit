import chainlit as cl
from openai import AsyncOpenAI
import json
from dotenv import load_dotenv
import os

load_dotenv()

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@cl.on_chat_start
async def start():
    """Initialize the chat session"""
    cl.user_session.set("message_history", [
        {"role": "system", "content": "You are a helpful assistant with access to note-taking tools."}
    ])
    
    await cl.Message(
        content="Hello! Connect to MCP servers using the plug icon (🔌) in the sidebar to enable tools."
    ).send()

@cl.on_mcp_connect
async def on_mcp_connect(connection):
    """Handle MCP server connections"""
    print(f"🔌 MCP server '{connection.name}' connected")
    
    # Get tools from the connection
    tools = await connection.list_tools()
    
    # Convert to OpenAI format and store in session
    mcp_tools = cl.user_session.get("mcp_tools", {})
    mcp_tools[connection.name] = [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema
            }
        }
        for tool in tools
    ]
    cl.user_session.set("mcp_tools", mcp_tools)
    
    # Store the connection for tool calls
    mcp_connections = cl.user_session.get("mcp_connections", {})
    mcp_connections[connection.name] = connection
    cl.user_session.set("mcp_connections", mcp_connections)
    
    tool_names = [tool.name for tool in tools]
    await cl.Message(
        content=f"✅ Connected to **{connection.name}**\n\nTools: {', '.join(tool_names)}",
        author="System"
    ).send()

@cl.on_mcp_disconnect
async def on_mcp_disconnect(name: str):
    """Handle MCP server disconnections"""
    print(f"🔌 MCP server '{name}' disconnected")
    
    # Remove tools and connection from session
    mcp_tools = cl.user_session.get("mcp_tools", {})
    mcp_tools.pop(name, None)
    cl.user_session.set("mcp_tools", mcp_tools)
    
    mcp_connections = cl.user_session.get("mcp_connections", {})
    mcp_connections.pop(name, None)
    cl.user_session.set("mcp_connections", mcp_connections)
    
    await cl.Message(
        content=f"❌ Disconnected from **{name}**",
        author="System"
    ).send()

@cl.on_message
async def main(message: cl.Message):
    """Handle incoming messages"""
    message_history = cl.user_session.get("message_history")
    message_history.append({"role": "user", "content": message.content})
    
    # Get all tools from all MCP connections
    mcp_tools = cl.user_session.get("mcp_tools", {})
    all_tools = [tool for connection_tools in mcp_tools.values() for tool in connection_tools]
    
    try:
        # Call OpenAI with or without tools
        params = {"model": "gpt-4o", "messages": message_history}
        if all_tools:
            params["tools"] = all_tools
            params["tool_choice"] = "auto"
        
        response = await client.chat.completions.create(**params)
        response_message = response.choices[0].message
        
        # Handle tool calls
        while response_message.tool_calls:
            message_history.append({
                "role": "assistant",
                "content": response_message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in response_message.tool_calls
                ]
            })
            
            # Execute each tool call
            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                await cl.Message(
                    content=f"🔧 Calling: `{function_name}`\n```json\n{json.dumps(function_args, indent=2)}\n```",
                    author="System"
                ).send()
                
                # Find which connection has this tool and call it
                mcp_connections = cl.user_session.get("mcp_connections", {})
                result = None
                
                for conn_name, connection in mcp_connections.items():
                    try:
                        result = await connection.call_tool(function_name, function_args)
                        break
                    except:
                        continue
                
                # Extract result text
                if result and hasattr(result, 'content'):
                    result_text = "\n".join([
                        item.text if hasattr(item, 'text') else str(item)
                        for item in result.content
                    ])
                else:
                    result_text = str(result) if result else "No result"
                
                message_history.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": result_text
                })
            
            # Get next response
            response = await client.chat.completions.create(
                model="gpt-4o",
                messages=message_history,
                tools=all_tools if all_tools else None
            )
            response_message = response.choices[0].message
        
        # Final response
        if response_message.content:
            message_history.append({"role": "assistant", "content": response_message.content})
            await cl.Message(content=response_message.content).send()
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        await cl.Message(content=f"❌ Error: {str(e)}").send()