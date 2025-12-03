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
        content="Hello! 👋\n\nConnect to MCP servers using the **plug icon (🔌)** in the sidebar to enable tools.\n\nOnce connected, I can help you manage your notes!"
    ).send()

@cl.on_mcp_connect
async def on_mcp_connect(connection, session):
    """Handle MCP server connections"""
    print(f"🔌 MCP Connection event triggered for: {connection.name}")
    
    try:
        # List available tools
        result = await session.list_tools()
        
        # Process tool metadata
        tools = [{
            "name": t.name,
            "description": t.description,
            "input_schema": t.inputSchema,
        } for t in result.tools]
        
        print(f"📋 Found {len(tools)} tools: {[t['name'] for t in tools]}")
        
        # Store tools for later use
        mcp_tools = cl.user_session.get("mcp_tools", {})
        mcp_tools[connection.name] = tools
        cl.user_session.set("mcp_tools", mcp_tools)
        
        # Notify user
        tool_names = [t['name'] for t in tools]
        await cl.Message(
            content=f"✅ **Connected to {connection.name}**\n\n🔧 Available tools:\n" + "\n".join([f"- `{name}`" for name in tool_names]),
            author="System"
        ).send()
        
    except Exception as e:
        print(f"❌ Error in on_mcp_connect: {str(e)}")
        import traceback
        traceback.print_exc()
        await cl.Message(
            content=f"❌ Failed to connect to {connection.name}: {str(e)}",
            author="System"
        ).send()

@cl.on_mcp_disconnect
async def on_mcp_disconnect(name: str):
    """Handle MCP server disconnections"""
    print(f"🔌 MCP server '{name}' disconnected")
    
    # Remove tools from user session
    mcp_tools = cl.user_session.get("mcp_tools", {})
    mcp_tools.pop(name, None)
    cl.user_session.set("mcp_tools", mcp_tools)
    
    await cl.Message(
        content=f"❌ Disconnected from **{name}**",
        author="System"
    ).send()

def find_mcp_for_tool(tool_name: str) -> str:
    """Find which MCP connection has the given tool"""
    mcp_tools = cl.user_session.get("mcp_tools", {})
    
    for connection_name, tools in mcp_tools.items():
        if any(t['name'] == tool_name for t in tools):
            return connection_name
    
    raise ValueError(f"Tool {tool_name} not found in any MCP connection")

@cl.step(type="tool")
async def call_mcp_tool(tool_name: str, tool_input: dict):
    """Execute MCP tool - following Chainlit documentation pattern"""
    try:
        # Find appropriate MCP connection for this tool
        mcp_name = find_mcp_for_tool(tool_name)
        
        # Get the MCP session from context
        mcp_session, _ = cl.context.session.mcp_sessions.get(mcp_name)
        
        print(f"🔧 Calling tool {tool_name} on connection {mcp_name}")
        
        # Call the tool
        result = await mcp_session.call_tool(tool_name, tool_input)
        
        # Extract text from result
        if hasattr(result, 'content') and result.content:
            result_text = "\n".join([
                item.text if hasattr(item, 'text') else str(item)
                for item in result.content
            ])
        else:
            result_text = str(result)
        
        print(f"✅ Tool {tool_name} executed successfully")
        return result_text
        
    except Exception as e:
        error_msg = f"Error calling tool {tool_name}: {str(e)}"
        print(f"❌ {error_msg}")
        import traceback
        traceback.print_exc()
        return error_msg

@cl.on_message
async def main(message: cl.Message):
    """Handle incoming messages"""
    message_history = cl.user_session.get("message_history", [])
    message_history.append({"role": "user", "content": message.content})
    
    # Get tools from all MCP connections (following documentation)
    mcp_tools = cl.user_session.get("mcp_tools", {})
    all_tools = [tool for connection_tools in mcp_tools.values() for tool in connection_tools]
    
    # Convert to OpenAI format
    openai_tools = [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["input_schema"]
            }
        }
        for tool in all_tools
    ]
    
    print(f"📊 Available tools: {len(openai_tools)}")
    
    try:
        # Call OpenAI with or without tools
        params = {
            "model": "gpt-4o",
            "messages": message_history
        }
        
        if openai_tools:
            params["tools"] = openai_tools
            params["tool_choice"] = "auto"
        
        response = await client.chat.completions.create(**params)
        response_message = response.choices[0].message
        
        # Handle tool calls in a loop
        while response_message.tool_calls:
            # Add assistant message with tool calls to history
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
                
                # Call the MCP tool using the documented pattern
                result_text = await call_mcp_tool(function_name, function_args)
                
                # Add tool result to history
                message_history.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": result_text
                })
            
            # Get next response from OpenAI
            response = await client.chat.completions.create(
                model="gpt-4o",
                messages=message_history,
                tools=openai_tools if openai_tools else None,
                tool_choice="auto" if openai_tools else None
            )
            response_message = response.choices[0].message
        
        # Final response without tool calls
        if response_message.content:
            message_history.append({
                "role": "assistant",
                "content": response_message.content
            })
            await cl.Message(content=response_message.content).send()
        
        # Update session history
        cl.user_session.set("message_history", message_history)
            
    except Exception as e:
        error_msg = f"Error: {type(e).__name__}: {str(e)}"
        print(f"❌ {error_msg}")
        import traceback
        traceback.print_exc()
        await cl.Message(content=f"❌ {error_msg}").send()