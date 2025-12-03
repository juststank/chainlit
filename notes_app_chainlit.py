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
    """Handle MCP server connections - receives connection AND session"""
    print(f"🔌 MCP Connection event triggered for: {connection.name}")
    
    try:
        # List available tools from the MCP session
        tools_response = await session.list_tools()
        tools = tools_response.tools
        
        print(f"📋 Found {len(tools)} tools: {[tool.name for tool in tools]}")
        
        # Convert tools to OpenAI format
        openai_tools = [
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
        
        # Store tools by connection name in user session
        mcp_tools = cl.user_session.get("mcp_tools", {})
        mcp_tools[connection.name] = openai_tools
        cl.user_session.set("mcp_tools", mcp_tools)
        
        # Store the session for later tool calls
        mcp_sessions = cl.user_session.get("mcp_sessions", {})
        mcp_sessions[connection.name] = session
        cl.user_session.set("mcp_sessions", mcp_sessions)
        
        # Notify user
        tool_names = [tool.name for tool in tools]
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
    
    # Remove tools and session from user session
    mcp_tools = cl.user_session.get("mcp_tools", {})
    mcp_tools.pop(name, None)
    cl.user_session.set("mcp_tools", mcp_tools)
    
    mcp_sessions = cl.user_session.get("mcp_sessions", {})
    mcp_sessions.pop(name, None)
    cl.user_session.set("mcp_sessions", mcp_sessions)
    
    await cl.Message(
        content=f"❌ Disconnected from **{name}**",
        author="System"
    ).send()

@cl.on_message
async def main(message: cl.Message):
    """Handle incoming messages"""
    message_history = cl.user_session.get("message_history", [])
    message_history.append({"role": "user", "content": message.content})
    
    # Get all tools from all connected MCP servers
    mcp_tools = cl.user_session.get("mcp_tools", {})
    all_tools = []
    for connection_tools in mcp_tools.values():
        all_tools.extend(connection_tools)
    
    print(f"📊 Available tools: {len(all_tools)}")
    
    try:
        # Call OpenAI with or without tools
        params = {
            "model": "gpt-4o",
            "messages": message_history
        }
        
        if all_tools:
            params["tools"] = all_tools
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
            mcp_sessions = cl.user_session.get("mcp_sessions", {})
            
            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                print(f"🔧 Calling tool: {function_name} with args: {function_args}")
                
                await cl.Message(
                    content=f"🔧 Calling: `{function_name}`\n```json\n{json.dumps(function_args, indent=2)}\n```",
                    author="System"
                ).send()
                
                # Try to find and call the tool from any connected MCP session
                result_text = None
                for session_name, session in mcp_sessions.items():
                    try:
                        # Call the tool through the MCP session
                        result = await session.call_tool(function_name, function_args)
                        
                        # Extract text from result
                        if hasattr(result, 'content') and result.content:
                            result_text = "\n".join([
                                item.text if hasattr(item, 'text') else str(item)
                                for item in result.content
                            ])
                        else:
                            result_text = str(result)
                        
                        print(f"✅ Tool {function_name} executed successfully")
                        break
                        
                    except Exception as tool_error:
                        print(f"⚠️ Failed to call {function_name} on {session_name}: {tool_error}")
                        continue
                
                if result_text is None:
                    result_text = f"Error: Tool {function_name} not found in any connected MCP server"
                
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
                tools=all_tools if all_tools else None,
                tool_choice="auto" if all_tools else None
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