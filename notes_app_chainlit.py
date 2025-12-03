import chainlit as cl
from openai import AsyncOpenAI
import json
from dotenv import load_dotenv
import os
from fastmcp import Client

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Global variables
available_tools = []
MCP_SERVER_URL = "http://127.0.0.1:8002/mcp"

async def fetch_mcp_tools():
    """Fetch available tools from FastMCP server"""
    global available_tools
    
    try:
        print(f"🔌 Connecting to MCP server at {MCP_SERVER_URL}...")
        
        async with Client(MCP_SERVER_URL) as mcp_client:
            print(f"✅ Connected successfully")
            
            # List tools
            tools = await mcp_client.list_tools()
            print(f"📡 Found {len(tools)} tools")
            
            # Convert to OpenAI format (matching official MCP format)
            available_tools = []
            for tool in tools:
                openai_tool = {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema  # This matches the official inputSchema
                    }
                }
                available_tools.append(openai_tool)
            
            print(f"✅ Loaded {len(available_tools)} tools")
            print(f"📋 Tools: {[t['function']['name'] for t in available_tools]}")
            return True
                
    except Exception as e:
        print(f"❌ Error: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def call_mcp_tool(tool_name: str, tool_args: dict):
    """Execute MCP tool - similar to official client"""
    try:
        print(f"🔧 Calling tool {tool_name} with args {tool_args}")
        
        async with Client(MCP_SERVER_URL) as mcp_client:
            # Call tool - matches official: await self.session.call_tool(tool_name, tool_args)
            result = await mcp_client.call_tool(tool_name, tool_args)
            
            # Extract content - matches official: result.content
            if hasattr(result, 'content') and result.content:
                # Handle list of content items
                content_parts = []
                for item in result.content:
                    if hasattr(item, 'text'):
                        content_parts.append(item.text)
                    else:
                        content_parts.append(str(item))
                return "\n".join(content_parts)
            return str(result)
                
    except Exception as e:
        error_msg = f"Error calling tool {tool_name}: {str(e)}"
        print(f"❌ {error_msg}")
        return error_msg

@cl.on_chat_start
async def start():
    """Initialize the chat session"""
    success = await fetch_mcp_tools()
    
    cl.user_session.set("message_history", [
        {"role": "system", "content": "You are a helpful assistant with access to note-taking tools. Use the tools when appropriate to help users manage their notes."}
    ])
    
    if success and available_tools:
        tool_names = [t['function']['name'] for t in available_tools]
        await cl.Message(
            content=f"Hello! I'm connected to your notes system. 📝\n\nAvailable tools: **{', '.join(tool_names)}**\n\nI can help you retrieve and manage your notes!"
        ).send()
    else:
        await cl.Message(
            content=f"⚠️ Could not connect to MCP server at {MCP_SERVER_URL}"
        ).send()

@cl.on_message
async def main(message: cl.Message):
    """Handle incoming messages - improved based on official client"""
    message_history = cl.user_session.get("message_history")
    
    if not message_history:
        await cl.Message(content="⚠️ Session not initialized. Please refresh the page.").send()
        return
    
    if not available_tools:
        message_history.append({"role": "user", "content": message.content})
        response = await client.chat.completions.create(model="gpt-4o", messages=message_history)
        assistant_message = response.choices[0].message.content
        message_history.append({"role": "assistant", "content": assistant_message})
        await cl.Message(content=assistant_message).send()
        return
    
    message_history.append({"role": "user", "content": message.content})
    
    try:
        # Initial API call with tools
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=message_history,
            tools=available_tools,
            tool_choice="auto"
        )
        
        response_message = response.choices[0].message
        
        # Process tool calls (similar to official client loop)
        while response_message.tool_calls:
            # Add assistant's response with tool calls to history
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
            
            # Execute all tool calls
            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                # Show user which tool is being called
                await cl.Message(
                    content=f"🔧 Calling tool: `{function_name}`\n```json\n{json.dumps(function_args, indent=2)}\n```",
                    author="System"
                ).send()
                
                # Call the MCP tool
                tool_result = await call_mcp_tool(function_name, function_args)
                
                # Add tool result to history
                message_history.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": tool_result
                })
            
            # Get next response from OpenAI (may trigger more tool calls)
            response = await client.chat.completions.create(
                model="gpt-4o",
                messages=message_history,
                tools=available_tools,
                tool_choice="auto"
            )
            
            response_message = response.choices[0].message
        
        # Final response without tool calls
        if response_message.content:
            message_history.append({
                "role": "assistant",
                "content": response_message.content
            })
            await cl.Message(content=response_message.content).send()
            
    except Exception as e:
        error_msg = f"Error processing message: {type(e).__name__}: {str(e)}"
        print(f"❌ {error_msg}")
        import traceback
        traceback.print_exc()
        await cl.Message(content=f"❌ {error_msg}").send()