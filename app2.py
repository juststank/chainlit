# app.py
import os
import json
import asyncio
import chainlit as cl
from openai import OpenAI

client = OpenAI()
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# MCP Server URL
MCP_SERVER_URL = "http://10.75.11.84:8000/mcp"

# Store MCP session globally
mcp_session = None
mcp_exit_stack = None

async def init_mcp_session():
    """Initialize MCP session with proper imports"""
    global mcp_exit_stack
    
    try:
        from mcp import ClientSession
        from mcp.client.sse import sse_client
        import contextlib
        
        print(f"Connecting to {MCP_SERVER_URL}...")
        
        # Create an exit stack to manage contexts
        exit_stack = contextlib.AsyncExitStack()
        mcp_exit_stack = exit_stack
        
        try:
            # Enter the SSE client context
            read_stream, write_stream = await exit_stack.enter_async_context(
                sse_client(url=MCP_SERVER_URL)
            )
            
            print("SSE streams established")
            
            # Enter the ClientSession context
            session = await exit_stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            
            print("ClientSession created, initializing...")
            
            # Initialize the session
            result = await asyncio.wait_for(session.initialize(), timeout=10.0)
            
            print(f"Session initialized! Server info: {result}")
            return session
            
        except asyncio.TimeoutError:
            print("Timeout during initialization")
            await exit_stack.aclose()
            return None
        except Exception as e:
            print(f"Error during connection: {e}")
            await exit_stack.aclose()
            raise
            
    except ImportError as e:
        print(f"Missing package: {e}")
        print("Install with: pip install mcp httpx-sse")
        return None
    except Exception as e:
        print(f"Connection error: {e}")
        import traceback
        traceback.print_exc()
        return None

@cl.on_chat_start
async def start():
    """Initialize MCP connection when chat starts"""
    global mcp_session
    
    await cl.Message(
        content="🔄 **Connecting to FortiManager MCP server...**\n\n*This may take 10-15 seconds*"
    ).send()
    
    # Try to connect with overall timeout
    try:
        mcp_session = await asyncio.wait_for(init_mcp_session(), timeout=30.0)
    except asyncio.TimeoutError:
        await cl.Message(
            content="❌ Connection timeout after 30 seconds.\n\nPlease check if the MCP server is responding properly."
        ).send()
        return
    
    if mcp_session:
        try:
            # List tools
            print("Listing tools...")
            tools_result = await asyncio.wait_for(mcp_session.list_tools(), timeout=10.0)
            
            if hasattr(tools_result, 'tools') and tools_result.tools:
                tool_names = [tool.name for tool in tools_result.tools]
                
                message = f"✅ **Connected to FortiManager MCP!**\n\n"
                message += f"**Available tools ({len(tool_names)}):**\n\n"
                
                # Group tools by category
                device_tools = [t for t in tool_names if 'device' in t.lower()]
                policy_tools = [t for t in tool_names if 'policy' in t.lower()]
                object_tools = [t for t in tool_names if 'object' in t.lower()]
                monitoring_tools = [t for t in tool_names if 'monitor' in t.lower()]
                other_tools = [t for t in tool_names if t not in device_tools + policy_tools + object_tools + monitoring_tools]
                
                if device_tools:
                    message += "**Device Tools:**\n" + "\n".join([f"• `{t}`" for t in device_tools[:5]]) + "\n\n"
                if policy_tools:
                    message += "**Policy Tools:**\n" + "\n".join([f"• `{t}`" for t in policy_tools[:5]]) + "\n\n"
                if object_tools:
                    message += "**Object Tools:**\n" + "\n".join([f"• `{t}`" for t in object_tools[:5]]) + "\n\n"
                if monitoring_tools:
                    message += "**Monitoring Tools:**\n" + "\n".join([f"• `{t}`" for t in monitoring_tools[:5]]) + "\n\n"
                if other_tools:
                    message += "**Other Tools:**\n" + "\n".join([f"• `{t}`" for t in other_tools[:5]]) + "\n\n"
                
                total_shown = len(device_tools[:5]) + len(policy_tools[:5]) + len(object_tools[:5]) + len(monitoring_tools[:5]) + len(other_tools[:5])
                if len(tool_names) > total_shown:
                    message += f"*...and {len(tool_names) - total_shown} more tools*\n\n"
                
                message += "You can now ask me to manage your FortiGate devices!"
                
                await cl.Message(content=message).send()
            else:
                await cl.Message(content="✅ Connected but no tools found.").send()
                
        except asyncio.TimeoutError:
            await cl.Message(content="⚠️ Connected but timeout listing tools.").send()
        except Exception as e:
            await cl.Message(content=f"⚠️ Connected but error listing tools: {str(e)}").send()
            import traceback
            traceback.print_exc()
    else:
        error_msg = f"❌ **Failed to connect to FortiManager MCP**\n\n"
        error_msg += "Please check:\n"
        error_msg += f"• Server is running at `{MCP_SERVER_URL}`\n"
        error_msg += "• Packages installed: `pip install mcp httpx-sse`\n"
        error_msg += "• Check terminal for detailed error logs\n\n"
        error_msg += "*You can still chat, but FortiManager tools won't be available.*"
        
        await cl.Message(content=error_msg).send()

@cl.on_message
async def on_message(message: cl.Message):
    global mcp_session
    
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
            system_content += " with access to FortiManager tools. Use them to help manage and monitor FortiGate devices, policies, objects, and more."
        
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
                    content=f"🔧 Calling: **{tool_name}**\n```json\n{json.dumps(tool_args, indent=2)}\n```"
                ).send()
                
                try:
                    # Call MCP tool with timeout
                    result = await asyncio.wait_for(
                        mcp_session.call_tool(tool_name, tool_args),
                        timeout=30.0
                    )
                    
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
                    
                except asyncio.TimeoutError:
                    error_msg = f"Timeout calling tool {tool_name}"
                    await cl.Message(content=f"⚠️ {error_msg}").send()
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": error_msg
                    })
                except Exception as e:
                    error_msg = f"Error calling tool {tool_name}: {str(e)}"
                    await cl.Message(content=f"⚠️ {error_msg}").send()
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": error_msg
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
    global mcp_exit_stack
    if mcp_exit_stack:
        try:
            await mcp_exit_stack.aclose()
            print("MCP connection closed")
        except Exception as e:
            print(f"Error closing connection: {e}")