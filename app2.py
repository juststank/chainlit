# app.py
import os
import json
import chainlit as cl
from openai import OpenAI
import httpx

client = OpenAI()
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# MCP Server URL
MCP_SERVER_URL = "http://localhost:8000"

async def call_mcp_tool(tool_name: str, arguments: dict):
    """Call MCP tool via HTTP"""
    async with httpx.AsyncClient() as http_client:
        response = await http_client.post(
            f"{MCP_SERVER_URL}/call_tool",
            json={
                "name": tool_name,
                "arguments": arguments
            },
            timeout=30.0
        )
        return response.json()

async def list_mcp_tools():
    """List available MCP tools"""
    async with httpx.AsyncClient() as http_client:
        response = await http_client.get(
            f"{MCP_SERVER_URL}/list_tools",
            timeout=10.0
        )
        return response.json()

@cl.on_chat_start
async def start():
    """Test MCP connection when chat starts"""
    try:
        tools = await list_mcp_tools()
        tool_names = [tool["name"] for tool in tools.get("tools", [])]
        
        await cl.Message(
            content=f"✅ Connected to FortiManager MCP server at {MCP_SERVER_URL}!\n\n**Available tools:**\n" + 
                    "\n".join([f"- {name}" for name in tool_names])
        ).send()
    except Exception as e:
        await cl.Message(
            content=f"❌ Failed to connect to MCP server: {str(e)}\n\nMake sure the server is running at {MCP_SERVER_URL}"
        ).send()
        import traceback
        print(traceback.format_exc())

@cl.on_message
async def on_message(message: cl.Message):
    try:
        # Get available tools
        tools_response = await list_mcp_tools()
        tools = tools_response.get("tools", [])
        
        # Convert to OpenAI format
        openai_tools = []
        for tool in tools:
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("inputSchema", {})
                }
            })
        
        # Initial API call
        messages = [
            {"role": "system", "content": "You are a helpful assistant with access to FortiManager tools. Use them to help manage and monitor FortiGate devices."},
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
            
            # Convert to dict for JSON serialization
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
                    content=f"🔧 Calling tool: `{tool_name}`\n```json\n{json.dumps(tool_args, indent=2)}\n```"
                ).send()
                
                # Call MCP tool
                result = await call_mcp_tool(tool_name, tool_args)
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result)
                })
            
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
        print(traceback.format_exc())