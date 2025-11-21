# app.py
import os
import json
import asyncio
import chainlit as cl
from openai import OpenAI
import httpx
from typing import Optional
import warnings

# Suppress the httpcore warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

client = OpenAI()
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# MCP Server URL
MCP_SERVER_URL = "http://localhost:8000/mcp"

# Store MCP session globally
mcp_session = None

def parse_sse_response(text: str) -> Optional[dict]:
    """Parse SSE formatted response"""
    lines = text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if line.startswith('data: '):
            data_json = line[6:]  # Remove 'data: ' prefix
            try:
                return json.loads(data_json)
            except json.JSONDecodeError as e:
                print(f"[ERROR] Failed to parse JSON: {e}")
                print(f"[ERROR] JSON string: {data_json[:200]}")
                return None
    
    return None

class MCPClient:
    """MCP client using httpx for SSE communication"""
    
    def __init__(self, url: str):
        self.url = url
        self.request_id = 0
        self.client = httpx.AsyncClient(timeout=30.0)
        
    async def send_request(self, method: str, params: dict = None) -> dict:
        """Send a JSON-RPC request and parse SSE response"""
        self.request_id += 1
        
        payload = {
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": method,
        }
        
        if params:
            payload["params"] = params
        
        print(f"[DEBUG] Sending: {method}")
        
        async with self.client.stream(
            "POST",
            self.url,
            json=payload,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream"
            }
        ) as response:
            if response.status_code != 200:
                text = await response.aread()
                raise Exception(f"HTTP {response.status_code}: {text.decode()}")
            
            # Collect the entire response
            full_response = ""
            async for chunk in response.aiter_text():
                full_response += chunk
                
                # Check if we have a complete message (ends with double newline)
                if "\n\n" in full_response:
                    # Try to parse
                    parsed = parse_sse_response(full_response)
                    if parsed:
                        print(f"[DEBUG] Parsed: {list(parsed.keys())}")
                        
                        if "error" in parsed:
                            raise Exception(f"MCP Error: {parsed['error']}")
                        
                        if "result" in parsed:
                            return parsed["result"]
                        
                        # If no result yet, keep collecting
                        full_response = ""
            
            # Try one more time with whatever we have
            if full_response:
                parsed = parse_sse_response(full_response)
                if parsed:
                    if "error" in parsed:
                        raise Exception(f"MCP Error: {parsed['error']}")
                    if "result" in parsed:
                        return parsed["result"]
            
            print(f"[DEBUG] Full response: {full_response[:500]}")
            raise Exception("No valid response received")
    
    async def initialize(self):
        """Initialize the MCP session"""
        return await self.send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "chainlit-client",
                "version": "1.0.0"
            }
        })
    
    async def list_tools(self):
        """List available tools"""
        result = await self.send_request("tools/list")
        return result.get("tools", [])
    
    async def call_tool(self, name: str, arguments: dict):
        """Call a tool"""
        return await self.send_request("tools/call", {
            "name": name,
            "arguments": arguments
        })
    
    async def close(self):
        """Close the client"""
        await self.client.aclose()

async def init_mcp_session() -> Optional[MCPClient]:
    """Initialize MCP session"""
    try:
        print(f"[DEBUG] Connecting to {MCP_SERVER_URL}...")
        
        mcp = MCPClient(MCP_SERVER_URL)
        
        # Initialize
        print("[DEBUG] Initializing MCP session...")
        init_result = await mcp.initialize()
        server_name = init_result.get('serverInfo', {}).get('name', 'Unknown')
        server_version = init_result.get('serverInfo', {}).get('version', 'Unknown')
        print(f"[DEBUG] Connected to: {server_name} v{server_version}")
        
        return mcp
        
    except Exception as e:
        print(f"[ERROR] Connection error: {e}")
        import traceback
        traceback.print_exc()
        return None

@cl.on_chat_start
async def start():
    """Initialize MCP connection when chat starts"""
    global mcp_session
    
    # Check OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or "REPLACE" in api_key:
        await cl.Message(
            content="❌ **OpenAI API Key not configured!**\n\n"
                    "Please set your OpenAI API key in `.env` file:\n"
                    "```\nOPENAI_API_KEY=sk-your-actual-key-here\n```"
        ).send()
        return
    
    await cl.Message(
        content="🔄 **Connecting to FortiManager MCP server...**"
    ).send()
    
    try:
        mcp_session = await asyncio.wait_for(init_mcp_session(), timeout=15.0)
    except asyncio.TimeoutError:
        await cl.Message(content="❌ Connection timeout").send()
        return
    
    if mcp_session:
        try:
            # List tools
            print("[DEBUG] Listing tools...")
            tools = await asyncio.wait_for(mcp_session.list_tools(), timeout=15.0)
            
            if tools:
                tool_names = [tool.get("name", "unknown") for tool in tools]
                
                message = f"✅ **Connected to FortiManager MCP!**\n\n"
                message += f"**Available tools ({len(tool_names)}):**\n\n"
                
                # Group by category
                device_tools = [t for t in tool_names if 'device' in t.lower()]
                policy_tools = [t for t in tool_names if 'policy' in t.lower()]
                object_tools = [t for t in tool_names if 'object' in t.lower() or 'address' in t.lower()]
                monitor_tools = [t for t in tool_names if 'monitor' in t.lower() or 'status' in t.lower()]
                service_tools = [t for t in tool_names if 'service' in t.lower() and t not in object_tools]
                other_tools = [t for t in tool_names if t not in device_tools + policy_tools + object_tools + monitor_tools + service_tools]
                
                if device_tools:
                    message += "**Device Management:**\n" + "\n".join([f"• `{t}`" for t in device_tools[:5]]) + "\n\n"
                if policy_tools:
                    message += "**Policy Management:**\n" + "\n".join([f"• `{t}`" for t in policy_tools[:5]]) + "\n\n"
                if object_tools:
                    message += "**Object Management:**\n" + "\n".join([f"• `{t}`" for t in object_tools[:5]]) + "\n\n"
                if service_tools:
                    message += "**Service Management:**\n" + "\n".join([f"• `{t}`" for t in service_tools[:5]]) + "\n\n"
                if monitor_tools:
                    message += "**Monitoring:**\n" + "\n".join([f"• `{t}`" for t in monitor_tools[:5]]) + "\n\n"
                
                shown = (len(device_tools[:5]) + len(policy_tools[:5]) + len(object_tools[:5]) + 
                        len(service_tools[:5]) + len(monitor_tools[:5]))
                if len(tool_names) > shown:
                    message += f"*...and {len(tool_names) - shown} more tools*\n\n"
                
                message += "**Try asking:**\n"
                message += "• List all FortiGate devices\n"
                message += "• Show firewall policies\n"
                message += "• List internet services\n"
                
                await cl.Message(content=message).send()
            else:
                await cl.Message(content="✅ Connected but no tools found.").send()
                
        except asyncio.TimeoutError:
            await cl.Message(content="⚠️ Connected but timeout listing tools (response too large).").send()
        except Exception as e:
            await cl.Message(content=f"⚠️ Connected but error listing tools: {str(e)}").send()
            print(f"[ERROR] Error listing tools:")
            import traceback
            traceback.print_exc()
    else:
        await cl.Message(content="❌ Failed to connect. Check terminal logs.").send()

@cl.on_message
async def on_message(message: cl.Message):
    global mcp_session
    
    if not mcp_session:
        await cl.Message(content="❌ Not connected to MCP server. Please restart the chat.").send()
        return
    
    try:
        # Get available tools
        tools = await mcp_session.list_tools()
        
        # Convert MCP tools to OpenAI format
        openai_tools = []
        for tool in tools:
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("inputSchema", {})
                }
            })
        
        # Initial API call
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant with access to FortiManager tools. Use them to help manage and monitor FortiGate devices, policies, objects, and more."
            },
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
                    # Call MCP tool
                    result = await mcp_session.call_tool(tool_name, tool_args)
                    
                    # Format result
                    if isinstance(result, dict):
                        # Check for content field (MCP format)
                        if "content" in result:
                            content = result["content"]
                            if isinstance(content, list) and content:
                                tool_response = content[0].get("text", str(content))
                            else:
                                tool_response = str(content)
                        else:
                            tool_response = json.dumps(result, indent=2)
                    else:
                        tool_response = str(result)
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_response
                    })
                    
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
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
    global mcp_session
    if mcp_session:
        try:
            await mcp_session.close()
            print("[INFO] MCP connection closed")
        except Exception as e:
            print(f"[ERROR] Error closing connection: {e}")