import os
import json
import asyncio
import chainlit as cl
from openai import AsyncOpenAI
import httpx
from typing import Optional, List
import warnings
import traceback

# Suppress httpcore warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Configuration
MCP_SERVER_URL = "http://localhost:8000/mcp"
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

# Initialize Async Client (Fix: Non-blocking)
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- CUSTOM MCP CLIENT (Preserved from your working code) ---

def parse_sse_response(text: str) -> Optional[dict]:
    """Parse SSE formatted response"""
    lines = text.strip().split('\n')
    for line in lines:
        line = line.strip()
        if line.startswith('data: '):
            data_json = line[6:]
            try:
                return json.loads(data_json)
            except json.JSONDecodeError as e:
                print(f"[ERROR] Failed to parse JSON: {e}")
                return None
    return None

class MCPClient:
    """Custom MCP client using httpx for SSE communication"""
    def __init__(self, url: str):
        self.url = url
        self.request_id = 0
        # Increased timeout to 60s for safety
        self.client = httpx.AsyncClient(timeout=60.0)
        
    async def send_request(self, method: str, params: dict = None) -> dict:
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
            
            full_response = ""
            async for chunk in response.aiter_text():
                full_response += chunk
                if "\n\n" in full_response:
                    parsed = parse_sse_response(full_response)
                    if parsed:
                        if "error" in parsed:
                            raise Exception(f"MCP Error: {parsed['error']}")
                        if "result" in parsed:
                            return parsed["result"]
                        full_response = ""
            
            # Fallback for remaining buffer
            if full_response:
                parsed = parse_sse_response(full_response)
                if parsed:
                    if "error" in parsed:
                        raise Exception(f"MCP Error: {parsed['error']}")
                    if "result" in parsed:
                        return parsed["result"]
            
            raise Exception("No valid response received")
    
    async def initialize(self):
        return await self.send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "chainlit-client", "version": "1.0.0"}
        })
    
    async def list_tools(self):
        result = await self.send_request("tools/list")
        return result.get("tools", [])
    
    async def call_tool(self, name: str, arguments: dict):
        return await self.send_request("tools/call", {
            "name": name,
            "arguments": arguments
        })
    
    async def close(self):
        await self.client.aclose()

# --- TOOL UTILITIES ---

def filter_relevant_tools(query: str, tools: List[dict], max_tools: int = 20) -> List[dict]:
    """Filter tools based on query relevance"""
    if not tools:
        return []

    query_lower = query.lower()
    keywords = query_lower.split()
    
    scored_tools = []
    for tool in tools:
        score = 0
        tool_name = tool.get("name", "").lower()
        tool_desc = tool.get("description", "").lower()
        
        if any(keyword in tool_name for keyword in keywords): score += 10
        if any(keyword in tool_desc for keyword in keywords): score += 5
        
        # Boost specific categories
        if "list" in query_lower and "list" in tool_name: score += 3
        if "create" in query_lower and "create" in tool_name: score += 3
        
        # Boost FortiManager entities
        entities = ["device", "policy", "address", "service", "firewall"]
        for entity in entities:
             if entity in query_lower and entity in tool_name:
                 score += 8
        
        if score > 0:
            scored_tools.append((score, tool))
    
    scored_tools.sort(reverse=True, key=lambda x: x[0])
    
    if not scored_tools:
        # Fallback: prefer list tools if no match
        default_tools = [t for t in tools if "list" in t.get("name", "").lower()]
        return default_tools[:max_tools]
    
    return [tool for score, tool in scored_tools[:max_tools]]

async def init_mcp_session() -> Optional[MCPClient]:
    try:
        print(f"[DEBUG] Connecting to {MCP_SERVER_URL}...")
        mcp = MCPClient(MCP_SERVER_URL)
        print("[DEBUG] Initializing MCP session...")
        await mcp.initialize()
        return mcp
    except Exception as e:
        print(f"[ERROR] Connection error: {e}")
        return None

# --- CHAINLIT HANDLERS ---

@cl.on_chat_start
async def start():
    """Initialize session and connection"""
    
    # 1. Check API Key
    if not os.getenv("OPENAI_API_KEY"):
        await cl.Message("❌ **OpenAI API Key not configured!**").send()
        return
    
    # 2. Initialize Message History
    cl.user_session.set("messages", [
        {"role": "system", "content": "You are a helpful assistant with access to FortiManager tools. Use them to help manage and monitor FortiGate devices."}
    ])

    # 3. Connect to MCP (Session Specific)
    msg = await cl.Message(content="🔄 **Connecting to FortiManager MCP...**").send()
    
    try:
        # Fix: Store in user_session, NOT global
        mcp_session = await asyncio.wait_for(init_mcp_session(), timeout=15.0)
        
        if mcp_session:
            cl.user_session.set("mcp_session", mcp_session)
            
            # List Tools
            print("[DEBUG] Listing tools...")
            all_tools = await asyncio.wait_for(mcp_session.list_tools(), timeout=15.0)
            cl.user_session.set("all_tools", all_tools)
            
            tool_names = [t.get("name", "") for t in all_tools]
            
            # Status Update
            status_text = f"✅ **Connected!**\n\n**Tools Available:** {len(tool_names)}\n"
            status_text += f"- Device Tools: {len([t for t in tool_names if 'device' in t])}\n"
            status_text += f"- Policy Tools: {len([t for t in tool_names if 'policy' in t])}"
            
            await msg.update(content=status_text)
        else:
            await msg.update(content="❌ Failed to connect to MCP server.")
            
    except asyncio.TimeoutError:
        await msg.update(content="❌ Connection timeout.")

@cl.on_message
async def on_message(message: cl.Message):
    # 1. Retrieve Session Data
    mcp_session = cl.user_session.get("mcp_session")
    all_tools = cl.user_session.get("all_tools", [])
    messages = cl.user_session.get("messages", [])

    if not mcp_session:
        await cl.Message("❌ Not connected to MCP. Please restart chat.").send()
        return

    # 2. Filter Tools
    relevant_tools = filter_relevant_tools(message.content, all_tools, max_tools=20)
    
    # 3. Format for OpenAI
    openai_tools = []
    for tool in relevant_tools:
        openai_tools.append({
            "type": "function",
            "function": {
                "name": tool.get("name"),
                "description": tool.get("description", "")[:500],
                "parameters": tool.get("inputSchema", {})
            }
        })

    # 4. Add User Message to History
    messages.append({"role": "user", "content": message.content})
    
    try:
        # 5. Call OpenAI (Async)
        response = await client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=openai_tools if openai_tools else None,
            tool_choice="auto" if openai_tools else "none",
            temperature=0.2
        )
        
        assistant_msg = response.choices[0].message
        
        # 6. Handle Tool Calls Loop
        iteration = 0
        while assistant_msg.tool_calls and iteration < 5:
            iteration += 1
            
            # Append Thought to History
            messages.append(assistant_msg)
            
            for tool_call in assistant_msg.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)
                
                # UI Step
                async with cl.Step(name=tool_name, type="tool") as step:
                    step.input = tool_args
                    
                    try:
                        # Execute Tool
                        result = await mcp_session.call_tool(tool_name, tool_args)
                        
                        # Parse Content
                        if isinstance(result, dict) and "content" in result:
                            c = result["content"]
                            output = str(c[0].get("text", c)) if isinstance(c, list) else str(c)
                        else:
                            output = json.dumps(result, indent=2)
                            
                        step.output = output
                    except Exception as e:
                        output = f"Error: {str(e)}"
                        step.output = output
                    
                    # Append Result to History
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": output
                    })
            
            # Get Follow-up Response
            response = await client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=openai_tools if openai_tools else None,
                temperature=0.2
            )
            assistant_msg = response.choices[0].message
            
        # 7. Final Answer
        final_answer = assistant_msg.content or ""
        await cl.Message(content=final_answer).send()
        
        # Update History
        messages.append({"role": "assistant", "content": final_answer})
        cl.user_session.set("messages", messages)

    except Exception as e:
        await cl.Message(f"❌ Error: {str(e)}").send()
        traceback.print_exc()

@cl.on_chat_end
async def end():
    mcp = cl.user_session.get("mcp_session")
    if mcp:
        await mcp.close()