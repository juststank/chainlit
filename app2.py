# app.py
import os
import json
import asyncio
import chainlit as cl
from openai import OpenAI
import httpx
from typing import Optional, List, Dict
import warnings

# Suppress the httpcore warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

client = OpenAI()
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# MCP Server URL
MCP_SERVER_URL = "http://localhost:8000/mcp"

# Store MCP session globally
mcp_session = None
all_tools = []  # Cache all tools

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
                
                # Check if we have a complete message
                if "\n\n" in full_response:
                    parsed = parse_sse_response(full_response)
                    if parsed:
                        print(f"[DEBUG] Parsed: {list(parsed.keys())}")
                        
                        if "error" in parsed:
                            raise Exception(f"MCP Error: {parsed['error']}")
                        
                        if "result" in parsed:
                            return parsed["result"]
                        
                        full_response = ""
            
            # Try one more time with whatever we have
            if full_response:
                parsed = parse_sse_response(full_response)
                if parsed:
                    if "error" in parsed:
                        raise Exception(f"MCP Error: {parsed['error']}")
                    if "result" in parsed:
                        return parsed["result"]
            
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

def filter_relevant_tools(query: str, tools: List[dict], max_tools: int = 100) -> List[dict]:
    """Filter tools based on query relevance with category awareness"""
    query_lower = query.lower()
    keywords = query_lower.split()
    
    # Category-specific keywords mapping
    category_keywords = {
        'device': ['device', 'firmware', 'vdom', 'ha', 'hardware', 'model', 'cluster'],
        'policy': ['policy', 'firewall', 'rule', 'nat', 'snat', 'dnat', 'package'],
        'object': ['address', 'service', 'zone', 'vip', 'pool', 'schedule'],
        'provision': ['template', 'provision', 'profile', 'cli template', 'system template'],
        'monitor': ['monitor', 'status', 'log', 'statistic', 'health', 'task', 'connectivity'],
        'adom': ['adom', 'workspace', 'revision', 'lock', 'commit'],
        'security': ['web filter', 'ips', 'antivirus', 'dlp', 'application control', 'waf'],
        'vpn': ['vpn', 'ipsec', 'ssl-vpn', 'tunnel', 'phase1', 'phase2'],
        'sdwan': ['sd-wan', 'sdwan', 'wan', 'health check'],
        'fortiap': ['fortiap', 'wtp', 'wireless'],
        'fortiswitch': ['fortiswitch', 'switch'],
        'fortiextender': ['fortiextender', 'extender'],
        'connector': ['connector', 'fabric', 'aws', 'azure', 'vmware'],
        'script': ['script', 'cli script', 'execute'],
        'fortiguard': ['fortiguard', 'update', 'contract', 'threat'],
        'internet_service': ['internet service', 'cloud service'],
    }
    
    # Detect categories from query
    detected_categories = set()
    for category, category_kws in category_keywords.items():
        if any(kw in query_lower for kw in category_kws):
            detected_categories.add(category)
    
    # Score tools
    scored_tools = []
    for tool in tools:
        score = 0
        tool_name = tool.get("name", "").lower()
        tool_desc = tool.get("description", "").lower()
        
        # Category match bonus
        for category in detected_categories:
            if any(kw in tool_name for kw in category_keywords[category]):
                score += 15
            if any(kw in tool_desc for kw in category_keywords[category]):
                score += 5
        
        # Exact keyword match in name
        for keyword in keywords:
            if keyword in tool_name:
                score += 10
            if keyword in tool_desc:
                score += 3
        
        # Operation type matches
        operation_types = {
            'list': ['list', 'get', 'show', 'view'],
            'create': ['create', 'add', 'new'],
            'update': ['update', 'modify', 'edit', 'set'],
            'delete': ['delete', 'remove'],
            'install': ['install', 'deploy', 'push'],
            'execute': ['execute', 'run'],
        }
        
        for op_type, op_keywords in operation_types.items():
            if any(op in query_lower for op in op_keywords):
                if any(op in tool_name for op in op_keywords):
                    score += 8
        
        # Specific entity matches with high priority
        high_priority_entities = [
            'device', 'policy', 'firewall', 'address', 'service', 
            'adom', 'vdom', 'template', 'vpn', 'sdwan'
        ]
        
        for entity in high_priority_entities:
            if entity in query_lower and entity in tool_name:
                score += 12
        
        if score > 0:
            scored_tools.append((score, tool))
    
    # Sort by score descending
    scored_tools.sort(reverse=True, key=lambda x: x[0])
    
    # If no scored tools, return common list/get operations
    if not scored_tools:
        default_tools = [
            t for t in tools 
            if any(op in t.get("name", "").lower() for op in ['list', 'get'])
        ]
        return default_tools[:max_tools]
    
    # Return top scored tools
    return [tool for score, tool in scored_tools[:max_tools]]

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
    global mcp_session, all_tools
    
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
            all_tools = await asyncio.wait_for(mcp_session.list_tools(), timeout=15.0)
            
            if all_tools:
                tool_names = [tool.get("name", "unknown") for tool in all_tools]
                
                message = f"✅ **Connected to FortiManager MCP Server!**\n\n"
                message += f"**Total tools available: {len(tool_names)}**\n\n"
                
                # Categorize tools
                categories = {
                    'Device Management': [t for t in tool_names if any(k in t.lower() for k in ['device', 'vdom', 'ha', 'firmware'])],
                    'Policy Management': [t for t in tool_names if any(k in t.lower() for k in ['policy', 'firewall', 'nat'])],
                    'Objects': [t for t in tool_names if any(k in t.lower() for k in ['address', 'service']) and 'internet' not in t.lower()],
                    'Provisioning': [t for t in tool_names if any(k in t.lower() for k in ['template', 'provision'])],
                    'Security Profiles': [t for t in tool_names if any(k in t.lower() for k in ['webfilter', 'ips', 'antivirus', 'dlp', 'application'])],
                    'VPN': [t for t in tool_names if 'vpn' in t.lower() or 'ipsec' in t.lower()],
                    'SD-WAN': [t for t in tool_names if 'sdwan' in t.lower() or 'sd_wan' in t.lower()],
                    'ADOM': [t for t in tool_names if 'adom' in t.lower()],
                    'Monitoring': [t for t in tool_names if any(k in t.lower() for k in ['monitor', 'status', 'log', 'statistic'])],
                    'Internet Services': [t for t in tool_names if 'internet_service' in t.lower()],
                }
                
                message += "**Tools by category:**\n"
                for category, tools in categories.items():
                    if tools:
                        message += f"• **{category}:** {len(tools)} tools\n"
                
                other_count = len(tool_names) - sum(len(tools) for tools in categories.values())
                if other_count > 0:
                    message += f"• **Other:** {other_count} tools\n"
                
                message += "\n**Example queries:**\n"
                message += "• List all FortiGate devices\n"
                message += "• Show firewall policies in the production package\n"
                message += "• List all ADOMs\n"
                message += "• Get device status\n"
                message += "• List internet services\n"
                message += "• Show VPN tunnels\n"
                message += "• Create an address group\n\n"
                message += "*💡 Tools are intelligently filtered based on your query*"
                
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
        await cl.Message(content="❌ Failed to connect. Check terminal logs.").send()

@cl.on_message
async def on_message(message: cl.Message):
    global mcp_session, all_tools
    
    if not mcp_session:
        await cl.Message(content="❌ Not connected to MCP server. Please restart the chat.").send()
        return
    
    try:
        # Filter tools based on query
        relevant_tools = filter_relevant_tools(message.content, all_tools, max_tools=100)
        
        print(f"[DEBUG] Filtered to {len(relevant_tools)} relevant tools from {len(all_tools)} total")
        if relevant_tools:
            top_tools = [t.get("name") for t in relevant_tools[:5]]
            print(f"[DEBUG] Top 5 tools: {top_tools}")
        
        # Convert to OpenAI format
        openai_tools = []
        for tool in relevant_tools:
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", "")[:1000],  # Truncate very long descriptions
                    "parameters": tool.get("inputSchema", {})
                }
            })
        
        # Initial API call
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant with access to FortiManager tools. "
                    "Use them to help manage and monitor FortiGate devices, policies, objects, and more. "
                    "When listing items, present the information in a clear, organized format. "
                    "For complex data, use tables or bullet points."
                )
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