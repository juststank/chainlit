# app.py
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║          FortiManager MCP Integration - Chainlit + OpenAI Client            ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

Conversational Interface for FortiManager Management via MCP Protocol

OVERVIEW
--------
This application provides natural language access to FortiManager operations by
integrating three key components:

1. **FortiManager MCP Server** (HTTP/SSE transport)
   - 590 MCP tools covering 100% of FortiManager 7.4.8 API
   - 555 documented operations across 24 functional categories
   - Production-ready with zero linter errors and full type coverage

2. **OpenAI API** (GPT-4o-mini)
   - Natural language understanding and intent detection
   - Function calling for tool selection and orchestration
   - Multi-turn conversation support

3. **Chainlit Web Interface**
   - Browser-based chat interface
   - Real-time message streaming
   - Session management and lifecycle handling

VERSION: 1.0.0
"""

import os
import json
import asyncio
from typing import Optional, List, Dict, Tuple
import warnings

import chainlit as cl
from openai import OpenAI
import httpx

# Suppress httpcore async generator warnings from SSE streaming
warnings.filterwarnings('ignore', category=RuntimeWarning)

# ============================================================================
# Configuration
# ============================================================================

# OpenAI Configuration
client = OpenAI()  # Reads OPENAI_API_KEY from environment
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# MCP Server Configuration
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8000/mcp")

# Global State
mcp_session: Optional['MCPClient'] = None
all_tools: List[Dict] = []


# ============================================================================
# MCP Protocol Implementation
# ============================================================================

def parse_sse_response(text: str) -> Optional[dict]:
    """Parse Server-Sent Events (SSE) formatted response from MCP server."""
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
    """HTTP/SSE client for Model Context Protocol communication."""
    
    def __init__(self, url: str):
        self.url = url
        self.request_id = 0
        self.client = httpx.AsyncClient(timeout=30.0)
        
    async def send_request(self, method: str, params: Optional[dict] = None) -> dict:
        """Send JSON-RPC request to MCP server and parse SSE response."""
        self.request_id += 1
        
        payload = {
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": method,
        }
        
        if params:
            payload["params"] = params
        
        print(f"[DEBUG] MCP Request: {method}")
        
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
                        print(f"[DEBUG] MCP Response keys: {list(parsed.keys())}")
                        
                        if "error" in parsed:
                            error = parsed['error']
                            raise Exception(
                                f"MCP Error [{error.get('code')}]: {error.get('message')}"
                            )
                        
                        if "result" in parsed:
                            return parsed["result"]
                        
                        full_response = ""
            
            if full_response:
                parsed = parse_sse_response(full_response)
                if parsed:
                    if "error" in parsed:
                        error = parsed['error']
                        raise Exception(
                            f"MCP Error [{error.get('code')}]: {error.get('message')}"
                        )
                    if "result" in parsed:
                        return parsed["result"]
            
            raise Exception("No valid response received from MCP server")
    
    async def initialize(self) -> dict:
        """Initialize MCP session with server."""
        return await self.send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "chainlit-fortimanager-client",
                "version": "1.0.0"
            }
        })
    
    async def list_tools(self) -> List[dict]:
        """List all available tools from MCP server."""
        result = await self.send_request("tools/list")
        return result.get("tools", [])
    
    async def call_tool(self, name: str, arguments: dict) -> dict:
        """Execute a tool on the MCP server."""
        return await self.send_request("tools/call", {
            "name": name,
            "arguments": arguments
        })
    
    async def close(self):
        """Close HTTP client connection."""
        await self.client.aclose()


# ============================================================================
# Intelligent Tool Filtering
# ============================================================================

def filter_relevant_tools(query: str, tools: List[dict], max_tools: int = 100) -> List[dict]:
    """
    Filter tools based on query relevance using category-aware scoring.
    
    Reduces 590+ tools to ~100 most relevant for OpenAI's 128 tool limit.
    """
    query_lower = query.lower()
    keywords = query_lower.split()
    
    # FortiManager MCP tool categories (590 tools across 24 categories)
    category_keywords = {
        'device': ['device', 'firmware', 'vdom', 'ha', 'hardware', 'model', 'cluster', 'revision', 'fortigate', 'fgt'],
        'policy': ['policy', 'firewall', 'rule', 'nat', 'snat', 'dnat', 'package', 'install', 'central'],
        'object': ['address', 'service', 'zone', 'vip', 'pool', 'schedule', 'wildcard', 'fqdn', 'geography'],
        'provision': ['template', 'provision', 'profile', 'cli template', 'system template', 'certificate'],
        'monitor': ['monitor', 'status', 'log', 'statistic', 'health', 'task', 'connectivity', 'performance'],
        'adom': ['adom', 'workspace', 'revision', 'lock', 'commit', 'assignment', 'clone'],
        'security': ['web filter', 'ips', 'antivirus', 'dlp', 'application control', 'waf', 'email filter'],
        'vpn': ['vpn', 'ipsec', 'ssl-vpn', 'tunnel', 'phase1', 'phase2', 'concentrator', 'forticlient'],
        'sdwan': ['sd-wan', 'sdwan', 'wan', 'health check', 'sla', 'link'],
        'fortiap': ['fortiap', 'wtp', 'wireless', 'wifi', 'ssid'],
        'fortiswitch': ['fortiswitch', 'switch', 'port'],
        'fortiextender': ['fortiextender', 'extender', 'lte'],
        'connector': ['connector', 'fabric', 'aws', 'azure', 'vmware', 'sdn'],
        'script': ['script', 'cli script', 'execute', 'run'],
        'fortiguard': ['fortiguard', 'update', 'contract', 'threat', 'database'],
        'internet_service': ['internet service', 'cloud service', 'saas'],
        'installation': ['install', 'deploy', 'push', 'preview', 'validate'],
    }
    
    # Detect relevant categories
    detected_categories = set()
    for category, category_kws in category_keywords.items():
        if any(kw in query_lower for kw in category_kws):
            detected_categories.add(category)
    
    # Score tools
    scored_tools: List[Tuple[int, dict]] = []
    
    for tool in tools:
        score = 0
        tool_name = tool.get("name", "").lower()
        tool_desc = tool.get("description", "").lower()
        
        # Category match
        for category in detected_categories:
            if any(kw in tool_name for kw in category_keywords[category]):
                score += 15
            if any(kw in tool_desc for kw in category_keywords[category]):
                score += 5
        
        # Keyword match
        for keyword in keywords:
            if len(keyword) >= 3:
                if keyword in tool_name:
                    score += 10
                if keyword in tool_desc:
                    score += 3
        
        # Operation type
        operation_types = {
            'list': ['list', 'get', 'show', 'view', 'retrieve'],
            'create': ['create', 'add', 'new'],
            'update': ['update', 'modify', 'edit', 'set', 'change'],
            'delete': ['delete', 'remove'],
            'install': ['install', 'deploy', 'push'],
            'execute': ['execute', 'run'],
        }
        
        for op_type, op_keywords in operation_types.items():
            if any(op in query_lower for op in op_keywords):
                if any(op in tool_name for op in op_keywords):
                    score += 8
        
        # High-priority entities
        high_priority = ['device', 'policy', 'firewall', 'address', 'service', 'adom', 'vdom', 'template', 'vpn', 'sdwan', 'ha', 'cluster']
        for entity in high_priority:
            if entity in query_lower and entity in tool_name:
                score += 12
        
        if score > 0:
            scored_tools.append((score, tool))
    
    scored_tools.sort(reverse=True, key=lambda x: x[0])
    
    if not scored_tools:
        default_tools = [t for t in tools if any(op in t.get("name", "").lower() for op in ['list', 'get'])]
        return default_tools[:max_tools]
    
    return [tool for score, tool in scored_tools[:max_tools]]


# ============================================================================
# MCP Session Management
# ============================================================================

async def init_mcp_session() -> Optional[MCPClient]:
    """Initialize connection to FortiManager MCP server."""
    try:
        print(f"[INFO] Connecting to MCP server at {MCP_SERVER_URL}")
        mcp = MCPClient(MCP_SERVER_URL)
        
        print("[INFO] Initializing MCP session...")
        init_result = await mcp.initialize()
        
        server_name = init_result.get('serverInfo', {}).get('name', 'Unknown')
        server_version = init_result.get('serverInfo', {}).get('version', 'Unknown')
        print(f"[INFO] Connected to: {server_name} v{server_version}")
        
        return mcp
    except Exception as e:
        print(f"[ERROR] MCP connection failed: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# Chainlit Event Handlers
# ============================================================================

@cl.on_chat_start
async def start():
    """Initialize when user starts a new chat."""
    global mcp_session, all_tools
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or "REPLACE" in api_key:
        await cl.Message(
            content=(
                "❌ **OpenAI API Key not configured!**\n\n"
                "Set in `.env` file:\n```\nOPENAI_API_KEY=sk-your-key\n```\n\n"
                "Get key: https://platform.openai.com/api-keys"
            )
        ).send()
        return
    
    await cl.Message(
        content=f"🔄 **Connecting to FortiManager MCP...**\n*Server: `{MCP_SERVER_URL}`*"
    ).send()
    
    try:
        mcp_session = await asyncio.wait_for(init_mcp_session(), timeout=15.0)
    except asyncio.TimeoutError:
        await cl.Message(content="❌ **Connection timeout**\nCheck MCP server status").send()
        return
    
    if not mcp_session:
        await cl.Message(content="❌ **Connection failed**\nCheck terminal logs").send()
        return
    
    try:
        print("[INFO] Fetching tool catalog...")
        all_tools = await asyncio.wait_for(mcp_session.list_tools(), timeout=15.0)
        
        if not all_tools:
            await cl.Message(content="⚠️ **No tools available**").send()
            return
        
        tool_names = [tool.get("name", "unknown") for tool in all_tools]
        
        categories = {
            'Device Management': [t for t in tool_names if any(k in t.lower() for k in ['device', 'vdom', 'ha', 'firmware'])],
            'Policy Management': [t for t in tool_names if any(k in t.lower() for k in ['policy', 'firewall', 'nat', 'package'])],
            'Objects': [t for t in tool_names if any(k in t.lower() for k in ['address', 'service', 'zone', 'vip'])],
            'Security': [t for t in tool_names if any(k in t.lower() for k in ['ips', 'antivirus', 'webfilter', 'dlp'])],
            'VPN': [t for t in tool_names if 'vpn' in t.lower() or 'ipsec' in t.lower()],
            'SD-WAN': [t for t in tool_names if 'sdwan' in t.lower() or 'wan' in t.lower()],
            'ADOM': [t for t in tool_names if 'adom' in t.lower()],
            'Monitoring': [t for t in tool_names if any(k in t.lower() for k in ['monitor', 'status', 'log', 'task'])],
        }
        
        message = f"✅ **Connected!** Total tools: **{len(tool_names)}**\n\n**By category:**\n"
        for cat, tools in categories.items():
            if tools:
                message += f"• **{cat}:** {len(tools)}\n"
        
        message += (
            "\n**Example queries:**\n"
            "• List all ADOMs\n"
            "• List all devices\n"
            "• Show policies in ADOM <name>\n"
            "• List policy packages in ADOM <name>\n"
            "• Get device status\n\n"
            "*💡 Smart filtering: 590 tools → ~100 most relevant per query*"
        )
        
        await cl.Message(content=message).send()
        
    except Exception as e:
        await cl.Message(content=f"⚠️ Error loading tools: {str(e)}").send()
        import traceback
        traceback.print_exc()


@cl.on_message
async def on_message(message: cl.Message):
    """Process user messages and execute operations."""
    global mcp_session, all_tools
    
    if not mcp_session:
        await cl.Message(content="❌ Not connected. Restart chat.").send()
        return
    
    try:
        # Filter tools
        relevant_tools = filter_relevant_tools(message.content, all_tools, max_tools=100)
        print(f"[INFO] Filtered to {len(relevant_tools)}/{len(all_tools)} tools")
        
        if relevant_tools:
            top_10 = [t.get("name") for t in relevant_tools[:10]]
            print(f"[DEBUG] Top 10 tools: {top_10}")
        else:
            await cl.Message(content="⚠️ No relevant tools found").send()
            return
        
        # Convert to OpenAI format
        openai_tools = [{
            "type": "function",
            "function": {
                "name": t.get("name", ""),
                "description": t.get("description", "")[:1000],
                "parameters": t.get("inputSchema", {})
            }
        } for t in relevant_tools]
        
        print(f"[DEBUG] Sending {len(openai_tools)} tools to OpenAI")
        
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a FortiManager assistant with access to management tools.\n\n"
                    "**CRITICAL INSTRUCTIONS:**\n"
                    "1. ALWAYS use available tools to get real data - NEVER give generic information\n"
                    "2. When information is missing, use discovery tools first\n"
                    "3. Follow correct multi-step workflows for complex operations\n\n"
                    "**IMPORTANT WORKFLOWS:**\n\n"
                    "**For Policy Queries:**\n"
                    "Policies are stored in policy packages within ADOMs. ALWAYS:\n"
                    "1. First call list_policy_packages(adom='<adom_name>') to see available packages\n"
                    "2. Then call list_firewall_policies(adom='<adom_name>', pkg='<package_name>')\n\n"
                    "Example: User asks 'show policies in ADOM XYZ'\n"
                    "→ Step 1: list_policy_packages(adom='XYZ')\n"
                    "→ Step 2: For each package, list_firewall_policies(adom='XYZ', pkg='package_name')\n\n"
                    "**For Device Queries:**\n"
                    "- Use list_devices(adom='<adom>') to list devices\n"
                    "- Use get_device(name='<device_name>', adom='<adom>') for details\n"
                    "- Use get_device_status for connectivity and status info\n\n"
                    "**For ADOM Queries:**\n"
                    "- Use list_adoms() to see all ADOMs\n"
                    "- Use get_adom(name='<adom_name>') for specific ADOM details\n\n"
                    "**For Objects:**\n"
                    "- list_addresses(adom='<adom>') for address objects\n"
                    "- list_services(adom='<adom>') for service objects\n"
                    "- list_address_groups(adom='<adom>') for address groups\n\n"
                    "**ERROR HANDLING:**\n"
                    "If a tool call fails due to missing parameters:\n"
                    "- Check what prerequisite information you need\n"
                    "- Use list_* tools to discover required values\n"
                    "- Example: If you need a package name, call list_policy_packages first\n\n"
                    "**KEY TOOLS AVAILABLE:**\n"
                    "- list_adoms, get_adom\n"
                    "- list_devices, get_device, get_device_status\n"
                    "- list_policy_packages, list_firewall_policies\n"
                    "- list_addresses, list_services, list_address_groups\n"
                    "- get_system_status, list_tasks\n\n"
                    "**REMEMBER:**\n"
                    "- Multi-step operations are NORMAL and EXPECTED\n"
                    "- Discovery before details: list packages before policies\n"
                    "- Always specify adom parameter when required\n"
                    "- Present results in clear, formatted tables when appropriate"
                )
            },
            {"role": "user", "content": message.content}
        ]
        
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=openai_tools if openai_tools else None,
            tool_choice="auto",
            temperature=0.2
        )
        
        # Check if OpenAI called any tools
        if not response.choices[0].message.tool_calls:
            print("[WARN] OpenAI did not call any tools")
            print(f"[WARN] Response: {response.choices[0].message.content[:200]}")
        else:
            print(f"[DEBUG] OpenAI called {len(response.choices[0].message.tool_calls)} tools")
        
        max_iterations = 5
        iteration = 0
        
        while response.choices[0].message.tool_calls and iteration < max_iterations:
            iteration += 1
            assistant_message = response.choices[0].message
            
            print(f"[INFO] Iteration {iteration}: {len(assistant_message.tool_calls)} tool calls")
            
            messages.append({
                "role": "assistant",
                "content": assistant_message.content,
                "tool_calls": [{
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments}
                } for tc in assistant_message.tool_calls]
            })
            
            for tool_call in assistant_message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)
                
                print(f"[INFO] Calling: {tool_name} with {tool_args}")
                
                await cl.Message(
                    content=f"🔧 **{tool_name}**\n```json\n{json.dumps(tool_args, indent=2)}\n```"
                ).send()
                
                try:
                    result = await mcp_session.call_tool(tool_name, tool_args)
                    
                    if isinstance(result, dict):
                        if "content" in result:
                            content = result["content"]
                            tool_response = content[0].get("text", str(content)) if isinstance(content, list) and content else str(content)
                        else:
                            tool_response = json.dumps(result, indent=2)
                    else:
                        tool_response = str(result)
                    
                    print(f"[INFO] Tool {tool_name} succeeded, response length: {len(tool_response)}")
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_response
                    })
                    
                except Exception as e:
                    error_msg = f"Error calling {tool_name}: {str(e)}"
                    print(f"[ERROR] {error_msg}")
                    await cl.Message(content=f"⚠️ {error_msg}").send()
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": error_msg
                    })
            
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=openai_tools if openai_tools else None,
                temperature=0.2
            )
        
        if iteration >= max_iterations and response.choices[0].message.tool_calls:
            await cl.Message(
                content="⚠️ **Reached iteration limit**\nOperation was complex. Try breaking it into smaller queries."
            ).send()
        
        if response.choices[0].message.content:
            await cl.Message(content=response.choices[0].message.content).send()
        else:
            await cl.Message(content="✅ Operation completed").send()
            
    except Exception as e:
        await cl.Message(content=f"❌ Error: {str(e)}").send()
        import traceback
        traceback.print_exc()


@cl.on_chat_end
async def end():
    """Cleanup when chat ends."""
    global mcp_session
    if mcp_session:
        try:
            await mcp_session.close()
            print("[INFO] MCP connection closed")
        except Exception as e:
            print(f"[ERROR] Cleanup error: {e}")
    mcp_session = None


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║     FortiManager MCP - Chainlit Integration v1.0.0          ║
║                                                              ║
║  Requirements:                                              ║
║  • FortiManager MCP Server at port 8000                    ║
║  • OpenAI API key in .env                                  ║
║                                                              ║
║  Run: chainlit run app.py --host 0.0.0.0 --port 8001      ║
╚══════════════════════════════════════════════════════════════╝
    """)