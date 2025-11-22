# app.py
"""
Chainlit Application for FortiManager MCP Server Integration

This application provides a conversational interface to FortiManager through:
- FortiManager MCP Server (590+ tools via HTTP/SSE)
- OpenAI API for natural language understanding
- Chainlit for web-based chat interface

Architecture:
┌─────────────────┐      ┌──────────────────┐      ┌─────────────────────┐
│   Chainlit UI   │ ───► │  OpenAI API      │ ───► │ FortiManager MCP    │
│  (Web Browser)  │ ◄─── │  (gpt-4o-mini)   │ ◄─── │ Server (HTTP/SSE)   │
└─────────────────┘      └──────────────────┘      └─────────────────────┘
                                                             │
                                                             ▼
                                                    ┌─────────────────────┐
                                                    │   FortiManager      │
                                                    │   (JSON-RPC API)    │
                                                    └─────────────────────┘

Component Responsibilities:
- Chainlit: User interface and session management
- OpenAI: Natural language processing and tool selection
- This App: Tool filtering, MCP communication, response formatting
- MCP Server: Tool execution and FortiManager API abstraction
- FortiManager: Network device management and policy configuration

Tool Filtering Strategy:
With 590+ tools available from the MCP server, we implement intelligent
filtering to stay within OpenAI's 128 tool limit:
- Category-aware scoring (Device, Policy, ADOM, VPN, SD-WAN, etc.)
- Operation type matching (list, create, update, delete)
- Keyword relevance scoring
- Returns top 100 most relevant tools per query

Environment Variables:
- OPENAI_API_KEY: Your OpenAI API key (required)
- OPENAI_MODEL: Model to use (default: gpt-4o-mini)

Author: Integration between Chainlit, OpenAI, and FortiManager MCP
Version: 1.0.0
License: Same as FortiManager MCP Server
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
# The FortiManager MCP server must be running and accessible at this URL
# Default: http://localhost:8000/mcp (streamable HTTP transport)
# For remote servers, update to: http://{host}:{port}/mcp
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8000/mcp")

# Global State
# These maintain the MCP connection and tool cache across the session
mcp_session: Optional['MCPClient'] = None
all_tools: List[Dict] = []  # Cache of all 590+ tools from MCP server


# ============================================================================
# MCP Protocol Implementation
# ============================================================================

def parse_sse_response(text: str) -> Optional[dict]:
    """
    Parse Server-Sent Events (SSE) formatted response from MCP server.
    
    The FortiManager MCP server uses SSE transport which returns responses in:
        event: message
        data: {"jsonrpc":"2.0","id":1,"result":{...}}
    
    This function extracts and parses the JSON from the data line.
    
    Args:
        text: Raw SSE response text containing event and data lines
        
    Returns:
        Parsed JSON-RPC response dictionary, or None if parsing fails
        
    Reference:
        See ADR-003-streamable-http.md in MCP server documentation
    """
    lines = text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if line.startswith('data: '):
            data_json = line[6:]  # Remove 'data: ' prefix
            try:
                return json.loads(data_json)
            except json.JSONDecodeError as e:
                print(f"[ERROR] Failed to parse JSON: {e}")
                print(f"[ERROR] JSON content: {data_json[:200]}...")
                return None
    
    return None


class MCPClient:
    """
    HTTP/SSE client for Model Context Protocol communication.
    
    Implements the MCP protocol over HTTP with Server-Sent Events (SSE) transport
    to communicate with the FortiManager MCP server. Handles JSON-RPC 2.0
    request/response format.
    
    The MCP protocol flow:
    1. Initialize: Establish session and negotiate protocol version
    2. List Tools: Retrieve available tools (590+ for FortiManager)
    3. Call Tool: Execute specific FortiManager operations
    
    Attributes:
        url: MCP server endpoint (e.g., http://localhost:8000/mcp)
        request_id: Incrementing ID for JSON-RPC requests
        client: HTTP client with connection pooling and timeout
        
    Reference:
        MCP Server Architecture: see architecture.md in server documentation
    """
    
    def __init__(self, url: str):
        """
        Initialize MCP client.
        
        Args:
            url: MCP server endpoint URL with /mcp path
        """
        self.url = url
        self.request_id = 0
        self.client = httpx.AsyncClient(timeout=30.0)
        
    async def send_request(self, method: str, params: Optional[dict] = None) -> dict:
        """
        Send JSON-RPC request to MCP server and parse SSE response.
        
        Constructs a JSON-RPC 2.0 request, sends it via HTTP POST with SSE
        headers, and parses the streaming response.
        
        Args:
            method: JSON-RPC method name (e.g., "initialize", "tools/list", "tools/call")
            params: Optional parameters dict for the method
            
        Returns:
            The "result" field from the JSON-RPC response
            
        Raises:
            Exception: If HTTP request fails, response is invalid, or MCP returns error
            
        Example:
            result = await client.send_request("tools/list")
            # result = {"tools": [...]}
        """
        self.request_id += 1
        
        # Construct JSON-RPC 2.0 request
        payload = {
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": method,
        }
        
        if params:
            payload["params"] = params
        
        print(f"[DEBUG] MCP Request: {method}")
        
        # Send request with SSE-compatible headers
        async with self.client.stream(
            "POST",
            self.url,
            json=payload,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream"  # Critical for SSE
            }
        ) as response:
            # Check HTTP status
            if response.status_code != 200:
                text = await response.aread()
                raise Exception(f"HTTP {response.status_code}: {text.decode()}")
            
            # Collect and parse SSE stream
            full_response = ""
            async for chunk in response.aiter_text():
                full_response += chunk
                
                # Process complete events (end with double newline)
                if "\n\n" in full_response:
                    parsed = parse_sse_response(full_response)
                    if parsed:
                        print(f"[DEBUG] MCP Response keys: {list(parsed.keys())}")
                        
                        # Handle JSON-RPC error
                        if "error" in parsed:
                            error = parsed['error']
                            raise Exception(
                                f"MCP Error [{error.get('code')}]: {error.get('message')}"
                            )
                        
                        # Return result
                        if "result" in parsed:
                            return parsed["result"]
                        
                        full_response = ""  # Clear buffer
            
            # Final attempt with remaining buffer
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
        """
        Initialize MCP session with server.
        
        Establishes connection, negotiates protocol version, and exchanges
        client/server capabilities.
        
        Returns:
            Server information including:
            - serverInfo: {name, version}
            - capabilities: {tools, resources, prompts}
            - protocolVersion: Negotiated protocol version
            
        Raises:
            Exception: If initialization fails
        """
        return await self.send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "chainlit-fortimanager-client",
                "version": "1.0.0"
            }
        })
    
    async def list_tools(self) -> List[dict]:
        """
        List all available tools from MCP server.
        
        Retrieves the complete tool catalog. For FortiManager MCP server,
        this returns 590+ tools across all functional categories.
        
        Returns:
            List of tool definitions, each containing:
            - name: Tool identifier (e.g., "list_devices")
            - description: Human-readable description for LLM
            - inputSchema: JSON schema for input parameters
            - outputSchema: JSON schema for output (optional)
            
        Reference:
            See API_COVERAGE.md for complete tool breakdown by category
        """
        result = await self.send_request("tools/list")
        return result.get("tools", [])
    
    async def call_tool(self, name: str, arguments: dict) -> dict:
        """
        Execute a tool on the MCP server.
        
        Invokes a FortiManager operation through the MCP server, which handles
        the actual FortiManager API communication.
        
        Args:
            name: Tool name from list_tools (e.g., "list_devices", "get_adom")
            arguments: Tool arguments matching the inputSchema
            
        Returns:
            Tool execution result, typically containing:
            - content: Array of text/data objects with results
            - isError: Boolean indicating if tool execution failed (optional)
            
        Raises:
            Exception: If tool call fails or MCP returns error
            
        Example:
            result = await client.call_tool("list_devices", {"adom": "root"})
        """
        return await self.send_request("tools/call", {
            "name": name,
            "arguments": arguments
        })
    
    async def close(self):
        """Close HTTP client connection and cleanup resources."""
        await self.client.aclose()


# ============================================================================
# Intelligent Tool Filtering
# ============================================================================

def filter_relevant_tools(query: str, tools: List[dict], max_tools: int = 100) -> List[dict]:
    """
    Filter tools based on query relevance using category-aware scoring.
    
    Challenge: FortiManager MCP server provides 590+ tools, but OpenAI's API
    has a limit of 128 tools per request. We solve this by:
    
    1. Detecting relevant categories from the query
    2. Scoring tools based on multiple factors
    3. Returning the top N most relevant tools
    
    Scoring Factors:
    - Category match: +15 points (query mentions "device" → device tools)
    - Entity match: +12 points (high-priority entities like device, policy)
    - Keyword in name: +10 points (exact matches in tool name)
    - Operation match: +8 points (query says "list" → list_* tools)
    - Category in description: +5 points
    - Keyword in description: +3 points
    
    Categories (based on FortiManager MCP implementation):
    - Device Management (65 tools): device, firmware, VDOM, HA
    - Policy Management (50 tools): policy, firewall, NAT
    - Objects Management (59 tools): address, service, zone, VIP
    - Provisioning & Templates (98 tools): template, profile
    - Security Profiles (27 tools): web filter, IPS, antivirus, DLP
    - VPN Management (18 tools): VPN, IPsec, SSL-VPN
    - SD-WAN Management (19 tools): SD-WAN, health check
    - ADOM Management (27 tools): ADOM, workspace, revision
    - Monitoring & Tasks (58 tools): monitor, status, logs
    - And 15+ more specialized categories
    
    Args:
        query: User's natural language query
        tools: All available tools from MCP server
        max_tools: Maximum tools to return (default: 100, OpenAI limit: 128)
        
    Returns:
        Filtered list of most relevant tools, sorted by relevance score
        
    Example:
        query = "list all devices in production ADOM"
        # Returns tools like: list_devices, get_device, list_adoms, get_adom
        # Filters out: create_policy, update_address, install_package, etc.
        
    Reference:
        Tool organization: see src/fortimanager_mcp/tools/ in server repo
        API coverage: see docs/API_COVERAGE.md for complete breakdown
    """
    query_lower = query.lower()
    keywords = query_lower.split()
    
    # Category-to-keyword mapping based on FortiManager MCP tool structure
    # Aligned with: src/fortimanager_mcp/tools/ directory organization
    category_keywords = {
        # Core Infrastructure (device_tools.py)
        'device': [
            'device', 'firmware', 'vdom', 'ha', 'hardware', 'model', 
            'cluster', 'revision', 'fortigate', 'fgt'
        ],
        
        # Policy Management (policy_tools.py)
        'policy': [
            'policy', 'firewall', 'rule', 'nat', 'snat', 'dnat', 
            'package', 'install', 'central'
        ],
        
        # Objects Management (object_tools.py)
        'object': [
            'address', 'service', 'zone', 'vip', 'pool', 'schedule',
            'wildcard', 'fqdn', 'geography'
        ],
        
        # Provisioning & Templates
        'provision': [
            'template', 'provision', 'profile', 'cli template', 
            'system template', 'certificate'
        ],
        
        # Monitoring & Tasks (monitoring_tools.py)
        'monitor': [
            'monitor', 'status', 'log', 'statistic', 'health', 
            'task', 'connectivity', 'performance'
        ],
        
        # ADOM Management
        'adom': [
            'adom', 'workspace', 'revision', 'lock', 'commit',
            'assignment', 'clone'
        ],
        
        # Security Profiles
        'security': [
            'web filter', 'ips', 'antivirus', 'dlp', 
            'application control', 'waf', 'email filter'
        ],
        
        # VPN Management
        'vpn': [
            'vpn', 'ipsec', 'ssl-vpn', 'tunnel', 'phase1', 'phase2',
            'concentrator', 'forticlient'
        ],
        
        # SD-WAN Management
        'sdwan': [
            'sd-wan', 'sdwan', 'wan', 'health check', 
            'sla', 'link'
        ],
        
        # FortiAP/FortiSwitch/FortiExtender
        'fortiap': ['fortiap', 'wtp', 'wireless', 'wifi', 'ssid'],
        'fortiswitch': ['fortiswitch', 'switch', 'port'],
        'fortiextender': ['fortiextender', 'extender', 'lte'],
        
        # Advanced Features
        'connector': ['connector', 'fabric', 'aws', 'azure', 'vmware', 'sdn'],
        'script': ['script', 'cli script', 'execute', 'run'],
        'fortiguard': ['fortiguard', 'update', 'contract', 'threat', 'database'],
        'internet_service': ['internet service', 'cloud service', 'saas'],
        'installation': ['install', 'deploy', 'push', 'preview', 'validate'],
    }
    
    # Detect which categories are relevant to this query
    detected_categories = set()
    for category, category_kws in category_keywords.items():
        if any(kw in query_lower for kw in category_kws):
            detected_categories.add(category)
    
    # Score each tool
    scored_tools: List[Tuple[int, dict]] = []
    
    for tool in tools:
        score = 0
        tool_name = tool.get("name", "").lower()
        tool_desc = tool.get("description", "").lower()
        
        # Category match bonus (highest weight)
        for category in detected_categories:
            if any(kw in tool_name for kw in category_keywords[category]):
                score += 15
            if any(kw in tool_desc for kw in category_keywords[category]):
                score += 5
        
        # Keyword match in tool name
        for keyword in keywords:
            if len(keyword) >= 3:  # Ignore short words like "in", "of"
                if keyword in tool_name:
                    score += 10
                if keyword in tool_desc:
                    score += 3
        
        # Operation type matching
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
        
        # High-priority entity matches (critical FortiManager concepts)
        high_priority_entities = [
            'device', 'policy', 'firewall', 'address', 'service',
            'adom', 'vdom', 'template', 'vpn', 'sdwan', 'ha', 'cluster'
        ]
        
        for entity in high_priority_entities:
            if entity in query_lower and entity in tool_name:
                score += 12
        
        # Add tool to scored list if relevant
        if score > 0:
            scored_tools.append((score, tool))
    
    # Sort by score (highest first)
    scored_tools.sort(reverse=True, key=lambda x: x[0])
    
    # Fallback: If no scored tools, return common read operations
    if not scored_tools:
        default_tools = [
            t for t in tools 
            if any(op in t.get("name", "").lower() for op in ['list', 'get'])
        ]
        return default_tools[:max_tools]
    
    # Return top N scored tools
    return [tool for score, tool in scored_tools[:max_tools]]


# ============================================================================
# MCP Session Management
# ============================================================================

async def init_mcp_session() -> Optional[MCPClient]:
    """
    Initialize connection to FortiManager MCP server.
    
    Establishes HTTP/SSE connection and negotiates MCP protocol. This must
    succeed before any tools can be used.
    
    Returns:
        Connected MCPClient instance, or None if connection fails
        
    Side Effects:
        Logs connection status and server information to console
    """
    try:
        print(f"[INFO] Connecting to MCP server at {MCP_SERVER_URL}")
        
        mcp = MCPClient(MCP_SERVER_URL)
        
        # Initialize MCP session (protocol handshake)
        print("[INFO] Initializing MCP session...")
        init_result = await mcp.initialize()
        
        # Extract server information
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
    """
    Chainlit lifecycle event: Initialize when user starts a new chat.
    
    This function:
    1. Validates OpenAI API key configuration
    2. Connects to FortiManager MCP server
    3. Loads all 590+ available tools
    4. Categorizes and displays tool summary
    5. Shows example queries to guide user
    
    The connection is maintained for the entire chat session and shared
    across all messages from the user.
    """
    global mcp_session, all_tools
    
    # Validate OpenAI configuration
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or "REPLACE" in api_key:
        await cl.Message(
            content=(
                "❌ **OpenAI API Key not configured!**\n\n"
                "Please set your OpenAI API key in `.env` file:\n"
                "```bash\n"
                "OPENAI_API_KEY=sk-your-actual-key-here\n"
                "```\n\n"
                "Get your key at: https://platform.openai.com/api-keys"
            )
        ).send()
        return
    
    # Show connection progress
    await cl.Message(
        content=(
            "🔄 **Connecting to FortiManager MCP server...**\n\n"
            f"*Server: `{MCP_SERVER_URL}`*\n"
            "*This may take a few seconds*"
        )
    ).send()
    
    # Connect to MCP server
    try:
        mcp_session = await asyncio.wait_for(init_mcp_session(), timeout=15.0)
    except asyncio.TimeoutError:
        await cl.Message(
            content=(
                "❌ **Connection timeout**\n\n"
                f"Could not connect to MCP server at `{MCP_SERVER_URL}` within 15 seconds.\n\n"
                "**Troubleshooting:**\n"
                "• Verify MCP server is running: `docker ps`\n"
                "• Check server logs: `docker logs fortimanager-mcp`\n"
                "• Test endpoint: `curl http://localhost:8000/health`\n"
                "• Verify URL matches server configuration"
            )
        ).send()
        return
    
    if not mcp_session:
        await cl.Message(
            content=(
                "❌ **Failed to connect**\n\n"
                "Check terminal logs for detailed error information.\n\n"
                "Common issues:\n"
                "• MCP server not running\n"
                "• FortiManager credentials incorrect\n"
                "• Network connectivity problems"
            )
        ).send()
        return
    
    # Load all tools from MCP server
    try:
        print("[INFO] Fetching tool catalog from MCP server...")
        all_tools = await asyncio.wait_for(mcp_session.list_tools(), timeout=15.0)
        
        if not all_tools:
            await cl.Message(
                content="⚠️ **Connected but no tools available**\n\n"
                        "The MCP server returned an empty tool list."
            ).send()
            return
        
        # Categorize tools for display
        tool_names = [tool.get("name", "unknown") for tool in all_tools]
        
        # Tool categories based on FortiManager MCP implementation
        # Reference: src/fortimanager_mcp/tools/ directory structure
        # Documentation: docs/API_COVERAGE.md
        categories = {
            'Device Management': [
                t for t in tool_names 
                if any(k in t.lower() for k in ['device', 'vdom', 'ha', 'firmware', 'revision', 'model'])
            ],
            'Policy Management': [
                t for t in tool_names 
                if any(k in t.lower() for k in ['policy', 'firewall', 'nat', 'package'])
            ],
            'Objects': [
                t for t in tool_names 
                if any(k in t.lower() for k in ['address', 'service', 'zone', 'vip', 'pool'])
                and 'internet' not in t.lower()
            ],
            'Provisioning & Templates': [
                t for t in tool_names 
                if any(k in t.lower() for k in ['template', 'provision', 'certificate', 'profile'])
            ],
            'Security Profiles': [
                t for t in tool_names 
                if any(k in t.lower() for k in ['webfilter', 'ips', 'antivirus', 'dlp', 'application', 'waf'])
            ],
            'VPN': [
                t for t in tool_names 
                if 'vpn' in t.lower() or 'ipsec' in t.lower() or 'ssl_vpn' in t.lower()
            ],
            'SD-WAN': [
                t for t in tool_names 
                if 'sdwan' in t.lower() or 'sd_wan' in t.lower() or 'wan' in t.lower()
            ],
            'ADOM Management': [
                t for t in tool_names 
                if 'adom' in t.lower() or 'workspace' in t.lower()
            ],
            'Monitoring & Tasks': [
                t for t in tool_names 
                if any(k in t.lower() for k in ['monitor', 'status', 'log', 'statistic', 'task', 'health'])
            ],
            'FortiGuard': [
                t for t in tool_names 
                if 'fortiguard' in t.lower() or 'update' in t.lower() or 'contract' in t.lower()
            ],
            'Internet Services': [
                t for t in tool_names 
                if 'internet_service' in t.lower()
            ],
            'CLI Scripts': [
                t for t in tool_names 
                if 'script' in t.lower() and 'cli' in t.lower()
            ],
            'Installation': [
                t for t in tool_names 
                if 'install' in t.lower() and 'policy' not in t.lower()
            ],
        }
        
        # Build welcome message
        message = (
            "✅ **Connected to FortiManager MCP Server!**\n\n"
            f"**Total tools available: {len(tool_names)}**\n\n"
        )
        
        message += "**Tools by category:**\n"
        for category, tools in categories.items():
            if tools:
                message += f"• **{category}:** {len(tools)} tools\n"
        
        # Count uncategorized tools
        categorized_count = sum(len(tools) for tools in categories.values())
        other_count = len(tool_names) - categorized_count
        if other_count > 0:
            message += f"• **Other:** {other_count} tools\n"
        
        message += (
            "\n**Example queries:**\n"
            "• List all FortiGate devices\n"
            "• Show firewall policies in the production ADOM\n"
            "• List all ADOMs with their status\n"
            "• Get device status and firmware version\n"
            "• List internet service groups\n"
            "• Show VPN tunnel status\n"
            "• Create an address group for web servers\n"
            "• Install policy package to devices\n"
            "• List pending tasks\n\n"
            "*💡 **Smart Tool Selection:** Tools are intelligently filtered based on your query. "
            "Up to 100 most relevant tools are selected from the 590+ available, ensuring "
            "optimal performance within OpenAI's limits.*"
        )
        
        await cl.Message(content=message).send()
        
    except asyncio.TimeoutError:
        await cl.Message(
            content=(
                "⚠️ **Connected but timeout listing tools**\n\n"
                "The tool list request took too long. The MCP server might be busy or "
                "the response is very large. Connection is established and you can try queries."
            )
        ).send()
    except Exception as e:
        await cl.Message(
            content=(
                f"⚠️ **Connected but error listing tools**\n\n"
                f"Error: `{str(e)}`\n\n"
                "Check terminal logs for detailed error information."
            )
        ).send()
        import traceback
        traceback.print_exc()


@cl.on_message
async def on_message(message: cl.Message):
    """
    Chainlit lifecycle event: Process user messages and execute operations.
    
    This is the main message handling loop that:
    1. Filters 590+ tools down to ~100 most relevant
    2. Converts tools to OpenAI function calling format
    3. Sends query to OpenAI with filtered tools
    4. Executes any tool calls via MCP
    5. Continues conversation until complete
    
    The flow supports multi-turn tool usage, where OpenAI can chain multiple
    tool calls together to accomplish complex tasks.
    
    Args:
        message: User's message from Chainlit UI
    """
    global mcp_session, all_tools
    
    # Verify MCP connection
    if not mcp_session:
        await cl.Message(
            content=(
                "❌ **Not connected to MCP server**\n\n"
                "The connection was lost or never established. "
                "Please restart the chat to reconnect."
            )
        ).send()
        return
    
    try:
        # Filter tools based on query relevance
        # This is critical: reduces 590+ tools to ~100 most relevant
        relevant_tools = filter_relevant_tools(
            query=message.content,
            tools=all_tools,
            max_tools=100
        )
        
        # Log filtering results
        print(f"[INFO] Filtered to {len(relevant_tools)} tools from {len(all_tools)} total")
        if relevant_tools:
            top_5 = [t.get("name") for t in relevant_tools[:5]]
            print(f"[INFO] Top 5 relevant tools: {top_5}")
        
        # Convert to OpenAI function calling format
        openai_tools = []
        for tool in relevant_tools:
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", "")[:1000],  # Truncate long descriptions
                    "parameters": tool.get("inputSchema", {})
                }
            })
        
        # Build conversation context
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert FortiManager assistant with access to 590+ management tools.\n\n"
                    "Your role:\n"
                    "- Help users manage FortiGate devices, policies, objects, and configurations\n"
                    "- Use available tools to retrieve information and perform operations\n"
                    "- Present information clearly with formatting, tables, or bullet points\n"
                    "- Provide context and explain technical details when helpful\n"
                    "- Confirm destructive operations before executing\n\n"
                    "Available capabilities:\n"
                    "- Device Management: List, configure, upgrade devices and VDOMs\n"
                    "- Policy Management: Create, modify, install firewall policies\n"
                    "- Objects: Manage addresses, services, schedules, VIPs\n"
                    "- Provisioning: Apply templates and profiles\n"
                    "- Security: Configure IPS, antivirus, web filters, application control\n"
                    "- VPN: Manage IPsec and SSL-VPN tunnels\n"
                    "- SD-WAN: Configure and monitor SD-WAN\n"
                    "- ADOM Management: Organize and manage administrative domains\n"
                    "- Monitoring: Check status, view logs, track tasks\n\n"
                    "When presenting data:\n"
                    "- Use markdown tables for structured data\n"
                    "- Use bullet points for lists\n"
                    "- Highlight important information with **bold**\n"
                    "- Use code blocks for configuration snippets\n"
                    "- Provide summaries before detailed data"
                )
            },
            {"role": "user", "content": message.content}
        ]
        
        # Initial OpenAI API call
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=openai_tools if openai_tools else None,
            temperature=0.2  # Lower temperature for more consistent, factual responses
        )
        
        # Handle tool calls (OpenAI may require multiple rounds)
        max_iterations = 5  # Prevent infinite loops
        iteration = 0
        
        while response.choices[0].message.tool_calls and iteration < max_iterations:
            iteration += 1
            assistant_message = response.choices[0].message
            
            print(f"[INFO] Tool call iteration {iteration}/{max_iterations}")
            
            # Add assistant's message with tool calls to conversation
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
            
            # Execute each requested tool
            for tool_call in assistant_message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)
                
                # Show user what we're doing
                await cl.Message(
                    content=(
                        f"🔧 **Executing:** `{tool_name}`\n"
                        f"```json\n{json.dumps(tool_args, indent=2)}\n```"
                    )
                ).send()
                
                try:
                    # Execute tool via MCP
                    print(f"[INFO] Calling MCP tool: {tool_name}")
                    result = await mcp_session.call_tool(tool_name, tool_args)
                    
                    # Extract result content from MCP response format
                    # MCP returns: {"content": [{"type": "text", "text": "..."}]}
                    if isinstance(result, dict):
                        if "content" in result:
                            content = result["content"]
                            if isinstance(content, list) and content:
                                # Extract text from first content item
                                tool_response = content[0].get("text", str(content))
                            else:
                                tool_response = str(content)
                        else:
                            # Fallback: stringify entire result
                            tool_response = json.dumps(result, indent=2)
                    else:
                        tool_response = str(result)
                    
                    # Add successful tool result to conversation
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_response
                    })
                    
                    print(f"[INFO] Tool {tool_name} executed successfully")
                    
                except Exception as e:
                    # Handle tool execution errors
                    error_msg = f"Error calling {tool_name}: {str(e)}"
                    print(f"[ERROR] {error_msg}")
                    
                    # Show error to user
                    await cl.Message(
                        content=f"⚠️ **Tool Error:** {error_msg}"
                    ).send()
                    
                    # Add error to conversation so OpenAI can handle it
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": f"Error: {error_msg}"
                    })
            
            # Get next response from OpenAI
            # OpenAI will use tool results to formulate next action or final answer
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=openai_tools if openai_tools else None,
                temperature=0.2
            )
        
        # Check if we hit max iterations
        if iteration >= max_iterations and response.choices[0].message.tool_calls:
            print(f"[WARN] Reached max iterations ({max_iterations}) with pending tool calls")
            await cl.Message(
                content=(
                    "⚠️ **Complex operation reached iteration limit**\n\n"
                    "The operation required more steps than allowed. "
                    "Try breaking down the request into smaller parts."
                )
            ).send()
        
        # Send final response to user
        final_response = response.choices[0].message.content
        if final_response:
            await cl.Message(content=final_response).send()
        else:
            # OpenAI sometimes returns empty content
            await cl.Message(
                content="✅ **Operation completed**\n\nThe requested operation finished successfully."
            ).send()
        
    except Exception as e:
        # Handle unexpected errors
        error_msg = f"Unexpected error: {str(e)}"
        print(f"[ERROR] {error_msg}")
        
        await cl.Message(
            content=(
                f"❌ **Error:** {error_msg}\n\n"
                "Check terminal logs for detailed error information."
            )
        ).send()
        
        import traceback
        traceback.print_exc()


@cl.on_chat_end
async def end():
    """
    Chainlit lifecycle event: Cleanup when chat session ends.
    
    Properly closes the MCP client connection and releases resources.
    """
    global mcp_session
    
    if mcp_session:
        try:
            await mcp_session.close()
            print("[INFO] MCP connection closed gracefully")
        except Exception as e:
            print(f"[ERROR] Error closing MCP connection: {e}")
    
    # Clear global state
    mcp_session = None


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║  FortiManager MCP Integration with Chainlit + OpenAI        ║
║                                                              ║
║  This application requires:                                 ║
║  • FortiManager MCP Server running at port 8000            ║
║  • OpenAI API key in environment                           ║
║  • Chainlit installed: pip install chainlit               ║
║                                                              ║
║  To run:                                                     ║
║  $ chainlit run app.py --host 0.0.0.0 --port 8001          ║
╚══════════════════════════════════════════════════════════════╝
    """)