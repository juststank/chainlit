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

TOOL CATEGORIES (590 tools total)
----------------------------------
Based on actual MCP server implementation:

Infrastructure Management:
  • Device Management (65+ tools): list_devices, get_device_details, add_device, etc.
  • ADOM Management (27+ tools): list_adoms, get_adom_statistics, create_adom, etc.
  • Provisioning (98+ tools): CLI templates, system templates, certificates

Policy & Security:
  • Policy Management (50+ tools): list_policy_packages, list_firewall_policies, etc.
  • Objects (59+ tools): list_firewall_addresses, create_firewall_address, etc.
  • Security Profiles (27+ tools): IPS, antivirus, web filter, DLP

Network Services:
  • VPN Management (18+ tools): IPsec, SSL-VPN configuration
  • SD-WAN (19+ tools): Zones, health checks, services, members
  • Internet Services: Service groups, custom applications

System Operations:
  • Monitoring (58+ tools): System status, tasks, logs, statistics
  • Installation (20+ tools): install_policy_package, install_device_settings
  • Scripts (12+ tools): create_script, execute_script, list_scripts
  • Workspace (20+ tools): lock_adom_workspace, commit_adom_workspace

Advanced Features:
  • FortiGuard (28+ tools): Updates, contracts, threat feeds
  • Connectors (11+ tools): SDN, cloud, fabric connectors
  • Meta Fields: Custom metadata and tagging
  • Docker, CSF, QoS, and more

CRITICAL WORKFLOWS
------------------
Policy Operations:
  IMPORTANT: Policies are stored in packages within ADOMs
  Always follow this sequence:
    1. list_policy_packages(adom='X') → get available packages
    2. list_firewall_policies(adom='X', package='Y') → get policies in package

Installation Operations:
  For policy installation:
    1. install_policy_package(package='X', device='Y', adom='Z')
  For device settings:
    1. install_device_settings(device='X', adom='Y')

Workspace Operations:
  For safe editing:
    1. lock_adom_workspace(adom='X')
    2. Make changes...
    3. commit_adom_workspace(adom='X')
    4. unlock_adom_workspace(adom='X')

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
    Based on actual FortiManager MCP tool implementation.
    """
    query_lower = query.lower()
    keywords = query_lower.split()
    
    # Category keywords based on actual MCP tool files
    category_keywords = {
        # Device Management (device_tools.py)
        'device': [
            'device', 'fortigate', 'fgt', 'firmware', 'vdom', 'ha', 'hardware', 
            'model', 'cluster', 'revision', 'serial', 'platform'
        ],
        
        # Policy Management (policy_tools.py)
        'policy': [
            'policy', 'firewall', 'rule', 'nat', 'snat', 'dnat', 'package', 
            'install', 'central', 'consolidated'
        ],
        
        # Objects (object_tools.py, additional_object_tools.py, advanced_object_tools.py)
        'object': [
            'address', 'service', 'zone', 'vip', 'pool', 'schedule', 'wildcard', 
            'fqdn', 'geography', 'addrgrp', 'service group', 'internet service'
        ],
        
        # Provisioning (provisioning_tools.py)
        'provision': [
            'template', 'provision', 'profile', 'cli template', 'system template', 
            'certificate', 'widget', 'admin'
        ],
        
        # Monitoring (monitoring_tools.py)
        'monitor': [
            'monitor', 'status', 'log', 'statistic', 'health', 'task', 
            'connectivity', 'performance', 'dashboard'
        ],
        
        # ADOM Management (adom_tools.py)
        'adom': [
            'adom', 'workspace', 'revision', 'lock', 'commit', 'assignment', 
            'clone', 'administrative domain'
        ],
        
        # Security Profiles (security_tools.py)
        'security': [
            'webfilter', 'web filter', 'ips', 'antivirus', 'av', 'dlp', 
            'application control', 'waf', 'email filter', 'profile group'
        ],
        
        # VPN Management (vpn_tools.py)
        'vpn': [
            'vpn', 'ipsec', 'ssl-vpn', 'ssl vpn', 'tunnel', 'phase1', 'phase2', 
            'concentrator', 'forticlient'
        ],
        
        # SD-WAN (sdwan_tools.py)
        'sdwan': [
            'sd-wan', 'sdwan', 'sd wan', 'wan', 'health check', 'sla', 'link', 
            'traffic class', 'wan profile'
        ],
        
        # Scripts (script_tools.py)
        'script': [
            'script', 'cli script', 'execute', 'run', 'jinja'
        ],
        
        # FortiGuard (fortiguard_tools.py)
        'fortiguard': [
            'fortiguard', 'update', 'contract', 'threat', 'database', 'license'
        ],
        
        # Installation (installation_tools.py from policy_tools.py)
        'installation': [
            'install', 'deploy', 'push', 'preview', 'validate', 'abort'
        ],
        
        # Workspace (workspace_tools.py)
        'workspace': [
            'lock', 'unlock', 'commit', 'workspace', 'revert'
        ],
        
        # Connectors (connector_tools.py)
        'connector': [
            'connector', 'fabric', 'aws', 'azure', 'vmware', 'sdn', 'cloud'
        ],
        
        # System (system_tools.py)
        'system': [
            'system', 'backup', 'restore', 'admin', 'certificate', 'interface',
            'snmp', 'syslog', 'ntp', 'dns', 'route', 'global', 'static route',
            'routing', 'gateway', 'routing_table', 'routing table', 'table', 'router'
        ],
        
        # Additional categories
        'fortiap': ['fortiap', 'wtp', 'wireless', 'wifi', 'ssid'],
        'fortiswitch': ['fortiswitch', 'switch', 'port'],
        'fortiextender': ['fortiextender', 'extender', 'lte'],
        'qos': ['qos', 'shaping', 'bandwidth', 'traffic shaping'],
        'csf': ['csf', 'fabric topology', 'security fabric'],
        'docker': ['docker', 'container'],
        'metafield': ['meta', 'metadata', 'tag', 'custom field'],
    }
    
    # Detect relevant categories
    detected_categories = set()
    for category, category_kws in category_keywords.items():
        if any(kw in query_lower for kw in category_kws):
            detected_categories.add(category)
    
    # High-priority entities
    high_priority = [
        'device', 'policy', 'firewall', 'address', 'service', 'adom', 
        'vdom', 'template', 'vpn', 'sdwan', 'ha', 'cluster', 'package',
        'script', 'install', 'workspace', 'route', 'static', 'router'
    ]
    
    # Critical tools that should always be included
    critical_tools = [
        'list_devices',
        'get_device_routing_table',
        'list_adoms',
        'list_policy_packages',
        'list_firewall_policies',
        'install_policy_package'
    ]
    
    # Score tools
    scored_tools: List[Tuple[int, dict]] = []
    
    for tool in tools:
        score = 0
        tool_name = tool.get("name", "").lower()
        tool_desc = tool.get("description", "").lower()
        
        # Critical tools always get highest priority
        if tool.get("name") in critical_tools:
            score += 50
        
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
            'list': ['list', 'get', 'show', 'view', 'retrieve', 'fetch'],
            'create': ['create', 'add', 'new'],
            'update': ['update', 'modify', 'edit', 'set', 'change'],
            'delete': ['delete', 'remove'],
            'install': ['install', 'deploy', 'push'],
            'execute': ['execute', 'run', 'exec'],
            'lock': ['lock', 'unlock'],
            'commit': ['commit', 'revert'],
        }
        
        for op_type, op_keywords in operation_types.items():
            if any(op in query_lower for op in op_keywords):
                if any(op in tool_name for op in op_keywords):
                    score += 8
        
        # High-priority entity boost
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
        
        # Categorize based on actual tool names
        categories = {
            'Device Management': [t for t in tool_names if any(k in t for k in ['device', 'vdom', 'ha', 'firmware'])],
            'ADOM Management': [t for t in tool_names if 'adom' in t],
            'Policy Management': [t for t in tool_names if any(k in t for k in ['policy', 'package'])],
            'Firewall Objects': [t for t in tool_names if any(k in t for k in ['address', 'service', 'zone', 'vip']) and 'internet' not in t],
            'Security Profiles': [t for t in tool_names if any(k in t for k in ['ips', 'antivirus', 'webfilter', 'dlp', 'waf', 'profile_group'])],
            'VPN Management': [t for t in tool_names if 'vpn' in t or 'ipsec' in t],
            'SD-WAN': [t for t in tool_names if 'sdwan' in t or 'wan' in t or 'traffic_class' in t],
            'Installation': [t for t in tool_names if 'install' in t],
            'Workspace & Locking': [t for t in tool_names if any(k in t for k in ['lock', 'unlock', 'commit', 'workspace'])],
            'CLI Scripts': [t for t in tool_names if 'script' in t],
            'Monitoring & Tasks': [t for t in tool_names if any(k in t for k in ['monitor', 'status', 'log', 'task', 'statistic'])],
            'FortiGuard': [t for t in tool_names if 'fortiguard' in t or 'update' in t],
            'Internet Services': [t for t in tool_names if 'internet_service' in t],
            'Connectors': [t for t in tool_names if 'connector' in t or 'sdn' in t or 'fabric' in t],
            'Provisioning': [t for t in tool_names if 'template' in t or 'provision' in t],
            'System': [t for t in tool_names if any(k in t for k in ['system', 'backup', 'certificate', 'admin'])],
        }
        
        message = f"✅ **Connected!** Total tools: **{len(tool_names)}**\n\n**By category:**\n"
        for cat, tools_list in categories.items():
            if tools_list:
                message += f"• **{cat}:** {len(tools_list)}\n"
        
        message += (
            "\n**Example queries:**\n"
            "• List all ADOMs\n"
            "• Show devices in production ADOM\n"
            "• List policies in ADOM [name]\n"
            "• Get device status for FGT-001\n"
            "• Show internet service groups\n"
            "• Create firewall address 10.0.0.0/24\n"
            "• Install policy package to device\n\n"
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
                    "You are a FortiManager expert assistant with access to 590+ management tools.\n\n"
                    
                    "**CRITICAL INSTRUCTIONS:**\n"
                    "1. ALWAYS use available tools to get real data - NEVER give generic information\n"
                    "2. When information is missing, use discovery tools first\n"
                    "3. Follow correct multi-step workflows for complex operations\n\n"
                    
                    "**IMPORTANT WORKFLOWS:**\n\n"
                    
                    "**Policy Operations (CRITICAL):**\n"
                    "Policies are stored in PACKAGES within ADOMs. You MUST:\n"
                    "1. First call list_policy_packages(adom='<adom_name>') to see packages\n"
                    "2. Then call list_firewall_policies(adom='<adom_name>', package='<pkg_name>')\n\n"
                    "Example: User asks 'show policies in ADOM production'\n"
                    "→ Step 1: list_policy_packages(adom='production')\n"
                    "→ Step 2: For each package, list_firewall_policies(adom='production', package='pkg_name')\n\n"
                    
                    "**Router Configuration & Static Routes:**\n"
                    "CRITICAL: Static routes are stored PER-DEVICE, not at ADOM level!\n"
                    
                    "CORRECT TOOL TO USE:\n"
                    "✓ get_device_routing_table(device_name='X', adom='Y') - Gets actual routes\n"
                    
                    "TOOLS THAT DO NOT WORK:\n"
                    "✗ get_device_routing_configuration - Does NOT exist or returns errors\n"
                    "✗ list_static_routes - Does NOT exist\n"
                    "✗ list_static_route_templates - Shows templates (empty), NOT actual routes\n"
                    "✗ get_current_device_config - Shows device metadata, NOT routes\n\n"
                    
                    "Workflow to get routes:\n"
                    "1. list_devices(adom='X') - Get device names\n"
                    "2. get_device_routing_table(device_name='Y', adom='X') - Get routes for each device\n"
                    "3. Repeat step 2 for each device\n\n"
                    
                    "If get_device_routing_table is not in available tools, tell user the tool is missing.\n"
                    "NEVER repeatedly call list_static_route_templates - it will never have routes!\n\n"
                    
                    "**ADOM Operations:**\n"
                    "- list_adoms() - Show all ADOMs\n"
                    "- get_adom_statistics(adom='<name>') - Get ADOM details\n\n"
                    
                    "**Object Operations:**\n"
                    "- list_firewall_addresses(adom='<adom>') - List addresses\n"
                    "- create_firewall_address(name='X', subnet='10.0.0.0/24', adom='Y')\n"
                    "- list_internet_service_groups(adom='<adom>') - Internet services\n\n"
                    
                    "**Network Configuration:**\n"
                    "IMPORTANT: Distinguish between FortiManager's config vs managed device config!\n"
                    "- get_system_routes() → FortiManager's OWN routes (not managed devices)\n"
                    "- Device routes in ADOM → Need device-specific or ADOM-level tools\n"
                    "For static routes, interfaces, zones in managed devices:\n"
                    "- Try device-level tools: get_device_* or list in device context\n"
                    "- Try ADOM-level configuration tools\n"
                    "- Check CLI templates or provisioning templates\n"
                    "- May need to query specific devices, not FortiManager system\n\n"
                    
                    "**Installation Operations:**\n"
                    "- install_policy_package(package='X', device='Y', adom='Z') - Install policies\n"
                    "- install_device_settings(device='X', adom='Y') - Install device config\n\n"
                    
                    "**Workspace Operations (for safe editing):**\n"
                    "1. lock_adom_workspace(adom='X') - Lock before changes\n"
                    "2. Make changes...\n"
                    "3. commit_adom_workspace(adom='X') - Save changes\n"
                    "4. unlock_adom_workspace(adom='X') - Release lock\n\n"
                    
                    "**Script Operations:**\n"
                    "- list_scripts(adom='X') - Show CLI scripts\n"
                    "- execute_script(script='name', adom='X') - Run script\n\n"
                    
                    "**Monitoring:**\n"
                    "- get_system_status() - FortiManager system info\n"
                    "- list_tasks(limit=10) - Recent tasks\n"
                    "- get_task_status(task_id=123) - Check task progress\n\n"
                    
                    "**ERROR HANDLING:**\n"
                    "If a tool fails due to missing parameters:\n"
                    "- Use discovery tools first (list_policy_packages before list_firewall_policies)\n"
                    "- Check if you need to list items before accessing specific ones\n"
                    "- For policies: ALWAYS list packages first\n\n"
                    
                    "**CRITICAL - Templates vs Actual Configuration:**\n"
                    "Tools with '_template' suffix show TEMPLATES, not actual configuration!\n"
                    "- list_static_route_templates → Shows route TEMPLATES (often empty)\n"
                    "- list_static_routes or get_device_routes → Shows ACTUAL routes\n"
                    "- list_interface_templates → TEMPLATES\n"
                    "- list_interfaces or get_device_interfaces → ACTUAL config\n\n"
                    
                    "**IMPORTANT - Empty Results:**\n"
                    "If a tool returns empty or no results:\n"
                    "- DON'T immediately conclude nothing exists\n"
                    "- Check if you called a *_template tool (shows templates, not actual config)\n"
                    "- Try the non-template version: remove '_template' or try 'get_device_*'\n"
                    "- Example: list_static_route_templates is empty → try list_static_routes\n"
                    "- Try device-level tools: get_device_* or device-specific queries\n"
                    "- Try multiple variations before concluding nothing exists\n\n"
                    
                    "**CRITICAL - Use Data You Already Have:**\n"
                    "When a tool returns data, PARSE IT and USE IT - don't call more tools!\n"
                    "Common mistake: Getting full config, then calling specific config tools.\n"
                    "Example:\n"
                    "- get_current_device_config returns FULL device config (includes routes)\n"
                    "- DON'T then call get_device_routing_configuration\n"
                    "- Instead: Parse the JSON from get_current_device_config\n"
                    "- Look for 'router' section with 'static' routes\n"
                    "- Extract and present the routes from that data\n\n"
                    
                    "**EFFICIENCY - Minimize Tool Calls:**\n"
                    "Be strategic to avoid hitting iteration limits:\n"
                    "1. Call tools in logical batches when possible\n"
                    "2. If a tool returns complete data, use it - don't call more tools\n"
                    "3. For 'show X for all devices' - get device list ONCE, then query each\n"
                    "4. If you get full config with get_current_device_config, parse it - don't call more tools\n"
                    "5. Prioritize tools likely to have the data over trial-and-error\n\n"
                    
                    "**PRESENTATION:**\n"
                    "- Use markdown tables for structured data\n"
                    "- Use bullet points for lists\n"
                    "- Highlight important info with **bold**\n"
                    "- Provide context and explanations\n\n"
                    
                    "**REMEMBER:**\n"
                    "- Multi-step operations are NORMAL and EXPECTED\n"
                    "- Discovery before details: list packages before policies\n"
                    "- Always specify adom parameter when required\n"
                    "- Tool names use underscores: list_policy_packages NOT list-policy-packages"
                )
            },
            {"role": "user", "content": message.content}
        ]
        
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=openai_tools if openai_tools else None,
            tool_choice="auto",
            temperature=0.1  # Lower for more focused, deterministic responses
        )
        
        # Check if OpenAI called any tools
        if not response.choices[0].message.tool_calls:
            print("[WARN] OpenAI did not call any tools")
            print(f"[WARN] Response: {response.choices[0].message.content[:200]}")
        else:
            print(f"[DEBUG] OpenAI called {len(response.choices[0].message.tool_calls)} tools")
        
        max_iterations = 10  # Increased for complex FortiManager operations
        iteration = 0
        
        while response.choices[0].message.tool_calls and iteration < max_iterations:
            iteration += 1
            assistant_message = response.choices[0].message
            
            print(f"[INFO] Iteration {iteration}/{max_iterations}: {len(assistant_message.tool_calls)} tool calls")
            
            # Warn user if approaching limit
            if iteration >= max_iterations - 2:
                await cl.Message(
                    content=f"⚠️ Iteration {iteration}/{max_iterations} - Complex operation, may need simplification"
                ).send()
            
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
                temperature=0.1  # Lower for more focused responses
            )
        
        if iteration >= max_iterations and response.choices[0].message.tool_calls:
            await cl.Message(
                content=(
                    f"⚠️ **Reached iteration limit ({max_iterations})**\n\n"
                    "The operation was very complex. Try:\n"
                    "• Break query into smaller parts\n"
                    "• Be more specific (e.g., specify device name)\n"
                    "• Ask for one thing at a time\n"
                    "• Use simpler queries\n\n"
                    "Example: Instead of 'show everything in ADOM X',\n"
                    "try 'list devices in ADOM X' first, then ask about specific device."
                )
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