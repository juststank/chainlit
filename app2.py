# app.py
"""
Chainlit application for FortiManager MCP Server integration with OpenAI.

This application connects to a FortiManager MCP server via HTTP/SSE transport
and enables natural language interaction with FortiManager through OpenAI's API.

Features:
- Connects to 590+ FortiManager MCP tools
- Intelligent tool filtering (OpenAI has 128 tool limit)
- Category-aware tool selection
- Real-time FortiManager operations via conversational interface

Environment Variables Required:
- OPENAI_API_KEY: Your OpenAI API key
- OPENAI_MODEL: Model to use (default: gpt-4o-mini)
"""

import os
import json
import asyncio
import chainlit as cl
from openai import OpenAI
import httpx
from typing import Optional, List, Dict, Tuple
import warnings

# Suppress httpcore async generator warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Initialize OpenAI client
client = OpenAI()
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# MCP Server Configuration
MCP_SERVER_URL = "http://localhost:8000/mcp"

# Global state
mcp_session = None
all_tools = []  # Cache all 590+ tools from FortiManager


def parse_sse_response(text: str) -> Optional[dict]:
    """
    Parse Server-Sent Events (SSE) formatted response.
    
    FortiManager MCP server returns responses in SSE format:
        event: message
        data: {"jsonrpc":"2.0","id":1,"result":{...}}
    
    Args:
        text: Raw SSE response text
        
    Returns:
        Parsed JSON-RPC response or None if parsing fails
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
                return None
    
    return None


class MCPClient:
    """
    HTTP/SSE client for Model Context Protocol (MCP) communication.
    
    Handles JSON-RPC requests to the FortiManager MCP server over HTTP
    with Server-Sent Events (SSE) for responses.
    """
    
    def __init__(self, url: str):
        """
        Initialize MCP client.
        
        Args:
            url: MCP server endpoint URL (e.g., http://localhost:8000/mcp)
        """
        self.url = url
        self.request_id = 0
        self.client = httpx.AsyncClient(timeout=30.0)
        
    async def send_request(self, method: str, params: dict = None) -> dict:
        """
        Send JSON-RPC request to MCP server and parse SSE response.
        
        Args:
            method: JSON-RPC method name (e.g., "initialize", "tools/list")
            params: Optional parameters for the method
            
        Returns:
            Result from JSON-RPC response
            
        Raises:
            Exception: If request fails or response is invalid
        """
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
                "Accept": "application/json, text/event-stream"  # Critical for SSE
            }
        ) as response:
            if response.status_code != 200:
                text = await response.aread()
                raise Exception(f"HTTP {response.status_code}: {text.decode()}")
            
            # Collect SSE response
            full_response = ""
            async for chunk in response.aiter_text():
                full_response += chunk
                
                # Check if we have a complete message (ends with double newline)
                if "\n\n" in full_response:
                    parsed = parse_sse_response(full_response)
                    if parsed:
                        print(f"[DEBUG] Parsed: {list(parsed.keys())}")
                        
                        if "error" in parsed:
                            raise Exception(f"MCP Error: {parsed['error']}")
                        
                        if "result" in parsed:
                            return parsed["result"]
                        
                        full_response = ""
            
            # Final attempt with remaining buffer
            if full_response:
                parsed = parse_sse_response(full_response)
                if parsed:
                    if "error" in parsed:
                        raise Exception(f"MCP Error: {parsed['error']}")
                    if "result" in parsed:
                        return parsed["result"]
            
            raise Exception("No valid response received")
    
    async def initialize(self) -> dict:
        """
        Initialize MCP session with server.
        
        Returns:
            Server information including name, version, and capabilities
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
        
        Returns:
            List of tool definitions with names, descriptions, and schemas
        """
        result = await self.send_request("tools/list")
        return result.get("tools", [])
    
    async def call_tool(self, name: str, arguments: dict) -> dict:
        """
        Execute a tool on the MCP server.
        
        Args:
            name: Tool name (e.g., "list_devices", "get_adom")
            arguments: Tool arguments as defined in inputSchema
            
        Returns:
            Tool execution result
        """
        return await self.send_request("tools/call", {
            "name": name,
            "arguments": arguments
        })
    
    async def close(self):
        """Close HTTP client connection."""
        await self.client.aclose()


def filter_relevant_tools(query: str, tools: List[dict], max_tools: int = 100) -> List[dict]:
    """
    Filter tools based on query relevance using category-aware scoring.
    
    With 590+ tools available, we need to intelligently select the most relevant
    subset to stay within OpenAI's 128 tool limit while maintaining quality.
    
    Scoring Strategy:
    - Category match: +15 points (e.g., "device" query → device tools)
    - Operation type match: +8 points (e.g., "list" query → list_* tools)
    - High-priority entity match: +12 points (device, policy, firewall, etc.)
    - Keyword in name: +10 points
    - Keyword in description: +3 points
    
    Args:
        query: User's natural language query
        tools: All available tools from MCP server
        max_tools: Maximum number of tools to return (default: 100)
        
    Returns:
        Filtered and scored list of most relevant tools
    """
    query_lower = query.lower()
    keywords = query_lower.split()
    
    # FortiManager tool categories based on 590-tool implementation
    # See: src/fortimanager_mcp/tools/ directory structure
    category_keywords = {
        # Device Management (65 tools)
        'device': ['device', 'firmware', 'vdom', 'ha', 'hardware', 'model', 'cluster', 'revision'],
        
        # Policy Management (50 tools)
        'policy': ['policy', 'firewall', 'rule', 'nat', 'snat', 'dnat', 'package', 'install'],
        
        # Objects Management (59 tools)
        'object': ['address', 'service', 'zone', 'vip', 'pool', 'schedule'],
        
        # Provisioning & Templates (98 tools)
        'provision': ['template', 'provision', 'profile', 'cli template', 'system template', 'certificate'],
        
        # Monitoring & Tasks (58 tools)
        'monitor': ['monitor', 'status', 'log', 'statistic', 'health', 'task', 'connectivity'],
        
        # ADOM Management (27 tools)
        'adom': ['adom', 'workspace', 'revision', 'lock', 'commit'],
        
        # Security Profiles (27 tools)
        'security': ['web filter', 'ips', 'antivirus', 'dlp', 'application control', 'waf'],
        
        # VPN Management (18 tools)
        'vpn': ['vpn', 'ipsec', 'ssl-vpn', 'tunnel', 'phase1', 'phase2'],
        
        # SD-WAN Management (19 tools)
        'sdwan': ['sd-wan', 'sdwan', 'wan', 'health check'],
        
        # FortiAP Management
        'fortiap': ['fortiap', 'wtp', 'wireless', 'wifi'],
        
        # FortiSwitch Management
        'fortiswitch': ['fortiswitch', 'switch'],
        
        # FortiExtender Management
        'fortiextender': ['fortiextender', 'extender'],
        
        # Fabric Connector Management (11 tools)
        'connector': ['connector', 'fabric', 'aws', 'azure', 'vmware', 'sdn'],
        
        # CLI Script Management (12 tools)
        'script': ['script', 'cli script', 'execute'],
        
        # FortiGuard Management (19 tools)
        'fortiguard': ['fortiguard', 'update', 'contract', 'threat', 'database'],
        
        # Internet Service Management
        'internet_service': ['internet service', 'cloud service'],
        
        # Installation Operations (16 tools)
        'installation': ['install', 'deploy', 'push', 'preview'],
    }
    
    # Detect categories from query
    detected_categories = set()
    for category, category_kws in category_keywords.items():
        if any(kw in query_lower for kw in category_kws):
            detected_categories.add(category)
    
    # Score each tool
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
        
        # Operation type matching
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
        
        # High-priority entity matches
        high_priority_entities = [
            'device', 'policy', 'firewall', 'address', 'service', 
            'adom', 'vdom', 'template', 'vpn', 'sdwan', 'ha'
        ]
        
        for entity in high_priority_entities:
            if entity in query_lower and entity in tool_name:
                score += 12
        
        if score > 0:
            scored_tools.append((score, tool))
    
    # Sort by score (highest first)
    scored_tools.sort(reverse=True, key=lambda x: x[0])
    
    # If no scored tools, return common read operations
    if not scored_tools:
        default_tools = [
            t for t in tools 
            if any(op in t.get("name", "").lower() for op in ['list', 'get'])
        ]
        return default_tools[:max_tools]
    
    # Return top scored tools
    return [tool for score, tool in scored_tools[:max_tools]]


async def init_mcp_session() -> Optional[MCPClient]:
    """
    Initialize connection to FortiManager MCP server.
    
    Returns:
        Connected MCPClient instance or None if connection fails
    """
    try:
        print(f"[DEBUG] Connecting to {MCP_SERVER_URL}...")
        
        mcp = MCPClient(MCP_SERVER_URL)
        
        # Initialize MCP session
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
    """
    Chainlit event handler: Initialize MCP connection when chat starts.
    
    This function:
    1. Validates OpenAI API key
    2. Connects to FortiManager MCP server
    3. Loads all 590+ available tools
    4. Displays categorized tool summary to user
    """
    global mcp_session, all_tools
    
    # Validate OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or "REPLACE" in api_key:
        await cl.Message(
            content="❌ **OpenAI API Key not configured!**\n\n"
                    "Please set your OpenAI API key in `.env` file:\n"
                    "```\nOPENAI_API_KEY=sk-your-actual-key-here\n```\n\n"
                    "Get your key at: https://platform.openai.com/api-keys"
        ).send()
        return
    
    await cl.Message(
        content="🔄 **Connecting to FortiManager MCP server...**\n\n"
                "*This may take a few seconds*"
    ).send()
    
    # Connect to MCP server
    try:
        mcp_session = await asyncio.wait_for(init_mcp_session(), timeout=15.0)
    except asyncio.TimeoutError:
        await cl.Message(
            content="❌ **Connection timeout**\n\n"
                    f"Could not connect to MCP server at `{MCP_SERVER_URL}` within 15 seconds.\n\n"
                    "**Troubleshooting:**\n"
                    "• Verify MCP server is running\n"
                    "• Check server URL in code\n"
                    "• Review server logs for errors"
        ).send()
        return
    
    if not mcp_session:
        await cl.Message(
            content="❌ **Failed to connect**\n\n"
                    "Check terminal logs for detailed error information."
        ).send()
        return
    
    # Load all tools from MCP server
    try:
        print("[DEBUG] Listing tools...")
        all_tools = await asyncio.wait_for(mcp_session.list_tools(), timeout=15.0)
        
        if not all_tools:
            await cl.Message(content="✅ Connected but no tools found.").send()
            return
        
        # Categorize tools for display
        tool_names = [tool.get("name", "unknown") for tool in all_tools]
        
        # Tool categories based on FortiManager MCP implementation
        # See: docs/API_COVERAGE.md for complete breakdown
        categories = {
            'Device Management': [t for t in tool_names if any(k in t.lower() for k in ['device', 'vdom', 'ha', 'firmware', 'revision'])],
            'Policy Management': [t for t in tool_names if any(k in t.lower() for k in ['policy', 'firewall', 'nat'])],
            'Objects': [t for t in tool_names if any(k in t.lower() for k in ['address', 'service']) and 'internet' not in t.lower()],
            'Provisioning & Templates': [t for t in tool_names if any(k in t.lower() for k in ['template', 'provision', 'certificate'])],
            'Security Profiles': [t for t in tool_names if any(k in t.lower() for k in ['webfilter', 'ips', 'antivirus', 'dlp', 'application'])],
            'VPN': [t for t in tool_names if 'vpn' in t.lower() or 'ipsec' in t.lower()],
            'SD-WAN': [t for t in tool_names if 'sdwan' in t.lower() or 'sd_wan' in t.lower()],
            'ADOM Management': [t for t in tool_names if 'adom' in t.lower()],
            'Monitoring & Tasks': [t for t in tool_names if any(k in t.lower() for k in ['monitor', 'status', 'log', 'statistic', 'task'])],
            'FortiGuard': [t for t in tool_names if 'fortiguard' in t.lower() or 'update' in t.lower()],
            'Internet Services': [t for t in tool_names if 'internet_service' in t.lower()],
            'CLI Scripts': [t for t in tool_names if 'script' in t.lower()],
            'Installation': [t for t in tool_names if 'install' in t.lower()],
        }
        
        # Build welcome message
        message = f"✅ **Connected to FortiManager MCP Server!**\n\n"
        message += f"**Total tools available: {len(tool_names)}**\n\n"
        
        message += "**Tools by category:**\n"
        for category, tools in categories.items():
            if tools:
                message += f"• **{category}:** {len(tools)} tools\n"
        
        # Count uncategorized tools
        categorized_count = sum(len(tools) for tools in categories.values())
        other_count = len(tool_names) - categorized_count
        if other_count > 0:
            message += f"• **Other:** {other_count} tools\n"
        
        message += "\n**Example queries:**\n"
        message += "• List all FortiGate devices\n"
        message += "• Show firewall policies in the production package\n"
        message += "• List all ADOMs\n"
        message += "• Get device status and firmware version\n"
        message += "• List internet service groups\n"
        message += "• Show VPN tunnel status\n"
        message += "• Create an address group for web servers\n"
        message += "• Install policies to a device\n\n"
        message += "*💡 Tip: Tools are intelligently filtered based on your query. " \
                   "Up to 100 most relevant tools are selected from the 590+ available.*"
        
        await cl.Message(content=message).send()
        
    except asyncio.TimeoutError:
        await cl.Message(
            content="⚠️ **Connected but timeout listing tools**\n\n"
                    "The response may be too large. Connection is established."
        ).send()
    except Exception as e:
        await cl.Message(
            content=f"⚠️ **Connected but error listing tools:** {str(e)}\n\n"
                    "Check terminal logs for details."
        ).send()
        import traceback
        traceback.print_exc()


@cl.on_message
async def on_message(message: cl.Message):
    """
    Chainlit event handler: Process user messages and execute FortiManager operations.
    
    This function:
    1. Filters relevant tools based on query (590+ → ~100)
    2. Converts tools to OpenAI function format
    3. Calls OpenAI API with filtered tools
    4. Executes tool calls via MCP
    5. Returns formatted results
    
    Args:
        message: User's message from Chainlit UI
    """
    global mcp_session, all_tools
    
    if not mcp_session:
        await cl.Message(
            content="❌ **Not connected to MCP server**\n\n"
                    "Please restart the chat to reconnect."
        ).send()
        return
    
    try:
        # Filter tools based on query relevance
        relevant_tools = filter_relevant_tools(message.content, all_tools, max_tools=100)
        
        print(f"[DEBUG] Filtered to {len(relevant_tools)} relevant tools from {len(all_tools)} total")
        if relevant_tools:
            top_tools = [t.get("name") for t in relevant_tools[:5]]
            print(f"[DEBUG] Top 5 tools: {top_tools}")
        
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
        
        # Initial OpenAI API call
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful FortiManager assistant with access to 590+ management tools. "
                    "Use the available tools to help users manage FortiGate devices, policies, objects, and more.\n\n"
                    "When presenting information:\n"
                    "- Use clear, organized formatting\n"
                    "- For lists, use tables or bullet points\n"
                    "- Highlight important details\n"
                    "- Provide context and explanations\n\n"
                    "Available categories: Device Management, Policy Management, Objects, "
                    "Provisioning, Security Profiles, VPN, SD-WAN, ADOM Management, Monitoring, and more."
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
        
        # Handle tool calls (may require multiple iterations)
        max_iterations = 5
        iteration = 0
        
        while response.choices[0].message.tool_calls and iteration < max_iterations:
            iteration += 1
            assistant_message = response.choices[0].message
            
            # Add assistant's tool calls to conversation
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
            
            # Execute each tool call
            for tool_call in assistant_message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)
                
                # Show what we're doing
                await cl.Message(
                    content=f"🔧 **Calling:** `{tool_name}`\n```json\n{json.dumps(tool_args, indent=2)}\n```"
                ).send()
                
                try:
                    # Execute tool via MCP
                    result = await mcp_session.call_tool(tool_name, tool_args)
                    
                    # Extract result content (MCP format)
                    if isinstance(result, dict):
                        if "content" in result:
                            content = result["content"]
                            if isinstance(content, list) and content:
                                # MCP returns content as array of text objects
                                tool_response = content[0].get("text", str(content))
                            else:
                                tool_response = str(content)
                        else:
                            tool_response = json.dumps(result, indent=2)
                    else:
                        tool_response = str(result)
                    
                    # Add tool result to conversation
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_response
                    })
                    
                except Exception as e:
                    error_msg = f"Error calling {tool_name}: {str(e)}"
                    print(f"[ERROR] {error_msg}")
                    await cl.Message(content=f"⚠️ {error_msg}").send()
                    
                    # Add error to conversation so OpenAI can handle it
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": error_msg
                    })
            
            # Get next response from OpenAI
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=openai_tools if openai_tools else None,
                temperature=0.2
            )
        
        # Send final response to user
        await cl.Message(content=response.choices[0].message.content).send()
        
    except Exception as e:
        await cl.Message(
            content=f"❌ **Error:** {str(e)}\n\nCheck terminal logs for details."
        ).send()
        import traceback
        traceback.print_exc()


@cl.on_chat_end
async def end():
    """
    Chainlit event handler: Cleanup when chat session ends.
    
    Properly closes the MCP client connection.
    """
    global mcp_session
    if mcp_session:
        try:
            await mcp_session.close()
            print("[INFO] MCP connection closed")
        except Exception as e:
            print(f"[ERROR] Error closing connection: {e}")