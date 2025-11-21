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

async def test_mcp_connection():
    """Test different MCP endpoints to find the right one"""
    async with httpx.AsyncClient() as http_client:
        # Try different possible endpoints
        endpoints_to_try = [
            "/",
            "/sse",
            "/mcp",
            "/health",
            "/tools",
            "/list_tools",
        ]
        
        results = {}
        for endpoint in endpoints_to_try:
            try:
                response = await http_client.get(
                    f"{MCP_SERVER_URL}{endpoint}",
                    timeout=5.0
                )
                results[endpoint] = {
                    "status": response.status_code,
                    "content_type": response.headers.get("content-type"),
                    "body": response.text[:200]  # First 200 chars
                }
            except Exception as e:
                results[endpoint] = {"error": str(e)}
        
        return results

@cl.on_chat_start
async def start():
    """Test MCP connection when chat starts"""
    try:
        await cl.Message(content="🔍 Testing MCP server endpoints...").send()
        
        results = await test_mcp_connection()
        
        message = f"**MCP Server Test Results ({MCP_SERVER_URL}):**\n\n"
        for endpoint, result in results.items():
            if "error" in result:
                message += f"❌ `{endpoint}`: {result['error']}\n"
            else:
                message += f"✅ `{endpoint}`: Status {result['status']}, Type: {result.get('content_type')}\n"
                message += f"   Body preview: `{result['body'][:100]}...`\n\n"
        
        await cl.Message(content=message).send()
        
    except Exception as e:
        await cl.Message(
            content=f"❌ Failed to test MCP server: {str(e)}"
        ).send()
        import traceback
        print(traceback.format_exc())

@cl.on_message
async def on_message(message: cl.Message):
    await cl.Message(content="Please check the endpoint test results above first.").send()