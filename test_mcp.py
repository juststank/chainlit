# test_mcp.py - Run this first to test the connection
import asyncio
import httpx

async def test_sse_connection():
    """Test SSE connection to MCP server"""
    url = "http://10.75.11.84:8000/mcp"
    
    async with httpx.AsyncClient() as client:
        try:
            async with client.stream(
                "GET",
                url,
                headers={
                    "Accept": "text/event-stream",
                    "Cache-Control": "no-cache",
                },
                timeout=10.0
            ) as response:
                print(f"Status: {response.status_code}")
                print(f"Headers: {response.headers}")
                
                # Read first few events
                count = 0
                async for line in response.aiter_lines():
                    print(f"Line: {line}")
                    count += 1
                    if count > 10:
                        break
                        
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_sse_connection())