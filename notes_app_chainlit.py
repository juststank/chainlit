import chainlit as cl
from openai import AsyncOpenAI
import json
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@cl.on_chat_start
async def start():
    """Initialize the chat session"""
    cl.user_session.set("message_history", [
        {"role": "system", "content": "You are a helpful assistant. Use available tools when appropriate to help users."}
    ])
    
    await cl.Message(
        content="Hello! I'm ready to help. If you've connected any MCP servers, I can use their tools to assist you."
    ).send()

@cl.on_message
async def main(message: cl.Message):
    """Handle incoming messages with MCP tool support"""
    message_history = cl.user_session.get("message_history")
    
    if not message_history:
        await cl.Message(content="⚠️ Session not initialized. Please refresh the page.").send()
        return
    
    message_history.append({"role": "user", "content": message.content})
    
    try:
        # Get available MCP tools from Chainlit context
        available_tools = []
        
        # Check if MCP tools are available in the context
        if hasattr(cl.context, 'mcp_tools') and cl.context.mcp_tools:
            available_tools = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema
                    }
                }
                for tool in cl.context.mcp_tools
            ]
        
        # Create initial response with or without tools
        request_params = {
            "model": "gpt-4o",
            "messages": message_history,
        }
        
        if available_tools:
            request_params["tools"] = available_tools
            request_params["tool_choice"] = "auto"
        
        response = await client.chat.completions.create(**request_params)
        response_message = response.choices[0].message
        
        # Process tool calls if any
        while response_message.tool_calls:
            # Add assistant's response with tool calls to history
            message_history.append({
                "role": "assistant",
                "content": response_message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in response_message.tool_calls
                ]
            })
            
            # Execute all tool calls
            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                # Show user which tool is being called
                await cl.Message(
                    content=f"🔧 Calling tool: `{function_name}`\n```json\n{json.dumps(function_args, indent=2)}\n```",
                    author="System"
                ).send()
                
                # Call the MCP tool through Chainlit's context
                try:
                    tool_result = await cl.context.call_mcp_tool(function_name, function_args)
                    
                    # Extract text from result
                    if hasattr(tool_result, 'content') and tool_result.content:
                        content_parts = []
                        for item in tool_result.content:
                            if hasattr(item, 'text'):
                                content_parts.append(item.text)
                            else:
                                content_parts.append(str(item))
                        result_text = "\n".join(content_parts)
                    else:
                        result_text = str(tool_result)
                    
                    # Add tool result to history
                    message_history.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": function_name,
                        "content": result_text
                    })
                    
                except Exception as e:
                    error_msg = f"Error executing tool: {str(e)}"
                    print(f"❌ {error_msg}")
                    message_history.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": function_name,
                        "content": error_msg
                    })
            
            # Get next response from OpenAI
            response = await client.chat.completions.create(
                model="gpt-4o",
                messages=message_history,
                tools=available_tools if available_tools else None,
                tool_choice="auto" if available_tools else None
            )
            
            response_message = response.choices[0].message
        
        # Final response without tool calls
        if response_message.content:
            message_history.append({
                "role": "assistant",
                "content": response_message.content
            })
            await cl.Message(content=response_message.content).send()
            
    except Exception as e:
        error_msg = f"Error processing message: {type(e).__name__}: {str(e)}"
        print(f"❌ {error_msg}")
        import traceback
        traceback.print_exc()
        await cl.Message(content=f"❌ {error_msg}").send()