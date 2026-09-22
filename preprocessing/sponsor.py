"""
Sponsor module - loads BACKBOARD_API_KEY from environment variables
"""

import os
import requests
from script import process_video
import glob
import json

# Try to load python-dotenv if available for .env file support
try:
    from dotenv import load_dotenv
    try:
        load_dotenv(encoding='utf-8')
    except (UnicodeDecodeError, IOError) as e:
        print(f"Warning: Could not load .env file ({e}). Continuing without it.", file=__import__('sys').stderr)
except ImportError:
    pass

# Load BACKBOARD_API_KEY from environment
BACKBOARD_API_KEY = os.getenv('BACKBOARD_API_KEY')
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')

def createAssistant():

    create = requests.post("https://app.backboard.io/api/assistants",
        headers={
          "Content-Type": "application/json",
          "X-API-Key": BACKBOARD_API_KEY
        },
        json={
          "name": "Video Stats Analyzer",
          "description": "You are a chatbot that analyzes videos and provides detailed statistics and information.",
          "system_prompt": "You are a chatbot that analyzes videos and provides detailed statistics and information about the content.",
          "tools": [
            {
              "type": "function",
              "function": {
                "name": "process_video",
                "description": "Processes a video file to identify the main character, extract their X and Y coordinates, and optionally describe the video content. Uses AWS Bedrock TwelveLabs Pegasus model for video analysis and Gemini for coordinate extraction.",
                "parameters": {
                  "type": "object",
                  "properties": {
                    "video_path": {
                      "type": "string",
                      "description": "Path to the video file to process (required). Must be a valid file path on the server."
                    },
                    "locate_main_prompt": {
                      "type": "string",
                      "description": "Optional custom prompt text for locating the main character. If not provided, uses default from locateMain.txt file."
                    },
                    "describe_prompt": {
                      "type": "string",
                      "description": "Optional custom prompt text for describing the video content. If not provided, uses default from describe.txt file."
                    },
                    "region": {
                      "type": "string",
                      "description": "AWS region for Bedrock (e.g., 'us-east-1'). If not provided, uses AWS_REGION environment variable."
                    },
                    "twelvelabs_model_id": {
                      "type": "string",
                      "description": "TwelveLabs Pegasus model ID (e.g., 'us.twelvelabs.pegasus-1-2-v1:0'). If not provided, uses TWELVELABS_MODEL_ID environment variable."
                    },
                    "gemini_model_id": {
                      "type": "string",
                      "description": "Gemini model ID for image analysis. Default: 'google/gemini-2.0-flash-exp:free'.",
                      "default": "google/gemini-2.0-flash-exp:free"
                    },
                    "extract_coordinates": {
                      "type": "boolean",
                      "description": "Whether to extract X, Y coordinates of the main character. Default: true.",
                      "default": True
                    },
                    "raw_output": {
                      "type": "boolean",
                      "description": "Whether to return raw API responses instead of processed results. Default: false.",
                      "default": False
                    }
                  },
                  "required": ["video_path"]
                }
              }
            },
            {
              "type": "function",
              "function": {
                "name": "get_all_stats",
                "description": "Gets all statistics and information about a video file. Uses AWS Bedrock TwelveLabs Pegasus model for video analysis and Google Gemini for processing. Returns comprehensive video statistics.",
                "parameters": {
                  "type": "object",
                  "properties": {
                    "video_path": {
                      "type": "string",
                      "description": "Path to the video file to process (required). Must be a valid file path on the server."
                    },
                    "locate_main_prompt": {
                      "type": "string",
                      "description": "Optional custom prompt text for locating the main character. If not provided, uses default from locateMain.txt file."
                    },
                    "describe_prompt": {
                      "type": "string",
                      "description": "Optional custom prompt text for describing the video content. If not provided, uses default from describe.txt file."
                    },
                    "region": {
                      "type": "string",
                      "description": "AWS region for Bedrock (e.g., 'us-east-1'). If not provided, uses AWS_REGION environment variable."
                    },
                    "twelvelabs_model_id": {
                      "type": "string",
                      "description": "TwelveLabs Pegasus model ID (e.g., 'us.twelvelabs.pegasus-1-2-v1:0'). If not provided, uses TWELVELABS_MODEL_ID environment variable."
                    },
                    "gemini_model_id": {
                      "type": "string",
                      "description": "Gemini model ID for analysis. Default: 'google/gemini-2.0-flash-exp:free'.",
                      "default": "google/gemini-2.0-flash-exp:free"
                    },
                    "extract_coordinates": {
                      "type": "boolean",
                      "description": "Whether to extract coordinates. Default: true.",
                      "default": True
                    },
                    "raw_output": {
                      "type": "boolean",
                      "description": "Whether to return raw API responses instead of processed results. Default: false.",
                      "default": False
                    }
                  },
                  "required": ["video_path"]
                }
              }
            }
          ]
        }
    )

    # Check if request was successful and extract assistant ID
    if create.status_code in (200, 201):
        response_data = create.json()
        assistant_id = response_data.get('id') or response_data.get('assistant_id') or response_data.get('assistantId')
        if assistant_id:
            print(f"Assistant ID: {assistant_id}")
        else:
            print(f"Error: Could not find ID in response. Response: {response_data}")
    else:
        print(f"Error: Request failed with status {create.status_code}")
        print(f"Response: {create.text}")
        assistant_id = None

    return assistant_id

def createThread(assistant_id):
    create_thread = requests.post(
        f"https://app.backboard.io/api/assistants/{assistant_id}/threads",
        headers={
          "Content-Type": "application/json",
          "X-API-Key": BACKBOARD_API_KEY
        },
        json={}
    )

    thread_id = create_thread.json()['thread_id']
    return thread_id

def createMemory(assistant_id, memory_content):
    response = requests.post(
        f"https://app.backboard.io/api/assistants/{assistant_id}/memories",
        headers={
        "Content-Type": "application/json",
        "X-API-Key": BACKBOARD_API_KEY
        },
        json={
            "content": memory_content,
            "metadata": {
                "type": "string",
                "description": "An example of a description of the main character."
            }
        }
    )
    if response.status_code in (200, 201):
        return response.json()
    else:
        print(f"Error creating memory: {response.status_code} - {response.text}")
        return None

def getMemories(assistant_id):
    memories = requests.get(
        f"https://app.backboard.io/api/assistants/{assistant_id}/memories",
        headers={
        "X-API-Key": BACKBOARD_API_KEY
        }
    )

    return memories.json()

def queryApi(thread_id, message, llm_provider="google", model_name="gemini-2.5-flash"):
    """
    Sends a message to the assistant and handles tool calls if needed.
    Returns different formats based on the tool called:
    - process_video: Returns a formatted string from another API call using structured output
    - get_all_stats: Returns structured output (Dict[str, Any]) matching process_video format
    - Regular messages: Returns string response from the assistant
    
    Args:
        thread_id: The thread ID to send the message to
        message: The message content to send
        llm_provider: LLM provider (default: "google")
        model_name: Model name (default: "gemini-2.5-flash")
        
    Returns:
        String for process_video tool calls (formatted LLM response),
        Dict[str, Any] for get_all_stats tool calls (same format as process_video returns),
        or string containing the final response from the assistant for regular messages
    """
    import json
    import time
    
    response = requests.post(
        f"https://app.backboard.io/api/threads/{thread_id}/messages",
        headers={
          "X-API-Key": BACKBOARD_API_KEY
        },
        data={
          "content": message,
          "llm_provider": llm_provider,
          "model_name": model_name,
          "stream": "false",
          "memory": "Auto",  # "Auto" enables both reading and writing memories
          "web_search": "off",
          "send_to_llm": "true",
          "metadata": {}
        },
        files={}
    )

    if response.status_code in (200, 201):
        response_data = response.json()
        print("Response:", response_data)
        
        # Check if the assistant wants to call the process_video tool
        tool_calls = response_data.get('tool_calls')
        
        if tool_calls:
            print(f"\nTool calls detected: {len(tool_calls)}")
            
            # Handle each tool call
            for tool_call in tool_calls:
                tool_name = tool_call.get('function', {}).get('name')
                tool_call_id = tool_call.get('id')
                arguments = tool_call.get('function', {}).get('arguments', '{}')
                
                print(f"\nExecuting tool: {tool_name}")
                print(f"Arguments: {arguments}")
                
                # Parse arguments (they come as JSON string)
                try:
                    args = json.loads(arguments) if isinstance(arguments, str) else arguments
                except json.JSONDecodeError:
                    args = {}
                
                # Execute the tool function
                if tool_name == "process_video" or tool_name == "get_all_stats":
                    try:
                        # Ensure extract_coordinates uses tool default (True) if not provided
                        # process_video has default=False, but tools have default=True
                        if "extract_coordinates" not in args:
                            args["extract_coordinates"] = True
                        
                        # Both tools call process_video and return the result as-is
                        tool_result = process_video(**args)
                        # Convert result to JSON string for submission
                        tool_result_str = json.dumps(tool_result)
                    except Exception as e:
                        tool_result_str = json.dumps({"error": str(e)})
                    
                    # Submit tool result as a follow-up message to the thread
                    # Backboard might handle tool results as messages with role "tool"
                    tool_response = requests.post(
                        f"https://app.backboard.io/api/threads/{thread_id}/messages",
                        headers={
                            "X-API-Key": BACKBOARD_API_KEY
                        },
                        data={
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": tool_result_str
                        },
                        files={}
                    )
                    
                    if tool_response.status_code in (200, 201):
                        print(f"Tool result submitted successfully")
                        
                        # For process_video: make another API call to get formatted string response
                        if tool_name == "process_video":
                            # Wait a moment for tool result to be processed
                            time.sleep(1)
                            
                            # Make another API call asking LLM to format the structured output
                            format_response = requests.post(
                                f"https://app.backboard.io/api/threads/{thread_id}/messages",
                                headers={
                                    "X-API-Key": BACKBOARD_API_KEY
                                },
                                data={
                                    "content": "Please provide a clear paragraph about the video description of the main character.",
                                    "llm_provider": llm_provider,
                                    "model_name": model_name,
                                    "stream": "false",
                                    "memory": "Auto",
                                    "web_search": "off",
                                    "send_to_llm": "true",
                                    "metadata": {}
                                },
                                files={}
                            )
                            
                            if format_response.status_code in (200, 201):
                                format_data = format_response.json()
                                format_content = format_data.get('content')
                                print(f"\nFormatted response (string): {format_content or format_data}")
                                return format_content if isinstance(format_content, str) else str(format_data)
                            else:
                                # Fallback to structured output if formatting fails
                                print(f"Error formatting response: {format_response.status_code} - {format_response.text}")
                                return str(tool_result)
                        
                        # For get_all_stats: return structured output directly
                        else:
                            print(f"\nTool result (structured output): {tool_result}")
                            return tool_result
                    else:
                        print(f"Error submitting tool result: {tool_response.status_code} - {tool_response.text}")
                        print("Attempting alternative: submit as regular message...")
                        
                        # Alternative: Submit tool result as part of a continuation message
                        tool_result_message = requests.post(
                            f"https://app.backboard.io/api/threads/{thread_id}/messages",
                            headers={
                                "X-API-Key": BACKBOARD_API_KEY
                            },
                            data={
                                "content": f"Tool {tool_name} executed successfully. Result: {tool_result_str}",
                                "llm_provider": "google",
                                "model_name": "gemini-2.5-flash",
                                "stream": "false",
                                "send_to_llm": "true"
                            },
                            files={}
                        )
                        
                        if tool_result_message.status_code in (200, 201):
                            print("Tool result submitted as message")
                            tool_result_data = tool_result_message.json()
                            tool_result_content = tool_result_data.get('content')
                            return tool_result_content if isinstance(tool_result_content, str) else str(tool_result_data)
                        else:
                            return f"Error submitting tool result as message: {tool_result_message.status_code}"
                    # If tool response failed, return error
                    return f"Error submitting tool result: {tool_response.status_code} - {tool_response.text}"
                else:
                    print(f"Unknown tool: {tool_name}")
                    return f"Error: Unknown tool '{tool_name}' requested"
            # No tool calls - return the direct message content
            content = response_data.get('content')
            if content:
                return content if isinstance(content, str) else str(response_data)
            else:
                return str(response_data)
        else:
            error_msg = f"Error: Request failed with status {response.status_code}"
            print(error_msg)
            print(f"Response: {response.text}")
            return error_msg

def trainAssistant(assistant_id):
    video_paths = glob.glob("training_videos/*.mov")
    for video_path in video_paths:
        result = queryApi(thread_id, f"Please process the video at {video_path} and identify the main character.")
        createMemory(assistant_id, result)



assistant_id = createAssistant()
thread_id = createThread(assistant_id)
trainAssistant(assistant_id)


video_path = "training_videos/finalsoccer.mov"
full_stats = queryApi(
    thread_id, 
    f"Please get all stats for the video at {video_path}",
    llm_provider="google",
    model_name="gemini-2.5-flash"
)

# Format the stats to be nicer - parse JSON strings into objects
formatted_stats = {}
if isinstance(full_stats, dict):
    # Parse locateMain JSON string if it exists
    if "locateMain" in full_stats and isinstance(full_stats["locateMain"], str):
        try:
            formatted_stats["locateMain"] = json.loads(full_stats["locateMain"])
        except json.JSONDecodeError:
            formatted_stats["locateMain"] = full_stats["locateMain"]
    elif "locateMain" in full_stats:
        formatted_stats["locateMain"] = full_stats["locateMain"]
    
    # Parse describe JSON string if it exists
    if "describe" in full_stats and isinstance(full_stats["describe"], str):
        try:
            formatted_stats["describe"] = json.loads(full_stats["describe"])
        except json.JSONDecodeError:
            formatted_stats["describe"] = full_stats["describe"]
    elif "describe" in full_stats:
        formatted_stats["describe"] = full_stats["describe"]
    
    # Keep coordinates as-is (already formatted)
    if "coordinates" in full_stats:
        formatted_stats["coordinates"] = full_stats["coordinates"]
    
    # Keep any other fields
    for key, value in full_stats.items():
        if key not in formatted_stats:
            formatted_stats[key] = value
else:
    formatted_stats = full_stats

# Write formatted stats to JSON file
with open("full_stats.json", "w", encoding="utf-8") as f:
    json.dump(formatted_stats, f, ensure_ascii=False, indent=2)
print("full_stats written to full_stats.json")

