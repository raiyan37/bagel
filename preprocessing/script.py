#!/usr/bin/env python3
"""
Video Processing Script using TwelveLabs Pegasus via Amazon Bedrock

This script takes a video file path and uses two prompts:
1. locateMain.txt - to locate the main person in the video
2. describe.txt - to describe the video content

Prerequisites:
- AWS credentials configured (via AWS CLI, environment variables, or IAM role)
- Access to TwelveLabs Pegasus model in Bedrock (e.g., us.twelvelabs.pegasus-1-2-v1:0)
- boto3 installed
- Python 3.7+

Usage:
    python script.py --video-path path/to/video.mp4 --region us-east-1 --model-id us.twelvelabs.pegasus-1-2-v1:0
"""

import argparse
import os
import sys
import json
import base64
import time
from typing import Dict, Any, Optional, Tuple

import boto3
from botocore.exceptions import ClientError, BotoCoreError

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("Warning: requests library not available. OpenRouter integration will not work.", file=sys.stderr)

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: OpenCV (cv2) not available. Frame extraction will not work.", file=sys.stderr)

# Try to load python-dotenv if available for .env file support
try:
    from dotenv import load_dotenv
    try:
        load_dotenv(encoding='utf-8')
    except (UnicodeDecodeError, IOError) as e:
        print(f"Warning: Could not load .env file ({e}). Continuing without it.", file=sys.stderr)
except ImportError:
    pass


def load_prompt_from_file(prompt_file: str) -> Optional[str]:
    """
    Loads prompt text from a file if it exists.
    
    Args:
        prompt_file: Path to the prompt file
        
    Returns:
        Prompt text if file exists and can be read, None otherwise
    """
    if os.path.exists(prompt_file) and os.path.isfile(prompt_file):
        try:
            with open(prompt_file, "r", encoding="utf-8") as f:
                prompt_text = f.read().strip()
                if prompt_text:
                    return prompt_text
        except (IOError, UnicodeDecodeError) as e:
            print(f"Warning: Could not read prompt file {prompt_file}: {e}", file=sys.stderr)
    return None

locate_main_prompt = load_prompt_from_file("locateMain.txt")
describe_prompt = load_prompt_from_file("describe.txt")

def extract_first_frame_as_base64(video_path: str) -> Tuple[str, int, int]:
    """
    Extracts the first frame from a video file and returns it as a base64-encoded JPEG image.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Tuple of (base64_encoded_image, width, height)
        
    Raises:
        ValueError: If OpenCV is not available or video cannot be read
    """
    if not CV2_AVAILABLE:
        raise ValueError("OpenCV (cv2) is required for frame extraction. Install with: pip install opencv-python")
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret or frame is None:
        raise ValueError(f"Could not read first frame from video: {video_path}")
    
    # Get frame dimensions
    height, width = frame.shape[:2]
    
    # Encode frame as JPEG
    _, buffer = cv2.imencode('.jpg', frame)
    
    # Convert to base64
    frame_base64 = base64.b64encode(buffer.tobytes()).decode('utf-8')
    
    return frame_base64, width, height


def read_file_as_base64(file_path: str) -> str:
    """
    Reads a local video file and returns its base64-encoded string.
    
    Args:
        file_path: Path to the video file
        
    Returns:
        Base64-encoded string of the file content
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        IOError: If the file cannot be read
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Video file not found: {file_path}")
    
    if not os.path.isfile(file_path):
        raise ValueError(f"Path is not a file: {file_path}")
    
    try:
        with open(file_path, "rb") as f:
            data = f.read()
        
        # Check file size (base64 limit is ~36 MB)
        file_size_mb = len(data) / (1024 * 1024)
        if file_size_mb > 36:
            print(f"Warning: File size ({file_size_mb:.2f} MB) exceeds recommended base64 limit (36 MB). "
                  f"Consider using S3 for larger files.", file=sys.stderr)
        
        return base64.b64encode(data).decode("utf-8")
    except IOError as e:
        raise IOError(f"Error reading file {file_path}: {e}")


def invoke_twelvelabs_api(
    video_path: str,
    prompt: str,
    region: str,
    model_id: str
) -> Dict[str, Any]:
    """
    Invokes TwelveLabs Pegasus API via Amazon Bedrock to process a video with a prompt.
    
    Args:
        video_path: Path to the local video file
        prompt: Prompt to guide the video processing
        region: AWS region for Bedrock
        model_id: Model ID for TwelveLabs Pegasus (e.g., us.twelvelabs.pegasus-1-2-v1:0)
        
    Returns:
        Response dictionary from the Bedrock API
        
    Raises:
        ClientError: For AWS API errors
    """
    bedrock = boto3.client("bedrock-runtime", region_name=region)
    
    try:
        b64_content = read_file_as_base64(video_path)
    except (FileNotFoundError, IOError, ValueError) as e:
        raise ValueError(f"Failed to read video file: {e}")
    
    request_body = {
        "inputPrompt": prompt,
        "mediaSource": {
            "base64String": b64_content
        }
    }
    
    try:
        # Start timer for API call
        start_time = time.time()
        
        response = bedrock.invoke_model(
            modelId=model_id,
            body=json.dumps(request_body),
            contentType="application/json",
            accept="application/json"
        )
        
        # End timer and calculate duration
        end_time = time.time()
        duration = end_time - start_time
        
        response_body = json.loads(response["body"].read())
        
        # Add timing info to response
        response_body["_timing"] = {
            "api_call_duration_seconds": round(duration, 2),
            "api_call_duration_formatted": f"{duration:.2f}s"
        }
        
        print(f"API call completed in {duration:.2f} seconds", file=sys.stderr)
        
        return response_body
        
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "Unknown")
        error_message = e.response.get("Error", {}).get("Message", str(e))
        raise ClientError(
            {"Error": {"Code": error_code, "Message": error_message}},
            "invoke_model"
        )


def extract_description(response: Dict[str, Any]) -> str:
    """
    Extracts the text description from the Bedrock API response.
    
    Different model versions may return the description in different fields.
    This function handles various response formats.
    
    Args:
        response: Response dictionary from Bedrock API
        
    Returns:
        Extracted text description
    """
    # Try common response fields
    possible_fields = ["text", "generatedText", "description", "output", "content", "message"]
    
    for field in possible_fields:
        if field in response:
            if isinstance(response[field], str):
                return response[field]
            elif isinstance(response[field], dict) and "text" in response[field]:
                return response[field]["text"]
    
    # If no standard field found, return the full response as JSON
        return json.dumps(response, indent=2)


def invoke_gemini_api_with_image(
    image_base64: str,
    prompt_text: str,
    region: str,
    model_id: str = "google.gemma-3-4b-it"
) -> Dict[str, Any]:
    """
    Invokes Gemma API via Amazon Bedrock to analyze an image with a text prompt.
    According to AWS Bedrock documentation, Gemma models support image input.
    
    Args:
        image_base64: Base64-encoded JPEG image
        prompt_text: Text prompt to guide the analysis
        region: AWS region for Bedrock
        model_id: Gemma model ID (default: google.gemma-3-4b-it)
        
    Returns:
        Response dictionary from the Bedrock API
        
    Raises:
        ClientError: For AWS API errors
    """
    bedrock = boto3.client("bedrock-runtime", region_name=region)
    
    # Gemma models in Bedrock use "messages" format
    # Content array contains objects with "text" for text and "image" for images
    request_body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "text": prompt_text
                    },
                    {
                        "image": {
                            "format": "jpeg",
                            "source": {
                                "bytes": image_base64
                            }
                        }
                    }
                ]
            }
        ]
    }
    
    try:
        # Start timer for API call
        start_time = time.time()
        
        response = bedrock.invoke_model(
            modelId=model_id,
            body=json.dumps(request_body),
            contentType="application/json",
            accept="application/json"
        )
        
        # End timer and calculate duration
        end_time = time.time()
        duration = end_time - start_time
        
        response_body = json.loads(response["body"].read())
        
        # Add timing info to response
        response_body["_timing"] = {
            "api_call_duration_seconds": round(duration, 2),
            "api_call_duration_formatted": f"{duration:.2f}s"
        }
        
        print(f"Gemma API call completed in {duration:.2f} seconds", file=sys.stderr)
        
        return response_body
        
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "Unknown")
        error_message = e.response.get("Error", {}).get("Message", str(e))
        raise ClientError(
            {"Error": {"Code": error_code, "Message": error_message}},
            "invoke_model"
        )


def invoke_gemini_via_openrouter(
    image_base64: str,
    prompt_text: str,
    model_id: str = "google/gemini-3-flash-preview"
) -> Dict[str, Any]:
    """
    Invokes Gemini API via OpenRouter to analyze an image with a text prompt.
    Uses the exact format from OpenRouter documentation.
    Automatically switches to alternative models if rate-limited.
    
    Args:
        image_base64: Base64-encoded JPEG image (without data URL prefix)
        prompt_text: Text prompt to guide the analysis
        model_id: Gemini model ID on OpenRouter (default: google/gemini-3-flash-preview)
        
    Returns:
        Response dictionary compatible with extract_coordinates_from_gemini_response
        
    Raises:
        ValueError: If requests library is not available or API key is missing
        Exception: For API errors
    """
    if not REQUESTS_AVAILABLE:
        raise ValueError("requests library is required for OpenRouter integration. Install with: pip install requests")
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set")
    
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Ensure base64 string is clean (no whitespace)
    image_base64_clean = ''.join(image_base64.split())
    
    # Validate base64 string
    try:
        base64.b64decode(image_base64_clean, validate=True)
    except Exception as e:
        raise ValueError(f"Invalid base64 image data: {e}")
    
    # Format exactly as shown in OpenRouter docs: data:image/jpeg;base64,{base64}
    data_url = f"data:image/jpeg;base64,{image_base64_clean}"
    
    # Build messages exactly as per OpenRouter documentation
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt_text
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": data_url
                    }
                }
            ]
        }
    ]
    
    # List of fallback models to try if rate-limited (in order of preference)
    fallback_models = [
        "google/gemini-3-flash-preview",
        "google/gemini-2.5-flash",
        "google/gemini-2.5-pro",
        "google/gemini-1.5-flash",
        "google/gemini-1.5-pro"
    ]
    
    # Remove current model from fallback list if it's already there
    fallback_models = [m for m in fallback_models if m != model_id]
    
    # Start timer for API call
    start_time = time.time()
    
    # Try the requested model first, then fallbacks if rate-limited
    models_to_try = [model_id] + fallback_models
    
    try:
        for attempt_model in models_to_try:
            print(f"Debug: Making request to OpenRouter with model: {attempt_model}", file=sys.stderr)
            
            # Payload exactly as per docs - simple format with just model and messages
            payload = {
                "model": attempt_model,
                "messages": messages
            }
            
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=60)
                
                # If successful, break out of loop
                if response.ok:
                    end_time = time.time()
                    duration = end_time - start_time
                    break
                
                # Check if it's a rate limit error (429)
                if response.status_code == 429:
                    try:
                        error_detail = response.json()
                        raw_error = error_detail.get("error", {}).get("metadata", {}).get("raw", "")
                        print(f"Debug: Model {attempt_model} is rate-limited. Trying next model...", file=sys.stderr)
                        
                        # If this is not the last model, continue to next one
                        if attempt_model != models_to_try[-1]:
                            continue
                        else:
                            # Last model also rate-limited
                            error_msg = error_detail.get("error", {}).get("message", "Rate limit exceeded")
                            raise requests.exceptions.HTTPError(
                                f"All models rate-limited. Last error: {error_msg}\n"
                                f"Tried models: {', '.join(models_to_try)}\n"
                                f"Consider adding your own Google API key at: https://openrouter.ai/settings/integrations"
                            )
                    except requests.exceptions.HTTPError:
                        raise
                    except:
                        # If we can't parse the error, try next model if available
                        if attempt_model != models_to_try[-1]:
                            continue
                        else:
                            raise requests.exceptions.HTTPError(
                                f"Rate limit error on all models. Tried: {', '.join(models_to_try)}"
                            )
                
                # For other errors, don't try other models
                try:
                    error_detail = response.json()
                    error_msg = error_detail.get("error", {}).get("message", f"HTTP {response.status_code}")
                    print(f"Debug: OpenRouter API error response: {json.dumps(error_detail, indent=2)}", file=sys.stderr)
                except:
                    error_msg = response.text or f"HTTP {response.status_code}"
                raise requests.exceptions.HTTPError(f"OpenRouter API error: {error_msg}")
                
            except requests.exceptions.HTTPError:
                # Re-raise HTTP errors (already handled above)
                raise
            except requests.exceptions.RequestException as e:
                # For other request errors, try next model if available
                if attempt_model != models_to_try[-1]:
                    print(f"Debug: Error with model {attempt_model}: {e}. Trying next model...", file=sys.stderr)
                    continue
                else:
                    raise
        
        # Check if we got a successful response (from any model)
        if not response.ok:
            # This shouldn't happen if we handled errors above, but just in case
            try:
                error_detail = response.json()
                error_msg = error_detail.get("error", {}).get("message", f"HTTP {response.status_code}")
                print(f"Debug: OpenRouter API error response: {json.dumps(error_detail, indent=2)}", file=sys.stderr)
            except:
                error_msg = response.text or f"HTTP {response.status_code}"
            raise requests.exceptions.HTTPError(f"OpenRouter API error: {error_msg}")
        
        # Calculate final duration if not already done
        if 'duration' not in locals():
            end_time = time.time()
            duration = end_time - start_time
        
        response.raise_for_status()
        response_body = response.json()
        
        # Transform OpenRouter response to format compatible with extract_coordinates_from_gemini_response
        # OpenRouter returns OpenAI-compatible format: choices[0].message.content
        transformed_response = {
            "_timing": {
                "api_call_duration_seconds": round(duration, 2),
                "api_call_duration_formatted": f"{duration:.2f}s"
            }
        }
        
        # Extract content from OpenRouter response
        if "choices" in response_body and len(response_body["choices"]) > 0:
            message = response_body["choices"][0].get("message", {})
            content = message.get("content", "")
            
            # Format as messages array to be compatible with existing extraction logic
            transformed_response["messages"] = [
                {
                    "role": "assistant",
                    "content": content
                }
            ]
        else:
            # Fallback: return raw response
            transformed_response["raw_response"] = response_body
        
        print(f"OpenRouter Gemini API call completed in {duration:.2f} seconds", file=sys.stderr)
        
        return transformed_response
    
    except requests.exceptions.RequestException as e:
        error_message = str(e)
        full_error_detail = None
        if hasattr(e, "response") and e.response is not None:
            try:
                error_detail = e.response.json()
                # Try multiple possible error formats from OpenRouter
                if "error" in error_detail:
                    if isinstance(error_detail["error"], dict):
                        error_message = error_detail["error"].get("message", error_message)
                    else:
                        error_message = str(error_detail["error"])
                # Also check for provider-specific errors
                if "provider" in error_detail:
                    provider_error = error_detail.get("provider", {})
                    if isinstance(provider_error, dict):
                        provider_msg = provider_error.get("error", {}).get("message", "")
                        if provider_msg:
                            error_message = f"{error_message} | Provider error: {provider_msg}"
                # Store full error for debugging
                full_error_detail = error_detail
            except Exception as json_error:
                error_message = e.response.text or error_message
        
        # Include full error details in exception message for better debugging
        if full_error_detail:
            raise Exception(f"OpenRouter API error: {error_message}\nFull error details: {json.dumps(full_error_detail, indent=2)}")
        else:
            raise Exception(f"OpenRouter API error: {error_message}")


def extract_coordinates_from_gemini_response(response: Dict[str, Any]) -> Optional[Dict[str, float]]:
    """
    Extracts coordinates from Gemma/Gemini API response.
    
    Args:
        response: Response dictionary from API
        
    Returns:
        Dictionary with 'x' and 'y' coordinates if found, None otherwise
    """
    # Try to extract text from response - check multiple formats
    text_content = None
    
    # Check for messages format (Gemma/Gemini format)
    if "messages" in response and len(response["messages"]) > 0:
        message = response["messages"][-1]  # Get last message
        if "content" in message:
            if isinstance(message["content"], str):
                text_content = message["content"]
            elif isinstance(message["content"], list):
                for content_item in message["content"]:
                    if isinstance(content_item, dict):
                        if "text" in content_item:
                            text_content = content_item["text"]
                            break
                        elif isinstance(content_item.get("content"), str):
                            text_content = content_item["content"]
                            break
    
    # Check for candidates format (older Gemini format)
    if not text_content and "candidates" in response and len(response["candidates"]) > 0:
        candidate = response["candidates"][0]
        if "content" in candidate:
            if isinstance(candidate["content"], str):
                text_content = candidate["content"]
            elif isinstance(candidate["content"], dict):
                if "parts" in candidate["content"]:
                    for part in candidate["content"]["parts"]:
                        if "text" in part:
                            text_content = part["text"]
                            break
                elif "text" in candidate["content"]:
                    text_content = candidate["content"]["text"]
    
    if not text_content:
        return None
    
    # Try to parse JSON from text
    try:
        # Look for JSON-like structures in the text
        import re
        json_match = re.search(r'\{[^{}]*"x"[^{}]*"y"[^{}]*\}', text_content)
        if json_match:
            coords = json.loads(json_match.group())
            if "x" in coords and "y" in coords:
                return {"x": float(coords["x"]), "y": float(coords["y"])}
    except (json.JSONDecodeError, ValueError):
        pass
    
    # Try to extract numbers from text
    import re
    numbers = re.findall(r'\d+\.?\d*', text_content)
    if len(numbers) >= 2:
        try:
            return {"x": float(numbers[0]), "y": float(numbers[1])}
        except ValueError:
            pass
    
    return None


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Process video using TwelveLabs Pegasus via Amazon Bedrock with locateMain and describe prompts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a video file
  python script.py --video-path video.mp4 --region us-east-1 --model-id us.twelvelabs.pegasus-1-2-v1:0
        """
    )
    
    parser.add_argument(
        "--video-path",
        type=str,
        required=True,
        help="Path to local video file"
    )
    
    # Required arguments (can also come from environment variables)
    parser.add_argument(
        "--region",
        type=str,
        default=os.getenv("AWS_REGION"),
        required=not bool(os.getenv("AWS_REGION")),
        help="AWS region for Bedrock (e.g., us-east-1). Can also be set via AWS_REGION environment variable."
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=os.getenv("TWELVELABS_MODEL_ID"),
        required=not bool(os.getenv("TWELVELABS_MODEL_ID")),
        help="TwelveLabs Pegasus model ID (e.g., us.twelvelabs.pegasus-1-2-v1:0). Can also be set via TWELVELABS_MODEL_ID environment variable."
    )
    
    parser.add_argument(
        "--raw-output",
        action="store_true",
        help="Output the full raw JSON response instead of just the description text"
    )
    
    parser.add_argument(
        "--gemini-model-id",
        type=str,
        default=os.getenv("GEMINI_MODEL_ID", "google/gemini-2.0-flash-exp:free"),
        help="Model ID for image analysis via OpenRouter (default: google/gemini-2.0-flash-exp:free). Can also be set via GEMINI_MODEL_ID environment variable."
    )
    
    return parser.parse_args()


def process_video(
    video_path: str,
    locate_main_prompt: Optional[str] = None,
    describe_prompt: Optional[str] = None,
    region: str = None,
    twelvelabs_model_id: str = None,
    gemini_model_id: str = "google/gemini-2.0-flash-exp:free",
    extract_coordinates: bool = False,
    raw_output: bool = False
) -> Dict[str, Any]:
    """
    High-level function to process a video with locateMain and describe prompts.
    
    This function abstracts the video processing logic and can be imported from other modules.
    
    Args:
        video_path: Path to the video file
        locate_main_prompt: Prompt text for locating the main character (if None, loads from locateMain.txt)
        describe_prompt: Prompt text for describing the video (if None, loads from describe.txt)
        region: AWS region for Bedrock (if None, uses AWS_REGION env var or defaults to us-east-1)
        twelvelabs_model_id: TwelveLabs Pegasus model ID (if None, uses TWELVELABS_MODEL_ID env var)
        gemini_model_id: Gemini model ID for image analysis (default: google/gemini-2.0-flash-exp:free)
        extract_coordinates: Whether to extract coordinates using Gemini (default: True)
        raw_output: Whether to return raw API responses (default: False)
        
    Returns:
        Dictionary with keys:
        - "locateMain": Description of the main character (or raw response if raw_output=True)
        - "coordinates": Dictionary with x, y coordinates and image dimensions (if extract_coordinates=True)
        - "describe": Video description (if describe_prompt provided)
        
    Raises:
        FileNotFoundError: If video file doesn't exist
        ValueError: If prompts can't be loaded or required args missing
        ClientError: For AWS API errors
    """
    # Validate video path
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Get region and model ID from environment if not provided
    if region is None:
        region = os.getenv("AWS_REGION")
        if not region:
            raise ValueError("AWS region not provided and AWS_REGION environment variable not set")
    
    if twelvelabs_model_id is None:
        twelvelabs_model_id = os.getenv("TWELVELABS_MODEL_ID")
        if not twelvelabs_model_id:
            raise ValueError("TwelveLabs model ID not provided and TWELVELABS_MODEL_ID environment variable not set")
    
    # Load prompts from files if not provided
    if locate_main_prompt is None:
        locate_main_prompt = load_prompt_from_file("locateMain.txt")
    
    if describe_prompt is None:
        describe_prompt = load_prompt_from_file("describe.txt")
    
    if not locate_main_prompt and not describe_prompt:
        raise ValueError("Neither locate_main_prompt nor describe_prompt provided, and prompt files not found")
    
    results = {}
    locate_main_description = None
    
    # Process with locateMain prompt if available
    if locate_main_prompt:
        response = invoke_twelvelabs_api(
            video_path=video_path,
            prompt=locate_main_prompt,
            region=region,
            model_id=twelvelabs_model_id
        )
        
        if raw_output:
            results["locateMain"] = response
        else:
            locate_main_description = extract_description(response)
            results["locateMain"] = locate_main_description
            
            # Extract first frame and get coordinates using Gemini
            if locate_main_description and extract_coordinates and CV2_AVAILABLE:
                try:
                    frame_base64, frame_width, frame_height = extract_first_frame_as_base64(video_path)
                    
                    # Create prompt for Gemini to find coordinates
                    gemini_prompt = f"""Based on this description of the main character: "{locate_main_description}"

Please identify the main character in this image and provide their X and Y coordinates in JSON format: {{"x": <number>, "y": <number>}}.

Coordinates should be pixel positions where (0, 0) is the top-left corner. X increases to the right, Y increases downward.

Provide only the JSON with x and y coordinates."""
                    
                    gemini_response = invoke_gemini_via_openrouter(
                        image_base64=frame_base64,
                        prompt_text=gemini_prompt,
                        model_id=gemini_model_id
                    )
                    
                    if raw_output:
                        results["coordinates"] = gemini_response
                    else:
                        coords = extract_coordinates_from_gemini_response(gemini_response)
                        if coords:
                            results["coordinates"] = coords
                            results["coordinates"]["image_width"] = frame_width
                            results["coordinates"]["image_height"] = frame_height
                        else:
                            # Fallback: include raw text response
                            results["coordinates"] = {"error": "Could not extract coordinates", "raw_response": gemini_response}
                except Exception as e:
                    results["coordinates"] = {"error": str(e)}
    
    # Process with describe prompt if available
    if describe_prompt:
        response = invoke_twelvelabs_api(
            video_path=video_path,
            prompt=describe_prompt,
            region=region,
            model_id=twelvelabs_model_id
        )
        
        if raw_output:
            results["describe"] = response
        else:
            results["describe"] = extract_description(response)
    
    return results


def validate_arguments(args: argparse.Namespace) -> None:
    """Validates command-line arguments."""
    if args.video_path and not os.path.exists(args.video_path):
        raise FileNotFoundError(f"Video file not found: {args.video_path}")


def main() -> int:
    """Main entry point for the script."""
    try:
        args = parse_arguments()
        validate_arguments(args)
        
        # Load prompts from files
        
        
        if not locate_main_prompt:
            print("Warning: locateMain.txt not found or empty. Skipping locate main prompt.", file=sys.stderr)
        if not describe_prompt:
            print("Warning: describe.txt not found or empty. Skipping describe prompt.", file=sys.stderr)
        
        if not locate_main_prompt and not describe_prompt:
            print("Error: Neither locateMain.txt nor describe.txt could be loaded.", file=sys.stderr)
            return 1
        
        results = {}
        locate_main_description = None
        
        # Process with locateMain prompt if available
        if locate_main_prompt:
            print(f"Processing video with locateMain prompt...", file=sys.stderr)
            response = invoke_twelvelabs_api(
                video_path=args.video_path,
                prompt=locate_main_prompt,
                region=args.region,
                model_id=args.model_id
            )
            # response = "The main speaker is the person in the center of the frame."
            
            if args.raw_output:
                results["locateMain"] = response
            else:
                locate_main_description = extract_description(response)
                results["locateMain"] = locate_main_description
                
                # Extract first frame and get coordinates using Gemini
                if locate_main_description and CV2_AVAILABLE:
                    try:
                        print(f"Extracting first frame from video...", file=sys.stderr)
                        frame_base64, frame_width, frame_height = extract_first_frame_as_base64(args.video_path)
                        
                        # Create prompt for Gemini to find coordinates
                        gemini_prompt = f"""Based on this description of the main speaker: "{locate_main_description}"

Please identify the main speaker in this image and provide their X and Y coordinates in JSON format: {{"x": <number>, "y": <number>}}.

Coordinates should be pixel positions where (0, 0) is the top-left corner. X increases to the right, Y increases downward.

Provide only the JSON with x and y coordinates."""
                        
                        print(f"Analyzing first frame with Gemini via OpenRouter to find coordinates...", file=sys.stderr)
                        gemini_response = invoke_gemini_via_openrouter(
                            image_base64=frame_base64,
                            prompt_text=gemini_prompt,
                            model_id=args.gemini_model_id
                        )
                        
                        if args.raw_output:
                            results["coordinates"] = gemini_response
                        else:
                            coords = extract_coordinates_from_gemini_response(gemini_response)
                            if coords:
                                results["coordinates"] = coords
                                results["coordinates"]["image_width"] = frame_width
                                results["coordinates"]["image_height"] = frame_height
                            else:
                                # Fallback: include raw text response
                                results["coordinates"] = {"error": "Could not extract coordinates", "raw_response": gemini_response}
                    except Exception as e:
                        print(f"Warning: Could not extract coordinates using Gemini: {e}", file=sys.stderr)
                        results["coordinates"] = {"error": str(e)}
        
        # Process with describe prompt if available
        if describe_prompt:
            print(f"Processing video with describe prompt...", file=sys.stderr)
            response = invoke_twelvelabs_api(
                video_path=args.video_path,
                prompt=describe_prompt,
                region=args.region,
                model_id=args.model_id
            )
            
            if args.raw_output:
                results["describe"] = response
            else:
                results["describe"] = extract_description(response)
        
        # Output results as JSON
        print(json.dumps(results, indent=2))
        
        return 0
        
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "Unknown")
        error_message = e.response.get("Error", {}).get("Message", str(e))
        print(f"AWS Error ({error_code}): {error_message}", file=sys.stderr)
        return 1
    except BotoCoreError as e:
        print(f"AWS SDK Error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

