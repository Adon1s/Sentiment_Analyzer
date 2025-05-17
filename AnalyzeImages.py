"""
ImageDebug.py - Debug script for image analysis with LM Studio
Usage:
python ImageDebug.py --image_file "TestPhoto.jpg"
"""
import os
import fire
import base64
import requests
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_file_details(file_path):
    """Check details of the image file"""
    path = Path(file_path)

    logger.info(f"Checking file: {file_path}")
    logger.info(f"Absolute path: {path.absolute()}")
    logger.info(f"File exists: {path.exists()}")

    if path.exists():
        logger.info(f"File size: {path.stat().st_size} bytes")
        logger.info(f"File extension: {path.suffix}")
        return True
    return False

def try_direct_approach(image_path, api_url):
    """Try to send the image directly without base64 encoding first"""
    logger.info("Trying direct file reading approach...")

    try:
        with open(image_path, 'rb') as f:
            image_data = f.read()

        logger.info(f"Read {len(image_data)} bytes from file")

        # Try with direct binary data
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": "What's in this image?",
                    "images": [base64.b64encode(image_data).decode('utf-8')]
                }
            ],
            "stream": False
        }

        headers = {"Content-Type": "application/json"}

        logger.info(f"Sending request to {api_url}")
        response = requests.post(api_url, headers=headers, json=payload)

        logger.info(f"Response status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]

        logger.info(f"Response text: {response.text[:500]}")
        return None

    except Exception as e:
        logger.error(f"Error in direct approach: {str(e)}")
        return None

def try_openai_format(image_path, api_url):
    """Try OpenAI format for vision"""
    logger.info("Trying OpenAI vision format...")

    try:
        # Convert image to base64
        with open(image_path, 'rb') as f:
            image_data = f.read()

        base64_image = base64.b64encode(image_data).decode('utf-8')

        payload = {
            "model": "gpt-4-vision-preview",  # This is ignored by LM Studio but included for format
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What's in this image?"},
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            "max_tokens": 500
        }

        headers = {"Content-Type": "application/json"}

        logger.info(f"Sending OpenAI format request to {api_url}")
        response = requests.post(api_url, headers=headers, json=payload)

        logger.info(f"Response status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]

        logger.info(f"Response text: {response.text[:500]}")
        return None

    except Exception as e:
        logger.error(f"Error in OpenAI format approach: {str(e)}")
        return None

def try_simple_base64(image_path, api_url):
    """Try the simplest possible approach"""
    logger.info("Trying simple base64 approach...")

    try:
        # Convert image to base64
        with open(image_path, 'rb') as f:
            image_data = f.read()

        base64_image = base64.b64encode(image_data).decode('utf-8')

        # Very simple payload
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": "Describe this image in detail.",
                    "images": [base64_image]
                }
            ]
        }

        headers = {"Content-Type": "application/json"}

        logger.info(f"Sending simple request to {api_url}")
        response = requests.post(api_url, headers=headers, json=payload)

        logger.info(f"Response status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]

        logger.info(f"Response text: {response.text[:500]}")
        return None

    except Exception as e:
        logger.error(f"Error in simple base64 approach: {str(e)}")
        return None

def main(image_file, api_url="http://127.0.0.1:1234/v1/chat/completions"):
    """
    Debug image analysis with LM Studio
    :param image_file: Path to the image file
    :param api_url: LM Studio API URL
    """
    print("\n=== LM Studio Image Debug Tool ===\n")

    # 1. Check if file exists and get details
    if not check_file_details(image_file):
        print(f"ERROR: File {image_file} not found!")
        return

    print("\nTrying different approaches to send the image to LM Studio...")

    # 2. Try direct approach
    print("\n--- Attempt 1: Direct approach ---")
    direct_result = try_direct_approach(image_file, api_url)
    if direct_result:
        print("\nSUCCESS! Direct approach worked!")
        print(f"Result: {direct_result}")
        return
    else:
        print("Direct approach failed.")

    # 3. Try OpenAI format
    print("\n--- Attempt 2: OpenAI format ---")
    openai_result = try_openai_format(image_file, api_url)
    if openai_result:
        print("\nSUCCESS! OpenAI format worked!")
        print(f"Result: {openai_result}")
        return
    else:
        print("OpenAI format failed.")

    # 4. Try simple base64
    print("\n--- Attempt 3: Simple base64 ---")
    simple_result = try_simple_base64(image_file, api_url)
    if simple_result:
        print("\nSUCCESS! Simple base64 worked!")
        print(f"Result: {simple_result}")
        return
    else:
        print("Simple base64 failed.")

    print("\nAll approaches failed. Possible issues:")
    print("1. Your LM Studio model may not support vision capabilities")
    print("2. The API endpoint might be different")
    print("3. The server might have request size limitations")
    print("4. The image format may not be supported")

    print("\nSuggestions:")
    print("1. Make sure you're running a Vision-Language Model (VLM) in LM Studio")
    print("2. Try with a smaller image file")
    print("3. Check LM Studio's console for error messages")
    print("4. Try converting your image to JPEG format")

if __name__ == "__main__":
    fire.Fire(main)