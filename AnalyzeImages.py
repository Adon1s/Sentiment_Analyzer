"""
AnalyzeImages.py - Image analysis using LM Studio (local SDK or remote API)
Usage:
python AnalyzeImages.py --image_file "TestPhoto.jpg"
"""
import fire
import base64
import requests
from pathlib import Path

# Remote LM Studio server (fallback)
LM_STUDIO_URL = "http://192.168.86.143:1234"


def analyze_image_local(image_path, model_name=None):
    """Try local LM Studio SDK first"""
    try:
        import lmstudio as lms
        model = lms.llm(model_name) if model_name else lms.llm()
        image_handle = lms.prepare_image(str(image_path))
        chat = lms.Chat()
        chat.add_user_message("Please describe this image in detail.", images=[image_handle])
        prediction = model.respond(chat)
        print("\nModel response:")
        print(prediction)
        return True
    except:
        return False


def analyze_image_remote(image_path, model_name=None):
    """Fallback to remote HTTP API"""
    with open(image_path, 'rb') as img_file:
        image_data = base64.b64encode(img_file.read()).decode('utf-8')

    payload = {
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Please describe this image in detail."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
            ]
        }],
        "max_tokens": 1000,
        "stream": False
    }

    if model_name:
        payload["model"] = model_name

    response = requests.post(f"{LM_STUDIO_URL}/v1/chat/completions", json=payload, timeout=60)
    result = response.json()
    prediction = result['choices'][0]['message']['content']
    print("\nModel response:")
    print(prediction)


def analyze_image(image_file, model_name=None):
    """
    Analyze an image using LM Studio (auto-detects local vs remote)

    :param image_file: Path to the image file
    :param model_name: Name of the VLM model to use (optional)
    """
    image_path = Path(image_file)
    if not image_path.exists():
        print(f"ERROR: File {image_file} not found!")
        return

    print(f"Analyzing image: {image_path.absolute()}")

    # Try local SDK first, fallback to remote API
    if not analyze_image_local(image_path, model_name):
        print("Using remote LM Studio server...")
        analyze_image_remote(image_path, model_name)


def main(image_file, model_name=None):
    """
    Main function to analyze images with LM Studio

    :param image_file: Path to the image file
    :param model_name: Name of the VLM model to use (optional)
    """
    print("\n=== LM Studio Image Analysis Tool ===\n")
    analyze_image(image_file, model_name)


if __name__ == "__main__":
    fire.Fire(main)