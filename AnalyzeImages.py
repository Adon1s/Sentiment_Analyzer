"""
AnalyzeImages.py - Image analysis using LM Studio Python SDK
Usage:
python AnalyzeImages.py --image_file "TestPhoto.jpg"
"""
import os
import fire
import lmstudio as lms
from pathlib import Path

UPLOAD_FOLDER = r'C:\Users\Steven\PycharmProjects\Sentiment_Analyzer\received_images'


def analyze_image(image_file, model_name=None):
    """
    Analyze an image using LM Studio's Vision-Language Model

    :param image_file: Path to the image file
    :param model_name: Name of the VLM model to use (optional - uses loaded model if not specified)
    """
    # Check if file exists
    image_path = Path(image_file)
    if not image_path.exists():
        print(f"ERROR: File {image_file} not found!")
        return

    print(f"Analyzing image: {image_path.absolute()}")

    try:
        # Get the model handle
        # If no model specified, get whatever is currently loaded
        model = lms.llm(model_name) if model_name else lms.llm()

        # Prepare the image
        image_handle = lms.prepare_image(str(image_path))

        # Create a chat and add the image with a prompt
        chat = lms.Chat()
        chat.add_user_message("Please describe this image in detail.", images=[image_handle])

        # Get the model's response
        print("\nModel response:")
        prediction = model.respond(chat)
        print(prediction)

    except Exception as e:
        print(f"Error analyzing image: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Make sure LM Studio is running")
        print("2. Ensure you have a VLM model loaded in LM Studio (e.g., qwen2-vl-2b-instruct)")
        print("3. Check that the loaded model supports vision capabilities")


def main(image_file, model_name=None):
    """
    Main function to analyze images with LM Studio

    :param image_file: Path to the image file
    :param model_name: Name of the VLM model to use (optional - uses loaded model if None)
    """
    print("\n=== LM Studio Image Analysis Tool ===\n")
    analyze_image(image_file, model_name)


if __name__ == "__main__":
    fire.Fire(main)
