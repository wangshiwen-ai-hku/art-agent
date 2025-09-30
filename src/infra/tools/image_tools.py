import logging

logger = logging.getLogger(__name__)

# from src.utils.models import client
from src.utils.input_processor import bytes_to_raw_reference_image
from google.genai.types import GenerateImagesConfig, EditImageConfig
from langchain_core.tools import tool
from google import genai
from google.genai.types import HttpOptions
import os
from dotenv import load_dotenv
load_dotenv()
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
from dataclasses import asdict
from langchain.chat_models import init_chat_model

client = genai.Client(http_options=HttpOptions(api_version="v1"))

generation_model = "imagen-4.0-generate-001"
edit_model = "imagen-3.0-capability-001"
@tool
def generate_image_tool(prompt: str, aspect_ratio: str):
    """
    draw image with image generation model
    """
    logger.info("[draw_image_tool] 进入")
    design_prompt = prompt
    ## load multi-modal messages 
    user_prompt = f"Highly artistic typography, logo, visual arts. no text. {design_prompt}"
    logger.info(f"[draw_image_tool] 📝 user_prompt: {user_prompt}")
    image = client.models.generate_images(
        model=generation_model,
        prompt=user_prompt,
        config=GenerateImagesConfig(
            aspect_ratio=aspect_ratio,
            number_of_images=1,
            image_size="1K",
        ),
    )
    
    return image.generated_images[0].image

# @tool
def generate_image(prompt: str, aspect_ratio: str):
    """
    draw image with image generation model
    """
    logger.info("[draw_image_tool] 进入")
    design_prompt = prompt
    ## load multi-modal messages 
    user_prompt = f"Highly artistic typography, logo, visual arts. no text. {design_prompt}"
    logger.info(f"[draw_image_tool] 📝 user_prompt: {user_prompt}")
    image = client.models.generate_images(
        model=generation_model,
        prompt=user_prompt,
        config=GenerateImagesConfig(
            aspect_ratio=aspect_ratio,
            number_of_images=1,
            image_size="1K",
        ),
    )
    
    return image.generated_images[0].image

# @tool
def edit_image_tool(edit_prompt, reference_image):
    """
    params:
    edit_prompt: str
    reference_image: bytes
    """
    logger.info("[edit] 进入")
    user_prompt = edit_prompt

    ## load multi-modal messages 
    
    logger.info(f"[edit_image_tool] 📝 user_prompt: {user_prompt}")
    ## from bytes to PIL Image
    raw_ref_image = bytes_to_raw_reference_image(reference_image)

    image = client.models.edit_image(
            model=edit_model,
            reference_images=[raw_ref_image],
            prompt=user_prompt,
            config=EditImageConfig(
                edit_mode="EDIT_MODE_DEFAULT",
                number_of_images=1,
                safety_filter_level="BLOCK_MEDIUM_AND_ABOVE",
                person_generation="ALLOW_ADULT",
            ),
        )
    
    return image.generated_images[0].image


