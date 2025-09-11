"""Metaso Image and Video Search Tool"""

import requests
import json
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Annotated, List
import uuid
import os
import dotenv
from urllib.parse import urljoin
import asyncio

from src.utils.download_image import download_images

dotenv.load_dotenv()

# API Configuration
METASO_API_URL = "https://metaso.cn/api/v1/search"
METASO_API_KEY = os.getenv("METASO_API_KEY")
IMAGE_STORE_PATH = os.getenv("IMAGE_STORE_PATH")
IMAGE_BASE_URL = os.getenv("IMAGE_BASE_URL")


class ResultImage(BaseModel):
    """Metaso API返回的相关图片"""

    id: Annotated[str, Field(description="该图片的 id")]
    description: Annotated[str, Field(description="该图片的文字描述")]
    url: Annotated[str, Field(description="该图片的文件链接")]


class ResultVideo(BaseModel):
    """Metaso API返回的相关视频"""

    id: Annotated[str, Field(description="该视频的 id")]
    name: Annotated[str, Field(description="该视频的标题")]
    description: Annotated[str, Field(description="该视频的文字描述")]
    url: Annotated[str, Field(description="该视频的文件链接")]


def _make_metaso_request(query: str, scope: str, size: int = 10) -> dict:
    """
    Helper function to make requests to the Metaso API.

    Args:
        query: Search query string
        scope: Search scope ("image" or "video")
        size: Number of results to return

    Returns:
        Parsed JSON response as dictionary
    """
    headers = {
        "Authorization": f"Bearer {METASO_API_KEY}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    data = {
        "q": query + " 高中",
        "scope": scope,
        "includeSummary": True,
        "size": str(size),
    }

    try:
        response = requests.post(METASO_API_URL, headers=headers, json=data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error making request to Metaso API: {str(e)}")
        return {}
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {str(e)}")
        return {}


@tool
def retrieve_relevant_image(keyword: str) -> List[dict]:
    """
    Retrieve relevant images based on a search query.

    Args:
        keyword: The search keyword string to find relevant images

    Returns:
        List of images
    """
    response_data = _make_metaso_request(keyword, "image", size=5)

    if not response_data or "images" not in response_data:
        return []

    results = []
    # 收集所有图片链接
    image_urls = []
    image_objects = []

    for img in response_data["images"]:
        try:
            if not img.get("imageUrl", ""):
                continue
            result_image = {
                "id": uuid.uuid4().hex,
                "name": img.get("title", ""),
                "description": img.get("title", ""),
                "url": img.get("imageUrl", ""),
            }
            image_urls.append(img.get("imageUrl", ""))
            image_objects.append(result_image)
        except Exception as e:
            print(f"Error processing image result: {str(e)}")
            continue

    # 调用批量下载
    download_results = download_images(image_urls, IMAGE_STORE_PATH, timeout=10)

    # 处理下载结果，替换URL和ID
    for i, img_obj in enumerate(image_objects):
        original_url = img_obj["url"]
        filename = download_results.get(original_url, None)

        if filename:
            img_obj["url"] = urljoin(IMAGE_BASE_URL, filename)
            img_obj["id"] = filename.split(".")[0]

        results.append(img_obj)

    return results


@tool
def retrieve_relevant_video(keyword: str) -> List[dict]:
    """
    Retrieve relevant videos based on a search query.

    Args:
        keyword: The search keyword string to find relevant videos

    Returns:
        List of videos
    """
    response_data = _make_metaso_request(keyword, "video", size=5)

    if not response_data or "videos" not in response_data:
        return []

    results = []
    for video in response_data["videos"]:
        try:
            if not video.get("link", ""):
                continue
            result_video = {
                "id": uuid.uuid4().hex,
                "name": video.get("title", ""),
                "description": video.get("snippet", ""),
                "url": video.get("link", ""),
            }
            results.append(result_video)
        except Exception as e:
            print(f"Error processing video result: {str(e)}")
            continue

    return results


async def test_metaso_tools():
    """Test function for metaso tools, using tool.ainvoke for async testing"""
    print("=== Testing Metaso Tools ===")

    test_keyword = "牛顿第一定律"

    # Test image retrieval with ainvoke
    print(
        f"\n1. Testing retrieve_relevant_image with keyword: '{test_keyword}' (using ainvoke)"
    )
    try:
        image_results = await retrieve_relevant_image.ainvoke({"keyword": test_keyword})
        print(f"Found {len(image_results)} images:")
        for i, img in enumerate(image_results, 1):
            print(f"  Image {i}:")
            print(f"    ID: {img.get('id', 'N/A')}")
            print(f"    Name: {img.get('name', 'N/A')}")
            print(f"    Description: {img.get('description', 'N/A')}")
            print(f"    URL: {img.get('url', 'N/A')}")
    except Exception as e:
        print(f"Error testing image retrieval: {str(e)}")

    # Test video retrieval with ainvoke
    print(
        f"\n2. Testing retrieve_relevant_video with keyword: '{test_keyword}' (using ainvoke)"
    )
    try:
        video_results = await retrieve_relevant_video.ainvoke({"keyword": test_keyword})
        print(f"Found {len(video_results)} videos:")
        for i, video in enumerate(video_results, 1):
            print(f"  Video {i}:")
            print(f"    ID: {video.get('id', 'N/A')}")
            print(f"    Name: {video.get('name', 'N/A')}")
            print(f"    Description: {video.get('description', 'N/A')}")
            print(f"    URL: {video.get('url', 'N/A')}")
    except Exception as e:
        print(f"Error testing video retrieval: {str(e)}")

    print("\n=== Metaso Tools Test Complete ===")


if __name__ == "__main__":
    import asyncio

    asyncio.run(test_metaso_tools())
