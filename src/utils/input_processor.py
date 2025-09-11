import logging
import base64
import mimetypes
import requests
import json
import time
import uuid
import os
from typing import Dict, List, Optional, Union, Any
from urllib.parse import urlparse

from langchain_core.messages import HumanMessage

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()


_processor = None


class MultimodalInputProcessor:
    """
    Process various modalities of data into HumanMessage objects for LangChain agents.
    Designed for OpenAI models with proper error handling and validation.

    Based on LangChain multimodal input documentation:
    https://python.langchain.com/docs/how_to/multimodal_inputs/
    """

    # Supported MIME types
    SUPPORTED_IMAGE_TYPES = {
        "image/jpeg",
        "image/jpg",
        "image/png",
        "image/gif",
        "image/webp",
        "image/svg+xml",
        "image/bmp",
    }

    SUPPORTED_AUDIO_TYPES = {
        "audio/mpeg",
        "audio/wav",
        "audio/m4a",
        "audio/aac",
        "audio/ogg",
        "audio/flac",
        "audio/webm",
    }

    SUPPORTED_VIDEO_TYPES = {
        "video/mp4",
        "video/mpeg",
        "video/quicktime",
        "video/x-msvideo",  # AVI
        "video/webm",
        "video/x-ms-wmv",  # WMV
        "video/x-flv",  # FLV
        "video/3gpp",  # 3GP
        "video/x-matroska",  # MKV
    }

    SUPPORTED_PDF_TYPES = {"application/pdf"}

    def __init__(self):
        """Initialize the input processor"""
        pass

    def create_human_message(
        self,
        text: str,
        image_data: Optional[str] = None,
        image_url: Optional[str] = None,
        audio_url: Optional[str] = None,
        **kwargs,
    ) -> HumanMessage:
        """
        Create a HumanMessage with multimodal content blocks.

        Args:
            text: Text content for the message
            image_data: Base64 encoded image data
            image_url: URL to an image
            audio_data: Base64 encoded audio data
            audio_url: URL to an audio file
            video_data: Base64 encoded video data
            video_url: URL to a video file
            **kwargs: Additional arguments

        Returns:
            HumanMessage object with multimodal content

        Raises:
            ValueError: For various validation errors
        """
        content_blocks = []
        if audio_url:
            transcribed_text = self._convert_audio_to_text(audio_url=audio_url)
            text = text + f"\n\n<上传音频的文字转录内容>\n{transcribed_text}\n</上传音频的文字转录内容>"
            content_blocks.append({"type": "text", "text": text})
        else:
            content_blocks.append({"type": "text", "text": text})

        # Process image inputs
        if image_data:
            content_blocks.append(self._process_image_base64(image_data))
        if image_url:
            content_blocks.append(self._process_image_url(image_url))

        # If only text provided, return simple text message
        if len(content_blocks) == 1 and content_blocks[0]["type"] == "text":
            return HumanMessage(content=text)

        # Return multimodal message
        return HumanMessage(content=content_blocks)

    def _process_image_base64(self, image_data: str) -> Dict[str, Any]:
        """Process base64 image data"""
        # Try to decode to verify it's valid
        decoded = base64.b64decode(image_data)

        # Detect MIME type from data
        mime_type = self._detect_image_mime_type(decoded)

        if mime_type not in self.SUPPORTED_IMAGE_TYPES:
            raise ValueError(f"Unsupported image format: {mime_type}")

        return {
            "type": "image",
            "source_type": "base64",
            "data": image_data,
            "mime_type": mime_type,
        }

    def _process_image_url(self, image_url: str) -> Dict[str, Any]:
        """Process image URL"""
        # Validate URL format
        parsed_url = urlparse(image_url)
        if not parsed_url.scheme or not parsed_url.netloc:
            raise ValueError("Invalid image URL format")

        # Try to fetch the image to validate it exists and get MIME type
        response = requests.head(image_url, timeout=10)
        response.raise_for_status()

        content_type = response.headers.get("content-type", "").lower()
        if content_type not in self.SUPPORTED_IMAGE_TYPES:
            raise ValueError(f"Unsupported image format: {content_type}")

        return {
            "type": "image",
            "source_type": "url",
            "url": image_url,
            "mime_type": content_type,
        }

    def _detect_image_mime_type(self, data: bytes) -> str:
        """Detect MIME type from image data"""
        # Check magic bytes for common image formats
        if data.startswith(b"\xff\xd8\xff"):
            return "image/jpeg"
        elif data.startswith(b"\x89PNG"):
            return "image/png"
        elif data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
            return "image/gif"
        elif data.startswith(b"RIFF") and b"WEBP" in data[:12]:
            return "image/webp"
        elif data.startswith(b"BM"):
            return "image/bmp"
        else:
            # Default to JPEG if can't detect
            return "image/jpeg"

    def _convert_audio_to_text(self, audio_url: str) -> str:
        """
        Convert audio to text using ByteDance Volcano Engine ASR service.

        Args:
            audio_url: URL to audio file

        Returns:
            Transcribed text from audio
        """
        return self._convert_audio_with_volcano_asr(audio_url)

    def _submit_asr_task(self, file_url: str) -> tuple[str, str]:
        """
        Submit ASR task to ByteDance Volcano Engine

        Args:
            file_url: URL to the audio/video file

        Returns:
            Tuple of (task_id, x_tt_logid)
        """
        appid = os.getenv("VOLC_APP_ID")
        token = os.getenv("VOLC_ACCESS_KEY")

        if not appid or not token:
            raise ValueError(
                "VOLC_APP_ID and VOLC_ACCESS_KEY environment variables must be set"
            )

        submit_url = "https://openspeech-direct.zijieapi.com/api/v3/auc/bigmodel/submit"
        task_id = str(uuid.uuid4())

        headers = {
            "X-Api-App-Key": appid,
            "X-Api-Access-Key": token,
            "X-Api-Resource-Id": "volc.bigasr.auc",
            "X-Api-Request-Id": task_id,
            "X-Api-Sequence": "-1",
        }

        request = {
            "user": {"uid": "fake_uid"},
            "audio": {
                "url": file_url,
            },
            "request": {
                "model_name": "bigmodel",
                "enable_channel_split": True,
                "enable_ddc": True,
                "enable_speaker_info": True,
                "enable_punc": True,
                "enable_itn": True,
                "corpus": {"correct_table_name": "", "context": ""},
            },
        }

        logger.info(f"Submitting ASR task with ID: {task_id}")
        response = requests.post(submit_url, data=json.dumps(request), headers=headers)

        if (
            "X-Api-Status-Code" in response.headers
            and response.headers["X-Api-Status-Code"] == "20000000"
        ):
            logger.info(
                f'ASR task submitted successfully: {response.headers["X-Api-Status-Code"]}'
            )
            x_tt_logid = response.headers.get("X-Tt-Logid", "")
            return task_id, x_tt_logid
        else:
            logger.error(f"ASR task submission failed: {response.headers}")
            raise Exception(f"ASR task submission failed: {response.headers}")

    def _query_asr_task(self, task_id: str, x_tt_logid: str) -> requests.Response:
        """
        Query ASR task result from ByteDance Volcano Engine

        Args:
            task_id: Task ID from submission
            x_tt_logid: Log ID from submission

        Returns:
            Response object
        """
        appid = os.getenv("VOLC_APP_ID")
        token = os.getenv("VOLC_ACCESS_KEY")

        if not appid or not token:
            raise ValueError(
                "VOLC_APP_ID and VOLC_ACCESS_KEY environment variables must be set"
            )

        query_url = "https://openspeech-direct.zijieapi.com/api/v3/auc/bigmodel/query"

        headers = {
            "X-Api-App-Key": appid,
            "X-Api-Access-Key": token,
            "X-Api-Resource-Id": "volc.bigasr.auc",
            "X-Api-Request-Id": task_id,
            "X-Tt-Logid": x_tt_logid,
        }

        response = requests.post(query_url, json.dumps({}), headers=headers)
        logger.debug(
            f'ASR query response status: {response.headers.get("X-Api-Status-Code")}'
        )

        return response

    def _convert_audio_with_volcano_asr(self, file_url: str) -> str:
        """
        Convert audio to text using ByteDance Volcano Engine ASR

        Args:
            file_url: URL to the audio/video file

        Returns:
            Transcribed text
        """
        task_id, x_tt_logid = self._submit_asr_task(file_url)

        # Poll for results with exponential backoff
        max_attempts = 60  # Maximum attempts (about 5 minutes with exponential backoff)
        attempt = 0
        sleep_time = 1

        while attempt < max_attempts:
            query_response = self._query_asr_task(task_id, x_tt_logid)
            code = query_response.headers.get("X-Api-Status-Code", "")

            if code == "20000000":  # Task finished successfully
                result = query_response.json()
                logger.info("ASR task completed successfully")

                # Extract transcribed text from result
                if "data" in result and "utterances" in result["data"]:
                    utterances = result["data"]["utterances"]
                    transcribed_text = " ".join(
                        [utt.get("text", "") for utt in utterances]
                    )
                    return transcribed_text.strip()
                else:
                    logger.warning("No utterances found in ASR result")
                    return "[音频转文字完成，但未找到转录内容]"

            elif code == "20000001" or code == "20000002":  # Task still processing
                logger.debug(f"ASR task still processing, attempt {attempt + 1}")
                time.sleep(sleep_time)
                attempt += 1
                sleep_time = min(sleep_time * 1.5, 10)  # Exponential backoff, max 10s

            else:  # Task failed
                logger.error(f"ASR task failed with code: {code}")
                return (
                    f"[音频转文字失败: {query_response.headers.get('X-Api-Message', '未知错误')}]"
                )

        logger.warning("ASR task timed out")
        return "[音频转文字超时]"


# Convenience function for easy import
def create_multimodal_message(
    text: Optional[str] = None,
    image_data: Optional[str] = None,
    image_url: Optional[str] = None,
    audio_url: Optional[str] = None,
    **kwargs,
) -> HumanMessage:
    """
    Convenience function to create a multimodal HumanMessage.

    Args:
        text: Text content for the message
        image_data: Base64 encoded image data
        image_url: URL to an image
        audio_data: Base64 encoded audio data
        audio_url: URL to an audio file
        **kwargs: Additional arguments

    Returns:
        HumanMessage object with multimodal content

    Raises:
        ValueError: For various validation errors
    """
    global _processor

    if _processor is None:
        _processor = MultimodalInputProcessor()

    return _processor.create_human_message(
        text=text,
        image_data=image_data,
        image_url=image_url,
        audio_url=audio_url,
        **kwargs,
    )
