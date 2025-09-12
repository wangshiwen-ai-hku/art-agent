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
        image_data: Optional[List[str]] = None,
        image_url: Optional[List[str]] = None,
        audio_url: Optional[List[str]] = None,
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
            for image in image_data:
                content_blocks.append(self._process_image_base64(image))
        if image_url:
            for url in image_url:
                content_blocks.append(self._process_image_url(url))

        # If only text provided, return simple text message
        if len(content_blocks) == 1 and content_blocks[0]["type"] == "text":
            return HumanMessage(content=text)

        # Return multimodal message
        return HumanMessage(content=content_blocks)
    
    def _process_image_base64(self, image_data: str) -> Dict[str, Any]:
        """返回 LangChain 标准 image_url block"""
        decoded = base64.b64decode(image_data)
        mime_type = self._detect_image_mime_type(decoded)
        if mime_type not in self.SUPPORTED_IMAGE_TYPES:
            raise ValueError(f"Unsupported image format: {mime_type}")

        # 拼成 data URL
        data_url = f"data:{mime_type};base64,{image_data}"
        return {"type": "image_url", "image_url": {"url": data_url}}


    def _process_image_url(self, image_url: str) -> Dict[str, Any]:
        """返回标准 image_url block"""
        parsed = urlparse(image_url)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError("Invalid image URL format")

        # 可选：校验可访问性
        resp = requests.head(image_url, timeout=10)
        resp.raise_for_status()
        mime_type = resp.headers.get("content-type", "").lower()
        if mime_type not in self.SUPPORTED_IMAGE_TYPES:
            raise ValueError(f"Unsupported image format: {mime_type}")

        # 直接透传
        return {"type": "image_url", "image_url": {"url": image_url}}

    # def _process_image_base64(self, image_data: str) -> Dict[str, Any]:
    #     """Process base64 image data"""
    #     # Try to decode to verify it's valid
    #     decoded = base64.b64decode(image_data)

    #     # Detect MIME type from data
    #     mime_type = self._detect_image_mime_type(decoded)

    #     if mime_type not in self.SUPPORTED_IMAGE_TYPES:
    #         raise ValueError(f"Unsupported image format: {mime_type}")

    #     return {
    #         "type": "image",
    #         "source_type": "base64",
    #         "data": image_data,
    #         "mime_type": mime_type,
    #     }

    # def _process_image_url(self, image_url: str) -> Dict[str, Any]:
    #     """Process image URL"""
    #     # Validate URL format
    #     parsed_url = urlparse(image_url)
    #     if not parsed_url.scheme or not parsed_url.netloc:
    #         raise ValueError("Invalid image URL format")

    #     # Try to fetch the image to validate it exists and get MIME type
    #     response = requests.head(image_url, timeout=10)
    #     response.raise_for_status()

    #     content_type = response.headers.get("content-type", "").lower()
    #     if content_type not in self.SUPPORTED_IMAGE_TYPES:
    #         raise ValueError(f"Unsupported image format: {content_type}")

    #     return {
    #         "type": "image",
    #         "source_type": "url",
    #         "url": image_url,
    #         "mime_type": content_type,
    #     }

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
    
# 在原来文件末尾追加即可
from typing import List, Union
from google.genai.types import Part   # 需要 google-genai>=1.0

from typing import List, Union, Optional
from google.genai.types import Part
import base64

def create_vertex_multimodal_message(
    text: str,
    image_data: Optional[List[str]] = None,
) -> List[Union[str, Part]]:
    """
    专为 Vertex AI (google-genai SDK) 准备的多模态输入函数。
    用法：
        text = "开头{image}中间{image}结尾"
        image_data = [b64_1, b64_2]
        contents = create_vertex_multimodal_message(text, image_data)
        client.models.generate_content(model="gemini-2.0-flash", contents=contents)
    """
    global _processor
    if _processor is None:
        _processor = MultimodalInputProcessor()

    if image_data is None:
        image_data = []

    # 按 {image} 切分，保留空串
    text_parts = text.split("image")
    img_cnt = len(image_data)
    placeholder_cnt = len(text_parts) - 1

    # if placeholder_cnt >= img_cnt:
    #     raise ValueError(
    #         f"text 中 {{image}} 占位符数量 ({placeholder_cnt}) "
    #         f"与 image_data 长度 ({img_cnt}) 不一致"
    #     )

    contents: List[Union[str, Part]] = []

    for idx, txt in enumerate(text_parts):
        # 1. 先放文字（可能为空）
        if txt:                       # 跳过空串，避免多余 text 节点
            contents.append(txt)
        try:
            img = image_data.pop(0)
        except IndexError:
            img = None
            
        if img:
            mime = _processor._detect_image_mime_type(img)
            if mime not in _processor.SUPPORTED_IMAGE_TYPES:
                raise ValueError(f"Unsupported image format: {mime}")
            contents.append(Part.from_bytes(data=img, mime_type=mime))
        if idx == placeholder_cnt and len(image_data) > 0:
            # 多余的image直接都放在最后
            for img in image_data:
                # raw = base64.b64decode(img)
                mime = _processor._detect_image_mime_type(img)
                if mime not in _processor.SUPPORTED_IMAGE_TYPES:
                    raise ValueError(f"Unsupported image format: {mime}")
                contents.append(Part.from_bytes(data=img, mime_type=mime))

    if not contents:
        raise ValueError("No text or media provided")

    return contents

from typing import Optional
from io import BytesIO
from PIL import Image
from google.genai.types import Image as GImage, RawReferenceImage

def bytes_to_raw_reference_image(
    image_bytes: bytes,
    reference_id: int = 0,
    format: str = "JPEG",          # 或 "PNG"
    quality: int = 95,
) -> RawReferenceImage:
    """
    把原始图片字节流包装成 RawReferenceImage，供 edit_image 使用。
    
    参数
    ----
    image_bytes : bytes
        原始图片字节（jpg/png/...）
    reference_id : int, optional
        引用编号，默认 0
    format : str, optional
        重新编码的格式，默认 "JPEG"
    quality : int, optional
        JPEG 质量，默认 95（仅对 JPEG 生效）
    
    返回
    ----
    RawReferenceImage
        可直接塞进 reference_images=[...]
    """
    # 1. 解码→编码，确保字节流干净且带正确文件头
    with Image.open(BytesIO(image_bytes)) as pil_img:
        out = BytesIO()
        pil_img.save(out, format=format, quality=quality)
        clean_bytes = out.getvalue()

    # 2. 构造 SDK 所需对象
    g_img = GImage(image_bytes=clean_bytes)
    return RawReferenceImage(reference_image=g_img, reference_id=reference_id)