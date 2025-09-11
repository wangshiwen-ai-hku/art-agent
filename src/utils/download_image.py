import requests
import os
import random
import string
from typing import Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Content-Type到文件扩展名的映射
content_type_to_ext = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/gif": ".gif",
    "image/webp": ".webp",
    "image/bmp": ".bmp",
    "image/tiff": ".tiff",
    "image/svg+xml": ".svg",
}


def download_image(url: str, basedir: str, timeout: int = 5) -> Optional[str]:
    """
    从URL下载图片文件并保存到指定目录

    Args:
        url: 图片文件的URL地址
        basedir: 转存文件的目标目录
        timeout: 下载超时秒数，默认5秒

    Returns:
        str: 成功时返回文件名，失败时返回None
    """
    time.sleep(random.random() * 0.1)
    try:
        # 1. 从url处下载图片文件，保存在内存中
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        content = response.content

        # 2. 优先从response header提取文件类型Content-Type
        # 如果header中不存在Content-Type则使用magic判断文件类型
        content_type = response.headers.get("Content-Type", "")
        extension = content_type_to_ext.get(content_type, ".jpg")

        # 3. 给这个文件分配一个随机的7位字符串id
        while True:
            image_id = "".join(
                random.choices(string.ascii_letters + string.digits, k=7)
            )
            filename = f"{image_id}{extension}"
            target_path = os.path.join(basedir, filename)

            # 检查是否存在冲突
            if not os.path.exists(target_path):
                break

        # 4. 直接将文件内容写入目标位置
        with open(target_path, "wb") as f:
            f.write(content)

        # 5. 返回文件名
        return filename

    except Exception as e:
        # 如果在任何步骤中出现异常，返回None
        print(e)
        return None


def download_images(urls: List[str], basedir: str, timeout: int = 10) -> List[str]:
    """
    多线程并行下载多个图片文件

    Args:
        urls: 图片文件的URL地址列表
        basedir: 转存文件的目标目录
        timeout: 下载超时秒数，默认10秒

    Returns:
        List[str]: 成功下载的文件名列表
    """
    # 存储成功下载的文件名
    results = {}

    # 使用线程池执行器进行多线程下载
    with ThreadPoolExecutor() as executor:
        # 提交所有下载任务
        future_to_url = {
            executor.submit(download_image, url, basedir, timeout): url for url in urls
        }

        # 处理完成的任务
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            filename = future.result()
            results[url] = filename

    return results


if __name__ == "__main__":
    from pprint import pprint

    result = download_images(
        [
            "https://www.bing.com/th?id=OHR.TigerDay_ZH-CN4359136631_1920x1080.webp",
            "https://www.bing.com/th?id=OHR.MongoliaYurts_ZH-CN4015475887_1920x1080.jpg&w=720",
            "https://www.bing.com/th?id=OHR.BlackfinBarracuda_ZH-CN3850642551_1920x1080.jpg",
        ],
        "./logs",
    )
    pprint(result)
