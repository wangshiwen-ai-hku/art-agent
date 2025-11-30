"""
SVG质量评估工具 - 从DiffVG启发而来的定量指标和差异分析
"""

import json
import base64
import logging
import os
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

try:
    from PIL import Image
    import cv2
    HAS_IMAGE_LIBS = True
except ImportError:
    HAS_IMAGE_LIBS = False
    logger.warning("PIL/cv2 not available, image metrics will be limited")

try:
    from skimage.metrics import structural_similarity as ssim
    HAS_SSIM = True
except ImportError:
    HAS_SSIM = False
    logger.warning("scikit-image not available, SSIM will be unavailable")

try:
    import lpips
    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False
    logger.warning("lpips not available, perceptual similarity will be unavailable")


def calculate_image_metrics(svg_image_path: str, target_image_path: str) -> Dict[str, float]:
    """
    计算SVG渲染图像与目标图像之间的定量指标。
    
    Args:
        svg_image_path: SVG渲染后的图像路径
        target_image_path: 目标图像路径
    
    Returns:
        包含各种指标的字典：
        - ssim: 结构相似度 (0-1, 越高越好)
        - mse: 均方误差 (越低越好)
        - psnr: 峰值信噪比 (越高越好)
        - lpips: 感知相似度 (越低越好，如果可用)
    """
    if not HAS_IMAGE_LIBS:
        return {"error": "Image processing libraries not available"}
    
    try:
        # 加载图像
        svg_img = cv2.imread(svg_image_path)
        target_img = cv2.imread(target_image_path)
        
        if svg_img is None or target_img is None:
            return {"error": "Failed to load images"}
        
        # 确保尺寸一致
        if svg_img.shape != target_img.shape:
            target_img = cv2.resize(target_img, (svg_img.shape[1], svg_img.shape[0]))
        
        # 转换为RGB（cv2默认是BGR）
        svg_img_rgb = cv2.cvtColor(svg_img, cv2.COLOR_BGR2RGB)
        target_img_rgb = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
        
        metrics = {}
        
        # 1. MSE (均方误差)
        mse = np.mean((svg_img_rgb.astype(float) - target_img_rgb.astype(float)) ** 2)
        metrics["mse"] = float(mse)
        
        # 2. PSNR (峰值信噪比)
        if mse > 0:
            max_pixel = 255.0
            psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
            metrics["psnr"] = float(psnr)
        else:
            metrics["psnr"] = float('inf')
        
        # 3. SSIM (结构相似度)
        if HAS_SSIM:
            # 计算每个通道的SSIM，然后取平均
            ssim_scores = []
            for i in range(3):  # RGB三个通道
                ssim_score = ssim(
                    svg_img_rgb[:, :, i],
                    target_img_rgb[:, :, i],
                    data_range=255
                )
                ssim_scores.append(ssim_score)
            metrics["ssim"] = float(np.mean(ssim_scores))
        else:
            metrics["ssim"] = None
        
        # 4. LPIPS (感知相似度) - 需要额外的库
        if HAS_LPIPS:
            try:
                # 初始化LPIPS模型（使用AlexNet作为backbone）
                loss_fn = lpips.LPIPS(net='alex')
                
                # 转换为tensor并归一化到[-1, 1]
                svg_tensor = lpips.im2tensor(svg_img_rgb)
                target_tensor = lpips.im2tensor(target_img_rgb)
                
                # 计算LPIPS
                lpips_score = loss_fn(svg_tensor, target_tensor)
                metrics["lpips"] = float(lpips_score.item())
            except Exception as e:
                logger.warning(f"LPIPS calculation failed: {e}")
                metrics["lpips"] = None
        else:
            metrics["lpips"] = None
        
        return metrics
    
    except Exception as e:
        logger.error(f"Error calculating image metrics: {e}", exc_info=True)
        return {"error": str(e)}


def analyze_svg_differences(
    svg_image_path: str,
    target_image_path: str,
    grid_size: int = 4
) -> Dict[str, any]:
    """
    分析SVG渲染图像与目标图像的差异区域。
    将图像分成网格，计算每个区域的差异。
    
    Args:
        svg_image_path: SVG渲染后的图像路径
        target_image_path: 目标图像路径
        grid_size: 网格大小（将图像分成grid_size x grid_size个区域）
    
    Returns:
        包含差异分析的字典：
        - regions: 差异区域列表，每个区域包含：
          - region: (x, y, width, height) 区域坐标
          - difference: 差异分数 (0-1, 越高差异越大)
          - mse: 该区域的MSE
    """
    if not HAS_IMAGE_LIBS:
        return {"error": "Image processing libraries not available"}
    
    try:
        # 加载图像
        svg_img = cv2.imread(svg_image_path)
        target_img = cv2.imread(target_image_path)
        
        if svg_img is None or target_img is None:
            return {"error": "Failed to load images"}
        
        # 确保尺寸一致
        if svg_img.shape != target_img.shape:
            target_img = cv2.resize(target_img, (svg_img.shape[1], svg_img.shape[0]))
        
        h, w = svg_img.shape[:2]
        cell_h = h // grid_size
        cell_w = w // grid_size
        
        regions = []
        
        for i in range(grid_size):
            for j in range(grid_size):
                # 计算区域坐标
                x = j * cell_w
                y = i * cell_h
                width = cell_w
                height = cell_h
                
                # 提取区域
                svg_region = svg_img[y:y+height, x:x+width]
                target_region = target_img[y:y+height, x:x+width]
                
                # 计算该区域的MSE
                mse = np.mean((svg_region.astype(float) - target_region.astype(float)) ** 2)
                
                # 归一化差异分数 (0-1)
                # 假设MSE最大值为255^2，归一化
                max_mse = 255.0 ** 2
                difference = min(mse / max_mse, 1.0)
                
                regions.append({
                    "region": (int(x), int(y), int(width), int(height)),
                    "difference": float(difference),
                    "mse": float(mse)
                })
        
        # 按差异排序，找出差异最大的区域
        regions.sort(key=lambda x: x["difference"], reverse=True)
        
        return {
            "regions": regions,
            "grid_size": grid_size,
            "top_differences": regions[:3]  # 返回差异最大的3个区域
        }
    
    except Exception as e:
        logger.error(f"Error analyzing SVG differences: {e}", exc_info=True)
        return {"error": str(e)}


def calculate_metrics_from_base64(
    svg_base64: str,
    target_image_path: str
) -> Dict[str, float]:
    """
    从base64编码的SVG图像计算指标。
    用于在reflect_node中直接使用。
    """
    import tempfile
    
    try:
        # 解码base64图像
        if svg_base64.startswith("data:image"):
            # 移除data URL前缀
            base64_data = svg_base64.split(",")[1]
        else:
            base64_data = svg_base64
        
        image_data = base64.b64decode(base64_data)
        
        # 保存到临时文件
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            tmp_file.write(image_data)
            tmp_svg_path = tmp_file.name
        
        # 计算指标
        metrics = calculate_image_metrics(tmp_svg_path, target_image_path)
        
        # 清理临时文件
        try:
            os.unlink(tmp_svg_path)
        except:
            pass
        
        return metrics
    
    except Exception as e:
        logger.error(f"Error calculating metrics from base64: {e}", exc_info=True)
        return {"error": str(e)}


def format_metrics_for_llm(metrics: Dict[str, float]) -> str:
    """
    将指标格式化为LLM可理解的文本。
    """
    if "error" in metrics:
        return f"指标计算失败: {metrics['error']}"
    
    lines = ["定量指标评估结果："]
    
    if metrics.get("ssim") is not None:
        ssim_val = metrics["ssim"]
        ssim_status = "✓" if ssim_val > 0.7 else "✗"
        lines.append(f"{ssim_status} 结构相似度(SSIM): {ssim_val:.3f} (目标>0.7, 越高越好)")
    
    if metrics.get("mse") is not None:
        mse_val = metrics["mse"]
        mse_status = "✓" if mse_val < 5000 else "✗"
        lines.append(f"{mse_status} 均方误差(MSE): {mse_val:.1f} (目标<5000, 越低越好)")
    
    if metrics.get("psnr") is not None:
        psnr_val = metrics["psnr"]
        psnr_status = "✓" if psnr_val > 20 else "✗"
        lines.append(f"{psnr_status} 峰值信噪比(PSNR): {psnr_val:.2f} dB (目标>20, 越高越好)")
    
    if metrics.get("lpips") is not None:
        lpips_val = metrics["lpips"]
        lpips_status = "✓" if lpips_val < 0.3 else "✗"
        lines.append(f"{lpips_status} 感知相似度(LPIPS): {lpips_val:.3f} (目标<0.3, 越低越好)")
    
    # 综合评估
    passed_count = sum([
        metrics.get("ssim", 0) > 0.7 if metrics.get("ssim") is not None else False,
        metrics.get("mse", float('inf')) < 5000 if metrics.get("mse") is not None else False,
        metrics.get("psnr", 0) > 20 if metrics.get("psnr") is not None else False,
        metrics.get("lpips", float('inf')) < 0.3 if metrics.get("lpips") is not None else False,
    ])
    
    total_metrics = sum([
        metrics.get("ssim") is not None,
        metrics.get("mse") is not None,
        metrics.get("psnr") is not None,
        metrics.get("lpips") is not None,
    ])
    
    if total_metrics > 0:
        pass_rate = passed_count / total_metrics
        lines.append(f"\n综合评估: {passed_count}/{total_metrics} 项指标达标 (通过率: {pass_rate:.1%})")
    
    return "\n".join(lines)


def format_differences_for_llm(diff_analysis: Dict[str, any]) -> str:
    """
    将差异分析格式化为LLM可理解的文本。
    """
    if "error" in diff_analysis:
        return f"差异分析失败: {diff_analysis['error']}"
    
    lines = ["差异区域分析："]
    
    top_diffs = diff_analysis.get("top_differences", [])
    if not top_diffs:
        return "未发现显著差异区域"
    
    for i, region_info in enumerate(top_diffs, 1):
        x, y, w, h = region_info["region"]
        diff_score = region_info["difference"]
        mse = region_info["mse"]
        
        # 计算区域中心
        center_x = x + w // 2
        center_y = y + h // 2
        
        lines.append(
            f"{i}. 区域 ({x},{y}) 到 ({x+w},{y+h}), "
            f"中心点约({center_x},{center_y}): "
            f"差异分数={diff_score:.3f}, MSE={mse:.1f}"
        )
    
    lines.append("\n建议：优先调整差异最大的区域，可以重新生成该区域的SVG元素。")
    
    return "\n".join(lines)

