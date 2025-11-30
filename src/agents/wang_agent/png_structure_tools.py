"""
PNG结构提取工具 - 从PNG图像中提取关键点、轮廓、颜色分布等结构信息
用于指导SVG生成，使SVG更贴近PNG图像
"""

import logging
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from langchain_core.tools import tool
from PIL import Image
import json

logger = logging.getLogger(__name__)

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    logger.warning("OpenCV not available, some structure extraction features will be limited")


@tool
def extract_image_structure(
    image_path: str,
    component_region: Optional[Tuple[int, int, int, int]] = None,
    extract_keypoints: bool = True,
    extract_contours: bool = True,
    extract_colors: bool = True,
) -> Dict[str, Any]:
    """
    从PNG图像中提取结构信息，包括关键点、轮廓、颜色分布等。
    这些信息用于指导SVG生成，使SVG更贴近PNG图像。
    
    Args:
        image_path: PNG图像路径
        component_region: 可选，指定要分析的区域 (x, y, width, height)
        extract_keypoints: 是否提取关键点
        extract_contours: 是否提取轮廓
        extract_colors: 是否提取颜色分布
    
    Returns:
        包含结构信息的字典：
        - keypoints: 关键点列表 [(x, y), ...]
        - contours: 轮廓点列表 [[(x, y), ...], ...]
        - dominant_colors: 主要颜色列表 [(r, g, b), ...]
        - shape_features: 形状特征（边界框、中心点等）
        - color_distribution: 颜色分布统计
    """
    try:
        # Load image
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Convert to numpy array
        img_array = np.array(img)
        height, width = img_array.shape[:2]
        
        # Crop to component region if specified
        if component_region:
            x, y, w, h = component_region
            x = max(0, min(x, width))
            y = max(0, min(y, height))
            w = min(w, width - x)
            h = min(h, height - y)
            img_array = img_array[y:y+h, x:x+w]
            offset_x, offset_y = x, y
        else:
            offset_x, offset_y = 0, 0
        
        result = {
            "image_size": (img_array.shape[1], img_array.shape[0]),
            "offset": (offset_x, offset_y),
        }
        
        if HAS_CV2:
            # Convert to grayscale for contour detection
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Extract contours
            if extract_contours:
                # Apply threshold to get binary image
                _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
                # Find contours
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # Simplify contours (reduce points)
                simplified_contours = []
                for contour in contours:
                    if len(contour) > 3:  # Need at least 3 points
                        # Approximate contour with fewer points
                        epsilon = 0.02 * cv2.arcLength(contour, True)
                        approx = cv2.approxPolyDP(contour, epsilon, True)
                        # Convert to list of tuples
                        points = [(int(p[0][0] + offset_x), int(p[0][1] + offset_y)) for p in approx]
                        if len(points) >= 3:
                            simplified_contours.append(points)
                
                result["contours"] = simplified_contours
                result["contour_count"] = len(simplified_contours)
            
            # Extract keypoints (corner detection)
            if extract_keypoints:
                # Use corner detection
                corners = cv2.goodFeaturesToTrack(
                    gray,
                    maxCorners=50,
                    qualityLevel=0.01,
                    minDistance=10
                )
                if corners is not None:
                    keypoints = [
                        (int(c[0][0] + offset_x), int(c[0][1] + offset_y))
                        for c in corners
                    ]
                    result["keypoints"] = keypoints
                    result["keypoint_count"] = len(keypoints)
                else:
                    result["keypoints"] = []
                    result["keypoint_count"] = 0
        else:
            # Fallback: simple edge detection using PIL
            if extract_contours:
                # Simple approach: find edges by color difference
                edges = []
                # This is a simplified version - for better results, use OpenCV
                result["contours"] = []
                result["contour_count"] = 0
            
            if extract_keypoints:
                result["keypoints"] = []
                result["keypoint_count"] = 0
        
        # Extract color information
        if extract_colors:
            # Flatten image to list of pixels
            pixels = img_array.reshape(-1, 3)
            
            # Get dominant colors using k-means (simplified version)
            # For simplicity, we'll use a histogram-based approach
            from collections import Counter
            
            # Quantize colors to reduce noise
            quantized = (pixels // 32) * 32  # Quantize to 32 levels
            color_counts = Counter([tuple(c) for c in quantized])
            
            # Get top 5 dominant colors
            top_colors = color_counts.most_common(5)
            dominant_colors = [
                {
                    "rgb": list(color),
                    "count": count,
                    "percentage": count / len(pixels) * 100
                }
                for color, count in top_colors
            ]
            
            result["dominant_colors"] = dominant_colors
            
            # Color distribution statistics
            result["color_distribution"] = {
                "mean_r": float(np.mean(pixels[:, 0])),
                "mean_g": float(np.mean(pixels[:, 1])),
                "mean_b": float(np.mean(pixels[:, 2])),
                "std_r": float(np.std(pixels[:, 0])),
                "std_g": float(np.std(pixels[:, 1])),
                "std_b": float(np.std(pixels[:, 2])),
            }
        
        # Extract shape features
        if HAS_CV2 and extract_contours and result.get("contours"):
            # Find bounding box of all contours
            all_points = []
            for contour in result["contours"]:
                all_points.extend(contour)
            
            if all_points:
                xs = [p[0] for p in all_points]
                ys = [p[1] for p in all_points]
                result["shape_features"] = {
                    "bbox": {
                        "x": min(xs),
                        "y": min(ys),
                        "width": max(xs) - min(xs),
                        "height": max(ys) - min(ys),
                    },
                    "center": {
                        "x": (min(xs) + max(xs)) // 2,
                        "y": (min(ys) + max(ys)) // 2,
                    },
                    "area": (max(xs) - min(xs)) * (max(ys) - min(ys)),
                }
        
        return result
        
    except Exception as e:
        logger.error(f"Error extracting image structure: {e}", exc_info=True)
        return {
            "error": str(e),
            "keypoints": [],
            "contours": [],
            "dominant_colors": [],
        }


@tool
def analyze_component_structure(
    image_path: str,
    component_description: str,
    design_prompt: str = "",
) -> Dict[str, Any]:
    """
    分析特定组件的结构信息，结合LLM理解组件描述，提取该组件在图像中的结构特征。
    
    Args:
        image_path: PNG图像路径
        component_description: 组件描述（如"树冠部分"、"字母A"等）
        design_prompt: 设计提示，用于上下文理解
    
    Returns:
        包含组件结构信息的字典，包括：
        - estimated_region: 估计的组件区域 (x, y, width, height)
        - structure_info: 结构信息（关键点、轮廓等）
        - visual_features: 视觉特征描述
    """
    try:
        # Load image to get dimensions
        img = Image.open(image_path)
        width, height = img.size
        
        # For now, we'll analyze the whole image
        # In a more advanced version, we could use object detection or LLM vision to locate the component
        structure_info = extract_image_structure(
            image_path=image_path,
            extract_keypoints=True,
            extract_contours=True,
            extract_colors=True,
        )
        
        # Estimate component region (for now, use full image or center region)
        # In production, this could be improved with object detection
        estimated_region = {
            "x": 0,
            "y": 0,
            "width": width,
            "height": height,
        }
        
        # Extract visual features summary
        visual_features = {
            "has_curves": len(structure_info.get("contours", [])) > 0,
            "keypoint_count": structure_info.get("keypoint_count", 0),
            "contour_count": structure_info.get("contour_count", 0),
            "dominant_colors": [
                {
                    "rgb": c["rgb"],
                    "percentage": round(c["percentage"], 1)
                }
                for c in structure_info.get("dominant_colors", [])[:3]
            ],
        }
        
        return {
            "component_description": component_description,
            "estimated_region": estimated_region,
            "structure_info": structure_info,
            "visual_features": visual_features,
        }
        
    except Exception as e:
        logger.error(f"Error analyzing component structure: {e}", exc_info=True)
        return {
            "error": str(e),
            "component_description": component_description,
        }


@tool
def get_contour_points_for_svg(
    image_path: str,
    region: Optional[Tuple[int, int, int, int]] = None,
    simplify: bool = True,
    max_points: int = 50,
) -> List[List[Tuple[int, int]]]:
    """
    提取轮廓点，格式化为适合SVG路径的点列表。
    返回的点可以直接用于生成SVG路径。
    
    Args:
        image_path: PNG图像路径
        region: 可选，指定区域 (x, y, width, height)
        simplify: 是否简化轮廓（减少点数）
        max_points: 每个轮廓的最大点数
    
    Returns:
        轮廓点列表，每个轮廓是一个点列表 [(x, y), ...]
    """
    try:
        structure = extract_image_structure(
            image_path=image_path,
            component_region=region,
            extract_contours=True,
            extract_keypoints=False,
            extract_colors=False,
        )
        
        contours = structure.get("contours", [])
        
        if simplify and max_points:
            # Further simplify if needed
            simplified = []
            for contour in contours:
                if len(contour) > max_points:
                    # Sample points evenly
                    step = len(contour) // max_points
                    simplified.append(contour[::step])
                else:
                    simplified.append(contour)
            return simplified
        
        return contours
        
    except Exception as e:
        logger.error(f"Error getting contour points: {e}", exc_info=True)
        return []


# Export tools
PNG_STRUCTURE_TOOLS = [
    extract_image_structure,
    analyze_component_structure,
    get_contour_points_for_svg,
]

