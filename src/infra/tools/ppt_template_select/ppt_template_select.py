from pathlib import Path
from typing import Dict, Optional
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


class PPTTemplateSelectInput(BaseModel):
    """PPT模板选择工具输入参数。"""
    template_style: Optional[str] = Field(
        default=None,
        description="模板风格，None时返回选项列表，可选值：base/modern/minimal/classic"
    )


def _ppt_template_select_func(template_style: Optional[str] = None) -> Dict:
    """PPT模板选择工具。
    
    Args:
        template_style: 模板风格，None时返回选项列表
        
    Returns:
        dict: 选项列表或整套模板内容
    """
    templates_dir = Path(__file__).parent / "templates"
    
    # 如果没有指定风格，返回选项列表给前端
    if not template_style:
        # 中文描述映射到文件夹
        style_mapping = {
            "经典风格": "base",
            "现代风格": "modern", 
            "简约风格": "minimal",
            "传统风格": "classic"
        }
        
        available_styles = []
        if templates_dir.exists():
            for folder in templates_dir.iterdir():
                if folder.is_dir() and folder.name not in ["__pycache__", ".DS_Store"]:
                    # 找到对应的中文描述
                    chinese_name = None
                    for cn, en in style_mapping.items():
                        if en == folder.name:
                            chinese_name = cn
                            break
                    if not chinese_name:
                        chinese_name = folder.name  # 回退到英文名
                    available_styles.append({"name": chinese_name, "value": folder.name})
        
        return {
            "type": "template_options",
            "options": available_styles or [{"name": "经典风格", "value": "base"}]
        }
    
    # 如果指定了风格，返回整套模板内容
    template_files = {
        "title": "first_temp.html",
        "toc": "content_temp.html", 
        "chapter": "content_temp.html",
        "content": "content_temp.html",
        "viewer": "viewer_template.html"
    }
    
    templates = {}
    style_dir = templates_dir / template_style
    
    # 如果指定风格不存在，回退到base
    if not style_dir.exists():
        style_dir = templates_dir / "base"
        template_style = "base"
    
    # 读取所有模板文件
    for slide_type, filename in template_files.items():
        template_path = style_dir / filename
        if template_path.exists():
            templates[slide_type] = template_path.read_text(encoding='utf-8')
        else:
            templates[slide_type] = f"<!-- Template {slide_type} not found -->"
    
    return {
        "type": "template_set",
        "template_style": template_style,
        "templates": templates
    }


# 创建StructuredTool实例
ppt_template_select = StructuredTool.from_function(
    func=_ppt_template_select_func,
    name="ppt_template_select",
    description="PPT模板选择工具：获取可选风格或返回整套模板",
    args_schema=PPTTemplateSelectInput
)
