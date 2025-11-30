# Wang Agent

Wang Agent 是一个迭代式SVG生成agent，通过权重分析和反思机制，分步骤生成高质量的SVG图像。

## 设计思路

根据图片中的设计流程：

1. **THINK节点**: 将用户的复杂意图转换为详细的设计提示
2. **图像生成**: 根据设计提示生成初始图像（或使用用户提供的图像）
3. **权重分析**: 将图像分解为按权重排序的组件
4. **分步生成**: 按权重从高到低，逐步生成每个组件的SVG
5. **反思评估**: 每生成一步，评估质量是否符合标准
6. **迭代改进**: 如果质量不达标，回退并重新生成
7. **动态停止**: 当所有组件都满足标准时停止

## 工作流程

```
用户意图 → THINK → 设计提示 + 图像提示
    ↓
生成初始图像 (或使用用户图像)
    ↓
权重分析 → [组件1(权重最高), 组件2, 组件3, ...]
    ↓
生成组件1的SVG → 反思 → [通过/不通过]
    ↓ (通过)
生成组件2的SVG → 反思 → [通过/不通过]
    ↓ (不通过，回退)
重新生成组件2的SVG → 反思 → ...
    ↓
... 直到所有组件完成
```

## 文件结构

```
wang_agent/
├── __init__.py          # 模块导出
├── schema.py            # 状态和数据结构定义
├── graph.py             # 核心agent实现
├── test_example.py      # 使用示例
├── README.md            # 本文档
└── prompt/              # Prompt模板
    ├── think_prompt.txt
    ├── weight_analysis_prompt.txt
    ├── svg_generation_prompt.txt
    ├── reflect_prompt.txt
    └── system_prompt.txt
```

## 使用方法

### 基本使用

```python
from dotenv import load_dotenv
load_dotenv()

from src.config.manager import config
from src.agents.wang_agent import WangAgent, WangState

# 获取agent配置
agent_config = config.get_agent_config("wang_agent", "core")

# 初始化agent
agent = WangAgent(agent_config)

# 创建初始状态
initial_state = WangState(
    user_intention="生成一棵树的SVG图像",
    project_dir="output/wang_agent_test",
    max_iterations=10,  # 最大迭代次数
)

# 运行agent
compiled_graph = agent.compile_graph()
final_state = await compiled_graph.ainvoke(initial_state)

# 获取结果
print(f"完成状态: {final_state['is_complete']}")
print(f"SVG代码: {final_state['current_svg']}")
```

### 使用用户提供的图像

```python
initial_state = WangState(
    user_intention="将这个logo转换为SVG",
    user_image_path="path/to/user/image.png",
    project_dir="output/wang_agent_test",
)
```

## 状态结构

`WangState` 包含以下主要字段：

- **输入**:
  - `user_intention`: 用户的复杂意图
  - `user_image_path`: 可选的用户提供的图像路径

- **THINK输出**:
  - `design_prompt`: 详细的设计提示
  - `image_prompt`: 图像生成提示
  - `instruction`: 包含intention和criteria的指令

- **生成资源**:
  - `initial_image_path`: 初始图像路径
  - `weighted_components`: 按权重排序的组件列表
  - `current_svg`: 当前SVG代码
  - `stages`: 所有阶段的生成历史

- **控制**:
  - `current_stage`: 当前阶段编号
  - `iteration_count`: 当前迭代次数
  - `max_iterations`: 最大迭代次数
  - `is_complete`: 是否完成

## 配置

在 `config.yaml` 中配置：

```yaml
agents:
  core:
    wang_agent:
      model:
        model_provider: google_vertexai
        model: gemini-2.5-flash
        api_key: ${GOOGLE_API_KEY}
        temperature: 0.7
      tools:
        - edit_agent_with_tool
        - draw_agent_with_tool
        - generate_image_tool
```

## 节点说明

1. **think_node**: 将用户意图转换为设计提示和指令
2. **generate_image_node**: 生成或使用初始图像
3. **analyze_weights_node**: 分析图像，分解为权重组件
4. **generate_svg_stage_node**: 生成当前阶段的SVG组件
5. **reflect_node**: 评估当前SVG质量
6. **router_condition**: 根据反思结果决定下一步（继续/回退/完成）

## 注意事项

1. 确保已配置环境变量（`GOOGLE_API_KEY`等）
2. 图像生成需要Google GenAI的imagen模型
3. SVG生成使用canvas_agent的drawer工具
4. 反思机制使用LLM评估，可能需要调整prompt以获得更好的评估结果

## 未来改进

- [ ] 更智能的回退机制（回退到上一个成功的阶段）
- [ ] 更细粒度的权重分析
- [ ] 支持多轮用户交互
- [ ] 更完善的错误处理
- [ ] 性能优化（缓存、并行处理等）

