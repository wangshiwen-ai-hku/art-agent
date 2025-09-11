# Agent系统开发者文档

## 概述

本文档详细说明如何在Art-Agents框架中定义和实现Agent或Multi-Agent System。该框架基于LangGraph构建，支持单个智能体和多智能体系统的开发。

## 系统架构

### 核心组件

```
src/agents/
├── base/                    # 基础抽象类
│   ├── graph.py            # BaseAgent 基类
│   ├── mas_graph.py        # MultiAgentBase 多智能体基类
│   ├── config.py           # 线程配置基类
│   ├── schema.py           # 基础状态Schema
│   └── prompt.py           # 基础系统提示词
├── coordinator/            # 协调器Agent实现
├── chat/                   # 聊天Agent基类
├── deep_researcher/        # 深度研究多智能体系统
└── [your_agent]/          # 你的Agent实现
```

## 1. 定义单个Agent

### 1.1 继承BaseAgent

所有单个Agent都需要继承自 `src.agents.base.graph.BaseAgent`：

```python
# src/agents/my_agent/graph.py
from src.agents.base import BaseAgent
from src.config.manager import AgentConfig
from langraph.graph import StateGraph, START

class MyAgent(BaseAgent):
    """我的自定义Agent"""

    name = "my_agent"
    description = "My custom agent description"

    def __init__(self, agent_config: AgentConfig):
        super().__init__(agent_config)

    def _load_system_prompt(self):
        """加载系统提示词 - 必须实现"""
        from .prompt import MY_SYSTEM_PROMPT
        self._system_prompt = MY_SYSTEM_PROMPT

    def build_graph(self) -> StateGraph:
        """构建Agent图 - 必须实现"""
        graph = StateGraph(state_schema=MyState)  # 使用自定义State

        # 添加节点
        graph.add_node("init_context_node", self.init_context_node)
        graph.add_node("my_main_node", self.my_main_node)

        # 添加边
        graph.add_edge(START, "init_context_node")
        graph.add_edge("init_context_node", "my_main_node")

        return graph

    async def my_main_node(self, state: MyState, config = None):
        """自定义节点实现"""
        # 实现你的逻辑
        messages = [SystemMessage(content=self._system_prompt + "\n" + self._context)]
        messages.extend(state["messages"])

        response = await self._llm.ainvoke(messages)
        return {"messages": [response]}
```

### 1.2 简化实现：继承BaseChatAgent

对于对话类Agent，可以继承 `src.agents.chat.BaseChatAgent`来简化实现：

```python
# src/service/student/agents/tutor/graph.py
from src.agents.chat import BaseChatAgent
from src.config.manager import AgentConfig

class StudentTutorAgent(BaseChatAgent):
    """学生导师Agent"""

    name = "student_tutor"
    description = "Student tutor agent that provides personalized learning guidance"

    def __init__(self, agent_config: AgentConfig):
        super().__init__(agent_config)

    def _load_system_prompt(self):
        """只需要重写系统提示词"""
        from .prompt import TUTOR_SYSTEM_PROMPT
        self._system_prompt = TUTOR_SYSTEM_PROMPT
```

## 2. 定义Multi-Agent System

### 2.1 继承MultiAgentBase

多智能体系统需要继承自 `src.agents.base.mas_graph.MultiAgentBase`：

```python
# src/agents/my_mas/graph.py
from src.agents.base import MultiAgentBase
from src.config.manager import MultiAgentConfig
from langraph.graph import StateGraph, START

class MyMultiAgentSystem(MultiAgentBase):
    """我的多智能体系统"""

    name = "my_multi_agent_system"
    description = "My multi-agent system description"

    def __init__(self, multi_agent_config: MultiAgentConfig):
        super().__init__(multi_agent_config)
        self._load_system_prompts()

    def _load_system_prompts(self):
        """加载各个智能体的系统提示词"""
        self._system_prompts = {
            "agent1": AGENT1_SYSTEM_PROMPT,
            "agent2": AGENT2_SYSTEM_PROMPT,
            "coordinator": COORDINATOR_SYSTEM_PROMPT,
        }

    def build_graph(self) -> StateGraph:
        """构建多智能体图 - 必须实现"""
        builder = StateGraph(MyMASState)

        # 添加节点
        builder.add_edge(START, "coordinator")
        builder.add_node("coordinator", self.coordinator_node)
        builder.add_node("agent1", self.agent1_node)
        builder.add_node("agent2", self.agent2_node)

        return builder

    async def coordinator_node(self, state: MyMASState, config = None):
        """协调器节点"""
        # 使用注册的LLM和工具
        llm = self._agent_llm_registry["coordinator"]
        tools = self._agent_tools_registry["coordinator"]

        # 实现协调逻辑
        pass
```

## 3. 业务Router实现

业务Router（总代理）通常基于CoordinatorAgent模式实现：

```python
# src/agents/my_router/graph.py
from src.agents.coordinator import CoordinatorAgent
from src.config.manager import AgentConfig

class MyBusinessRouter(CoordinatorAgent):
    """业务路由器"""

    name = "my_business_router"
    description = "Business router that coordinates different domain agents"

    def __init__(self, agent_config: AgentConfig, domain_agents: list):
        """
        Args:
            agent_config: 路由器配置
            domain_agents: 领域专家Agent列表
        """
        # 确保包含handoff工具
        assert "handoff_to_other_agent" in agent_config.tools

        super().__init__(agent_config, sub_agents=domain_agents)

    def _load_system_prompt(self):
        """自定义路由器提示词"""
        from .prompt import BUSINESS_ROUTER_SYSTEM_PROMPT
        self._system_prompt = BUSINESS_ROUTER_SYSTEM_PROMPT
```

## 4. 必须实现的抽象方法

### 4.1 BaseAgent抽象方法

```python
class BaseAgent(ABC):
    @abstractmethod
    def _load_system_prompt(self):
        """加载系统提示词 - 必须实现"""
        pass

    @abstractmethod
    def build_graph(self) -> StateGraph:
        """构建Agent图 - 必须实现"""
        pass
```

### 4.2 MultiAgentBase抽象方法

```python
class MultiAgentBase(ABC):
    @abstractmethod
    def build_graph(self) -> StateGraph:
        """构建多智能体图 - 必须实现"""
        pass
```

## 5. 配置系统

### 5.1 Agent配置

在 `config.yaml`中定义Agent配置：

```yaml
agents:
  core:
    my_agent:
      model:
        model_provider: openai
        model: gpt-4.1-mini
        temperature: 0.7
        base_url: ${OPENAI_BASE_URL}
        api_key: ${OPENAI_API_KEY}
      tools: [tavily_search, handoff_to_other_agent]
      config:
        # Agent特定配置
        max_steps: 10
        custom_param: "value"

  student:
    my_student_agent:
      tools: [retrieve_relevant_document]
```

### 5.2 ThreadConfiguration

定义线程配置类：

```python
# src/agents/my_agent/config.py
from dataclasses import dataclass
from src.config.manager import BaseThreadConfiguration

@dataclass(kw_only=True)
class ThreadConfiguration(BaseThreadConfiguration):
    """我的Agent线程配置"""

    # 自定义配置字段
    max_iterations: int = 5
    enable_memory: bool = True
    custom_setting: str = "default"
```

### 5.3 使用配置

```python
# 获取Agent配置
agent_config = config.get_agent_config("my_agent", "core")
agent = MyAgent(agent_config)

# 获取多智能体配置
multi_config = config.get_multi_agent_config("my_mas", "core")
mas = MyMultiAgentSystem(multi_config)

# 在节点中使用线程配置
async def my_node(self, state, config):
    configuration = ThreadConfiguration.from_runnable_config(config)
    max_iter = configuration.max_iterations
    # 使用配置...
```

## 6. Prompt系统

### 6.1 定义提示词

```python
# src/agents/my_agent/prompt.py
MY_SYSTEM_PROMPT = """你是一个专业的AI助手。

# 主要职责
- 处理用户请求
- 提供准确信息
- 维持礼貌友好的对话

# 行为准则
- 始终保持专业态度
- 拒绝不当请求
- 基于事实回答问题

当前时间：{CURRENT_TIME}
用户信息：{user_name}
"""
```

### 6.2 模板系统

对于复杂的多智能体系统，可以使用模板系统：

```python
def apply_prompt_template(self, system_prompt: str, state: State, config = None) -> list:
    """应用提示词模板"""
    state_vars = {
        "CURRENT_TIME": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **state,
    }

    if config:
        configuration = ThreadConfiguration.from_runnable_config(config)
        state_vars.update(configuration.to_dict())

    formatted_prompt = system_prompt.format(**state_vars)
    return [{"role": "system", "content": formatted_prompt}]
```

## 7. State Schema系统

### 7.1 基础State

所有Agent都应该基于 `BaseState`：

```python
# src/agents/base/schema.py
class BaseState(TypedDict):
    """基础状态Schema"""
    user_id: str
    user_name: Optional[str]
    messages: Annotated[list, add_messages]
```

### 7.2 自定义State

```python
# src/agents/my_agent/schema.py
from ..base import BaseState

class MyAgentState(BaseState):
    """我的Agent状态"""
    task_description: str
    progress: float
    custom_data: dict
```

### 7.3 复杂State示例

```python
# src/agents/deep_researcher/schema.py
from typing import Optional, List
from pydantic import BaseModel, Field
from enum import Enum

class StepType(Enum):
    RESEARCH = "research"
    GENERATION = "generation"

class Step(BaseModel):
    title: str = Field(..., description="步骤标题")
    description: str = Field(..., description="步骤描述")
    step_type: StepType = Field(..., description="步骤类型")
    execution_res: Optional[str] = Field(None, description="执行结果")

class Plan(BaseModel):
    title: str = Field(..., description="计划标题")
    steps: List[Step] = Field(..., description="执行步骤")

class State(BaseState):
    task_description: str
    current_plan: Optional[Plan]
    observations: List[str]
    locale: str
```

## 8. 工具系统

### 8.1 工具注册

工具在 `src/infra/tools/manager.py`中注册：

```python
# 原生工具
vanilla_tools_registry = {
    "tavily_search": tavily_search,
    "handoff_to_other_agent": handoff_to_other_agent,
    "my_custom_tool": my_custom_tool,  # 添加你的工具
}
```

### 8.2 自定义工具

```python
# src/infra/tools/my_tool.py
from langchain_core.tools import tool

@tool
def my_custom_tool(query: str) -> str:
    """我的自定义工具

    Args:
        query: 查询参数

    Returns:
        处理结果
    """
    # 实现工具逻辑
    return f"处理结果：{query}"
```

### 8.3 MCP工具配置

在 `config.yaml`中配置MCP工具：

```yaml
mcp_servers:
  my_server:
    transport: http
    url: http://localhost:8001/mcp/my_server
    enabled_tools: [my_mcp_tool]
```

## 9. 完整示例

### 9.1 简单Agent示例

```python
# src/agents/simple_qa/graph.py
from src.agents.chat import BaseChatAgent
from src.config.manager import AgentConfig

class SimpleQAAgent(BaseChatAgent):
    name = "simple_qa"
    description = "Simple Q&A agent"

    def __init__(self, agent_config: AgentConfig):
        super().__init__(agent_config)

    def _load_system_prompt(self):
        self._system_prompt = "你是一个问答助手，请简洁准确地回答问题。"

# src/agents/simple_qa/config.py
from dataclasses import dataclass
from src.config.manager import BaseThreadConfiguration

@dataclass(kw_only=True)
class ThreadConfiguration(BaseThreadConfiguration):
    max_history: int = 10
```

### 9.2 多智能体示例

```python
# src/agents/team_work/graph.py
from src.agents.base import MultiAgentBase
from src.config.manager import MultiAgentConfig
from langraph.graph import StateGraph, START
from langraph.prebuilt import create_react_agent

class TeamWorkSystem(MultiAgentBase):
    name = "team_work"
    description = "Team collaboration system"

    def __init__(self, multi_agent_config: MultiAgentConfig):
        super().__init__(multi_agent_config)
        self._system_prompts = {
            "planner": "你是规划专家，负责制定计划。",
            "executor": "你是执行专家，负责执行任务。",
            "reviewer": "你是审查专家，负责质量控制。"
        }

    def build_graph(self):
        builder = StateGraph(TeamState)
        builder.add_edge(START, "planner")
        builder.add_node("planner", self.planner_node)
        builder.add_node("executor", self.executor_node)
        builder.add_node("reviewer", self.reviewer_node)
        return builder

    async def planner_node(self, state, config):
        llm = self._agent_llm_registry["planner"]
        tools = self._agent_tools_registry["planner"]
        agent = create_react_agent(name="planner", model=llm, tools=tools)

        result = await agent.ainvoke({
            "messages": [{"role": "user", "content": state["task"]}]
        })

        return {"messages": result["messages"], "plan": "详细计划..."}
```

## 10. 最佳实践

### 10.1 设计原则

1. **单一职责**：每个Agent专注于特定领域
2. **模块化**：使用清晰的模块结构
3. **可配置**：通过配置文件管理参数
4. **可测试**：编写单元测试验证功能

### 10.2 错误处理

```python
async def my_node(self, state, config):
    try:
        result = await self.process_task(state)
        return {"messages": [result]}
    except Exception as e:
        logger.error(f"Node execution failed: {e}")
        return {"messages": [f"执行失败：{str(e)}"]}
```

### 10.3 日志记录

```python
import logging

logger = logging.getLogger(__name__)

class MyAgent(BaseAgent):
    def __init__(self, agent_config):
        super().__init__(agent_config)
        logger.info(f"{self.name} initialized successfully")

    async def my_node(self, state, config):
        logger.info(f"Processing task: {state.get('task')}")
        # 处理逻辑
        logger.info("Task completed successfully")
```

### 10.4 测试

```python
# test/test_my_agent.py
import pytest
import asyncio
from src.agents.my_agent import MyAgent
from src.config.manager import config

@pytest.fixture
async def agent():
    agent_config = config.get_agent_config("my_agent", "core")
    return MyAgent(agent_config)

@pytest.mark.asyncio
async def test_agent_basic_functionality(agent):
    state = {
        "user_id": "test_user",
        "messages": [{"role": "user", "content": "测试消息"}]
    }

    result = await agent.compiled_graph.ainvoke(state)
    assert "messages" in result
    assert len(result["messages"]) > 0
```

## 11. 部署和运行

### 11.1 初始化MCP工具

```python
# main.py
import asyncio
from src.infra.tools.manager import load_all_mcp_tools

async def main():
    # 加载MCP工具
    await load_all_mcp_tools()

    # 初始化Agent
    agent_config = config.get_agent_config("my_agent", "core")
    agent = MyAgent(agent_config)

    # 运行Agent
    result = await agent.compiled_graph.ainvoke(initial_state)

if __name__ == "__main__":
    asyncio.run(main())
```

### 11.2 API集成

参考 `src/service/api/app.py`中的集成方式，将Agent集成到Web API中。

## 总结

本文档涵盖了在art-Agents框架中定义Agent的完整流程。关键要点：

1. **单个Agent**：继承 `BaseAgent`或 `BaseChatAgent`
2. **多智能体系统**：继承 `MultiAgentBase`
3. **业务Router**：基于 `CoordinatorAgent`模式
4. **必须实现**：`_load_system_prompt()`和 `build_graph()`方法
5. **配置系统**：使用 `config.yaml`和自定义 `ThreadConfiguration`
6. **Prompt系统**：支持模板和变量替换
7. **State系统**：基于 `BaseState`扩展自定义状态
8. **工具系统**：支持原生工具和MCP工具

遵循这些指导原则，你就能够成功地在框架中实现自己的智能体系统。
