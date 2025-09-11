# 智能体系统

一个基于智能体的个性化智能辅导系统，支持多角色协作和记忆管理。

## 环境搭建

### 1. 安装 uv

> 比 pip 快百倍的python包管理工具

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. 项目初始化

```bash
# 克隆项目
git clone <your-repo-url>
cd art-agents

# 安装依赖
uv sync

# 复制环境配置
cp env_example .env
# 编辑 .env 文件，填入必要的配置信息

# 运行一个智能体示例
uv run examples/core/router_react_example.py
```

## 代码结构

```
art-agents/
├── src/                    # 核心代码
│   ├── agents/            # 智能体定义
│   │   ├── base/         # 基础智能体
│   │   ├── chat/         # 聊天智能体
│   │   ├── deep_researcher/ # 深度研究智能体
│   │   └── router_react/ # 路由智能体
│   ├── service/          # 服务层
│   │   ├── api/          # API服务
│   │   ├── student/      # 学生服务
│   │   └── teacher/      # 教师服务
│   ├── infra/            # 基础设施
│   │   ├── memory/       # 记忆管理
│   │   └── tools/        # 工具集
│   ├── config/           # 配置管理
│   └── utils/            # 工具函数
├── examples/             # 示例代码
├── docs/                # 文档
├── test/                # 测试代码
└── scripts/             # 脚本文件
```

## 文档

详细文档请参考：

- [开发文档](./docs/development/) - 开发指南和架构说明
- [API文档](./docs/api/) - 接口使用说明（待修改）
- [用户指南](./docs/user_guide/) - 使用教程（待修改）

## 快速开始

查看 `examples/` 目录下的示例代码来快速了解系统功能。

## 运行

```bash
uv run start_api.py
```
