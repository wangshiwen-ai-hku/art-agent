# 智能体系统

一个基于langgraph的艺术设计领域智能体

## 环境搭建

```bash
conda create -n agent python=3.12 # >=3.11
pip install -r requirements.txt
```

### 2. 项目初始化

```bash
# 克隆项目
git clone <your-repo-url>
cd art-agents

cp env_example .env
# 编辑 .env 文件，填入必要的配置信息
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
│   ├── infra/            # 基础设施
│   │   ├── memory/       # 记忆管理
│   │   └── tools/        # 工具集
│   ├── config/           # 配置管理
│   └── utils/            # 工具函数
├── examples/             # 示例代码
├── docs/                # 文档
└── scripts/             # 脚本文件
```

## 文档

详细文档请参考：

- [开发文档](./docs/development/) - 开发指南和架构说明
- [API文档](./docs/api/) - 接口使用说明（待修改）
- [用户指南](./docs/user_guide/) - 使用教程（待修改）

## 快速开始

查看 `examples/` 目录下的示例代码来快速了解系统功能
