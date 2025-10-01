# 智能体系统

一个基于langgraph的艺术设计领域智能体

## 环境搭建

例子

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

### 启动后端服务

1. **打开一个新的终端**。
2. **确保你在项目的根目录** (`art-agent/`)。
3. **运行以下命令**来启动后端 FastAPI 应用：
   note: 8001端口写在./frontend/vite.config.ts里，如果换的话这里需要换掉

   ```bash
   uvicorn src.service:app --reload --port 8001
   ```

   * `uvicorn` 是一个 ASGI 服务器，用于运行 FastAPI 应用。
   * `src.service:app` 指向我们在 `src/service/__init__.py` 文件中创建的 FastAPI `app` 实例。
   * `--reload` 会在代码更改时自动重启服务器，方便开发。
   * `--port 8001` 指定服务运行在 8001 端口。

   你应该会看到类似以下的输出，表示后端服务已成功启动：

   ```
   INFO:     Uvicorn running on http://127.0.0.1:8001 (Press CTRL+C to quit)
   INFO:     Started reloader process [xxxx] using statreload
   Initializing CanvasAgent...
   CanvasAgent initialized successfully.
   INFO:     Started server process [xxxx]
   INFO:     Waiting for application startup.
   INFO:     Application startup complete.
   ```

### 启动前端服务

1. **再打开一个新的终端**。
2. **进入 `frontend` 目录**：

   ```bash
   cd frontend
   ```
3. **安装依赖**（如果你之前没有安装过）：
   note: 先安装npm [安装教程 windows](https://blog.csdn.net/zhouyan8603/article/details/109039732)

   ```bash
   npm install
   ```
4. **启动前端开发服务器**：

   ```bash
   npm run dev
   ```

   你应该会看到类似以下的输出，并得到一个本地访问地址（通常是 `http://localhost:5173`）：

   ```
     VITE v5.x.x  ready in xxx ms

     ➜  Local:   http://localhost:5173/
     ➜  Network: use --host to expose
     ➜  press h + enter to show help
   ```

### 如何访问和测试

1. 在你的浏览器中打开前端服务的地址（例如 `http://localhost:5173`）。
2. 现在你应该可以看到应用的界面，包括画布和聊天窗口。
3. 在聊天框中输入你的设计需求（例如 "为一家咖啡店设计一个logo"），然后发送。
4. 观察终端和浏览器开发者工具的网络请求，你应该能看到对后端 `http://localhost:8001/api/canvas/chat` 的 POST 请求。
5. 后端 Agent 处理后，聊天窗口会显示 AI 的回复和工具使用情况，并且画布上会展示生成的 SVG 图像。

TODO:
[ ] Context managing
