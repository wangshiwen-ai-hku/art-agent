import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command
from langgraph.graph.state import CompiledStateGraph
from src.config.manager import config
from .canvas_agent import CanvasAgent, CanvasState
from langchain_core.runnables import RunnableConfig
import tempfile
import base64
from .canvas_agent.utils import svg_to_png
import os
from .canvas_agent.schema import create_initial_state, AgentStage, UserIntent

# Global agent instance
agent_instance: CanvasAgent | None = None
# In a real multi-user app, you would generate/manage this ID per user session
thread_id = "user_session_main"
thread_config = RunnableConfig({"configurable": {"thread_id": thread_id}})

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Correctly initialize the agent and its checkpointer on startup, keeping the
    database connection alive for the entire application lifespan.
    """
    global agent_instance
    
    # The 'async with' block ensures the checkpointer connection is properly
    # managed across the application's entire lifecycle.
    async with AsyncSqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
        print("Initializing CanvasAgent with SQLite checkpointer...")
        try:
            await checkpointer.adelete_thread(thread_id)
        except Exception as e:
            print(f"Failed to delete thread: {e} or thread not exists")
            
        try:
            agent_config = config.get_agent_config('canvas_agent', 'core')
            agent = CanvasAgent(agent_config)
            # The graph is compiled with the active checkpointer from the context.
            agent.compiled_graph: CompiledStateGraph = agent.build_graph().compile(checkpointer=checkpointer)
            agent_instance = agent
            init_state = create_initial_state(project_dir="output/session_" + thread_id)
            await agent.compiled_graph.ainvoke(init_state, config=thread_config)
            print("CanvasAgent initialized and compiled successfully.")
            # Yield control to the running application. The connection remains open.
            yield
            
        except Exception as e:
            print(f"Failed to initialize CanvasAgent: {e}")
            agent_instance = None
            # Still yield to allow the app to start, even if agent fails.
            # Requests will then fail cleanly with a 503 error.
            yield
        finally:
            # This block is not strictly necessary for shutdown logic if yield is inside,
            # but it's good for clarity. The 'async with' handles cleanup.
            print("Application shutdown.")


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有请求头
)

class ChatRequest(BaseModel):
    message: str
    svg: str | None = None
    stage: str = 'chat'

@app.post('/api/canvas/chat')
async def canvas_chat(req: ChatRequest):
    """
    Endpoint to interact with the persistent, SQLite-backed CanvasAgent.
    """
    if not agent_instance or not agent_instance.compiled_graph:
        raise HTTPException(status_code=503, detail="Agent is not initialized or is misconfigured.")

    current_state = await agent_instance.compiled_graph.aget_state(thread_config)

    # We need to map the flat request to the nested state
    # 获取当前状态，如果不存在则使用默认值
    current_values = current_state.values if current_state else {}
    current_content = current_values.get("content", {})
    current_conversation = current_values.get("conversation", {})
    current_project = current_values.get("project", {})
    current_workflow = current_values.get("workflow", {})
    
    update_payload = {
        "user_input": req.message,
        "conversation": {
            "messages": current_conversation.get("messages", []),
            "current_topic": current_conversation.get("current_topic")
        },
        "workflow": {
            "current_stage": AgentStage(req.stage),
            "current_intent": current_workflow.get("current_intent", UserIntent.CHAT),
            "is_completed": current_workflow.get("is_completed", False)
        },
        "content": {
            "current_svg": current_content.get("current_svg"),
            "svg_history": current_content.get("svg_history", []),
            "reference_images": current_content.get("reference_images", [])
        },
        "project": {
            "project_dir": current_project.get("project_dir"),
            "saved_files": current_project.get("saved_files", [])
        }
    }
    
    # If an SVG is provided in the request, update the history.
    if req.svg:
        from .canvas_agent.schema import SvgArtwork
        # This assumes the full SVG code is sent.
        # In a real app, you might just send an ID and retrieve the full SVG on the backend.
        if current_state:
            # Append to existing history
            history = current_state.values.get("content", {}).get("svg_history", [])
            history.append(SvgArtwork(svg_code=req.svg, elements=[]))
            update_payload["content"]["svg_history"] = history

    try:
        # ainvoke will use the checkpointer that was configured and kept alive at startup
        msg = await agent_instance.compiled_graph.ainvoke(Command(resume=update_payload), config=thread_config)
        
        all_messages = msg.get("conversation", {}).get("messages", [])
        last_ai_message = next((m.content for m in reversed(all_messages) if m.type == 'ai'), "Task received. Waiting for next input.")
        
        current_svg_artwork = msg.get("content", {}).get("current_svg")
        # SvgArtwork 是 Pydantic BaseModel，使用属性访问而不是字典方法
        latest_svg = current_svg_artwork.svg_code if current_svg_artwork else None
        tool_outputs = [m.content for m in all_messages if m.type == 'tool']

        # 调试信息
        print(f"📊 API Response Debug:")
        print(f"  - Current SVG artwork: {current_svg_artwork}")
        print(f"  - Latest SVG length: {len(latest_svg) if latest_svg else 0}")
        print(f"  - Reply length: {len(last_ai_message) if last_ai_message else 0}")

        return {
            "reply": last_ai_message,
            "svg": latest_svg,
            "tool_outputs": tool_outputs
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")


@app.post('/api/canvas/convert_svg_to_png')
async def convert_svg_to_png_endpoint(payload: dict):
    """Receive an SVG string, write to a temp file, convert to PNG using svg_to_png, return base64 PNG."""
    svg = payload.get('svg')
    if not svg:
        raise HTTPException(status_code=400, detail='Missing svg in payload')
    tmp_svg = None
    try:
        with tempfile.NamedTemporaryFile(suffix='.svg', delete=False, mode='w', encoding='utf-8') as f:
            f.write(svg)
            tmp_svg = f.name

        png_path = svg_to_png(tmp_svg)
        if not png_path:
            raise HTTPException(status_code=500, detail='SVG to PNG conversion failed on server')

        with open(png_path, 'rb') as pf:
            png_bytes = pf.read()
        b64 = base64.b64encode(png_bytes).decode('ascii')
        data_url = f'data:image/png;base64,{b64}'
        return { 'png_base64': data_url }
    finally:
        try:
            if tmp_svg and os.path.exists(tmp_svg):
                os.remove(tmp_svg)
            if png_path and os.path.exists(png_path):
                os.remove(png_path)
        except Exception:
            pass
