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
from src.agents.canvas_agent import CanvasAgent, CanvasState
from langchain_core.runnables import RunnableConfig

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
            init_state = {
                "messages": [],
                "stage": "chat",
                "user_message": ""
            }
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

    graph_input = {
        "user_message": req.message,
        "stage": req.stage,
        "svg_history": [req.svg] if req.svg else [],
    }

    try:
        # ainvoke will use the checkpointer that was configured and kept alive at startup
        msg = await agent_instance.compiled_graph.ainvoke(Command(resume=graph_input), config=thread_config)
        
        all_messages = msg.get("messages", [])
        last_ai_message = next((m.content for m in reversed(all_messages) if m.type == 'ai'), "Task received. Waiting for next input.")
        
        latest_svg = msg.get("svg_history", [])[-1] if msg.get("svg_history") else None
        tool_outputs = [m.content for m in all_messages if m.type == 'tool']

        return {
            "reply": last_ai_message,
            "svg": latest_svg,
            "tool_outputs": tool_outputs
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")
