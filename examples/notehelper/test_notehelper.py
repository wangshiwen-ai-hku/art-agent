"""
Chat Agent with MCP Tools Example

This example demonstrates how to use the BaseChatAgent with MCP tools properly loaded.
The agent will have access to both vanilla tools and MCP tools configured in config.yaml.
"""

import sys

# from langgraph.graph.state import LangGraphDeprecatedSinceV05

sys.path.append(".")
import asyncio
import logging
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.types import Command
import os
from dotenv import load_dotenv

load_dotenv()

from src.agents.notehelper import BaseNoteHelperAgent
from src.config.manager import config, AgentConfig, ModelConfig
from src.utils.input_processor import create_multimodal_message
from src.infra.tools.manager import load_all_mcp_tools
from langchain_core.runnables import RunnableConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import time
    
import asyncio, os, logging
from langchain_core.messages import BaseMessage
from src.agents.notehelper import BaseNoteHelperAgent
from src.config.manager import config

# from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.memory import MemorySaver
# from langgraph.checkpoint.memory.aio import AsyncMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver


logging.basicConfig(level=logging.INFO)
from contextlib import AsyncExitStack

def main():
    agent = BaseNoteHelperAgent(config.get_agent_config("notehelper", "classroom"))
    async def run_once(reset=False, thread_id="room-002"):
        async with AsyncExitStack() as stack:
            checkpointer = await stack.enter_async_context(
                AsyncSqliteSaver.from_conn_string("checkpoints.db")
            )
            if reset:
                await checkpointer.adelete_thread(thread_id)
            
            # checkpointer = await stack.enter_async_context(
            #     AsyncSqliteSaver.from_conn_string("checkpoints.db")
            # )
            graph = agent.build_graph().compile(checkpointer=checkpointer)
                # 用户要求
            user_format = {
                    "language": "中文",
                    "format": "markdown",
                    "requirements": "我需要一个十分详细的具有分级标题的课堂笔记，包括chapter 1，2，3等分级标题。"
                }

            state = {
                    "user_specified_format": user_format,
                    "final_notes": "",
                    "history_summaries": [],
                    "currimulate_asr_result": "",
                    "current_asr_chunk": "",
                }

            thread = RunnableConfig(configurable={"thread_id": thread_id})
            logger.info(f"state: {state}")
            msg = await graph.ainvoke(
                    state,
                    config=thread
                )
                # state = await graph.get_state(thread)
            logger.info(f"initial state: {state}")
            time.sleep(1)
            
            # 逐段喂 ASR
            for i, chunk in enumerate(asr_chunks):
                msg = await graph.ainvoke(Command(resume={"current_asr_chunk": chunk}), config=thread)

                    # break
                await asyncio.sleep(10)
                if i == 0: 
                    msg = await graph.ainvoke(
                       Command(resume={"current_message": "请为我再介绍一下这本书的作者"}),
                        config=thread
                    )
                if i == 2:
                    msg = await graph.ainvoke(
                        Command(resume={"asr_end": True}),
                        config=thread
                    )
                   
            print("======== 最终笔记 ========")
            print(msg["final_notes"])
    
    asyncio.run(run_once(reset=True, thread_id="room-002"))
    
if __name__ == "__main__":
    print("🚀 启动聊天代理...")
    print("📋 注意: 确保MCP服务器正在运行 (如果配置了的话)")
    print("   例如: uv run scripts/run_mcp_servers.py calculator")
    print("=" * 50)
    asr_chunks = [
        
        "你好，欢迎每天听本书，我是池晓，是钥匙玩笑夏令营和好奇学习社区的创办人。今天我想向你推荐一本书是今天我想向你推荐一本书，是活出生命的意义，上一讲里我们和生死课老师陆小燕探讨了死亡的意义。这一讲里我为你请到了",
        "这一讲里，我为你请到了一位心理学老师，让他来告诉我们生命的意义。他曾遭遇人类历史上罕见的挫折，却重获新生。他在活出的生命意义这本书里说，苦难不一定是追寻意义所必需的，承受不必要的苦难，与其说是英雄行为，不如说是自虐",
        "尽管有苦难，生命仍然可能有意义在正式开始讲他的故事之前，我必须先向你坦白我的担心。我很怕你以为这本书是心灵鸡汤，光是看活出生命的意义这个标题就能闻到一股鸡汤味了。我上学的时候最讨厌两类读物就是心灵鸡汤和成功学，而本书作者的经历可以说是成功学圣地了。不瞒你说，因为这本书的不瞒你说，因为这本书的标题和简介，我知道这本书之后很多年",
        "这是一种提问的技术，它是通过巧妙的提问引导对方发现自己思维中的矛盾。当苏格拉底和一名对话者交谈时，对话者可能会提出一个自认为不错的论点。为了方便区分，我们把这个论点称为 a 论点，此时苏格拉底并不会直接支持或反对 a 论点，而是会顺着对话者的思路提出与之相关的另外一个论点，我们可以就叫它是 b 论点，并询问对话者是否也同意 b 论点、如果对话者说他也同意 b 论点，那么苏格拉底便会通过提问引导对话者发现一个问题，也就是 a 论点和 b 论点实际上是不可能同时。于是对话者便会意识到，原来自己的想法是有自相矛盾的，存在着某些不易发现的错误。",
        '反解法的高明之处是苏格拉底总是能通过提问让对话者自己领悟到思考中的偏颇。我们可以比较一下，自己，发现自己错了，和别人告诉你错了是很不一样的，因为人总有一种防御机制，会自觉不自觉的维护自己，如果别人告诉我们自己哪里想错了，我们会忍不住去辩解，会感到懊恼，甚至会做出反击。'
        "但如果我们是自己发现错了，那么就只能承认这个错误，甚至会做出深刻的反思。从这个意义上说，反劫法是一种澄清自己思维的有效方法，不仅在两个人之间的对话中可以用，而且一个人也可以用，怎么用就是自己在头脑中进行对话，用反劫法的方式向自己提问，然后看看会不会发现自相矛盾的地方",
        "这里后来举个例子，看苏格拉底具体是怎么运用反间法的。苏格拉底曾经跟一位对话者讨论什么是勇敢对话者现给出了自己的定义，他说勇敢就是在精神上坚持不懈。苏格拉底顺着对话者的这个观点就问对话者说，你是不是认同勇敢是值得钦佩的？对话者说是啊，勇敢当然是值得钦佩的。然后苏格拉底接着就说，但是有些坚持是不明智的，属于盲目的坚持",
        "难道盲目的坚持是值得钦佩的吗？不值得钦佩。这时苏格拉底就告诉对话者说，你看你刚才说勇敢是一种坚持，并且勇敢是值得钦佩的。然后你又同意盲目的坚持是不值得钦佩的，这不就自相矛盾了吗？",
        "对话者一想说，对，那我刚才对勇敢的定义确实有问题。那么什么是盲目的坚持？例子很多，比如一个人搞投资总是亏钱，越投越亏，那这种坚持难道是勇敢吗？又或者一个人生了病躺在床上嗷嗷叫，但是医生给他开的药他就是不吃，难道这种坚持是勇敢吗"
    ]

    # asyncio.run(main())
    main()