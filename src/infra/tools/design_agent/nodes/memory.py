# src/infra/tools/design_agent/nodes/memory.py
import logging
import os
import json
from dataclasses import asdict
from typing import List

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, AnyMessage

from src.config.manager import config
from src.infra.tools.design_agent.schemas import StageEnum
from src.infra.tools.design_agent.nodes.utils import design_llm

logger = logging.getLogger(__name__)

class NoteHelper():
    def __init__(self):
        self.notes = ""
        self.note_file = os.path.join(os.path.dirname(__file__), '../../data/notes.md')
        os.makedirs(os.path.dirname(self.note_file), exist_ok=True)
        self.write = True
        if self.write:
            self.f = open(self.note_file, 'w')
        self.pending_messages = []
        self.pending_notes = ""
        try:
            self.llm = init_chat_model(**asdict(config.get_agent_config("notehelper_agent", "core").model))
        except Exception:
            try:
                self.llm = design_llm
            except Exception:
                self.llm = None

    def get_notes_messages(self, key='') -> List[HumanMessage]:
        try:
            note_text = self.get_notes()
            if not note_text:
                return []
            if key:
                note_text = f"[{key}]\n" + note_text
            return [HumanMessage(content=f"NOTES_SUMMARY:\n{note_text}")]
        except Exception:
            return []
    
    def get_notes(self):
        return self.notes + self.pending_notes

    async def do_notes(self, messages, use_llm=True):
        try:
            parts = []
            for m in messages or []:
                try:
                    prefix = ""
                    clsname = m.__class__.__name__ if hasattr(m, '__class__') else ''
                    if 'AIMessage' in clsname:
                        prefix = '[AI] '
                    elif 'HumanMessage' in clsname:
                        prefix = '[H] '
                    elif 'ToolMessage' in clsname or 'AnyMessage' in clsname:
                        prefix = '[TOOL] '
                    parts.append(prefix + getattr(m, 'content', str(m)))
                except Exception:
                    parts.append(str(m))

            for p in parts:
                self.pending_messages.append(p)
            joined = "\n\n".join(parts) if parts else ""
            if joined:
                self.pending_notes = (self.pending_notes + "\n\n" + joined) if self.pending_notes else joined

            if use_llm and self.llm is not None and len(self.pending_messages) >= getattr(self, 'llm_interval', 3):
                human = HumanMessage(content=("Existing notes:\n\n % s\n\n Summarise the following new messages to short markdown notes chunk, increasingly based on exsiting notes:\n\n % s" % (self.notes[-200:], self.pending_notes)))
                llm_input = [SystemMessage(content=(getattr(self, 'system_message', 'You are a concise note-taking assistant.'))), human]
                try:
                    llm_resp = await self.llm.ainvoke(llm_input)
                    note_text = getattr(llm_resp, 'content', str(llm_resp))
                    self.notes = (self.notes + "\n\n" + note_text).strip() if self.notes else note_text
                    self.pending_notes = ""
                    self.pending_messages = []
                    if self.write:
                        self.f.write(note_text)
                        self.f.flush()
                        logger.info(f"-> NoteHelper.do_notes: {note_text}")
                except Exception:
                    if self.pending_notes:
                        self.notes = (self.notes + "\n\n" + self.pending_notes).strip() if self.notes else self.pending_notes
                        self.pending_notes = ""
                        self.pending_messages = []
            else:
                if not use_llm:
                    pass
            return self.notes
        except Exception as e:
            logger.warning(f"NoteHelper.do_notes failed: {e}")
            return self.notes

try:
    note_helper = NoteHelper()
except Exception:
    note_helper = None

async def update_memory_node(state, config=None):
    logger.info("-> update_memory_node entered")
    try:
        short_list = state.get('short_memorys', [])
        if not short_list:
            return {
                "messages": [AIMessage(content="No short memories to update.")],
                "stage": StageEnum.REFLECT,
            }

        summaries = []
        for s in short_list[-10:]:
            try:
                if isinstance(s, dict):
                    summaries.append(s.get('current_idea', str(s)))
                else:
                    summaries.append(str(s))
            except Exception:
                summaries.append(str(s))

        long_summary = " | ".join([str(x) for x in summaries if x])[:2000]

        long_mem = state.get('long_memory', {}) or {}
        long_mem['most_consistent_with_user_instructions_ideas'] = long_summary

        ai_msg = AIMessage(content=json.dumps({"updated_long_memory": True, "summary_len": len(long_summary)}))

        return {
            "long_memory": long_mem,
            "short_memorys": [],
            "messages": [ai_msg],
            "stage": StageEnum.REFLECT,
            "stage_iteration": state.get('stage_iteration', 0) + 1,
        }
    except Exception as e:
        logger.error(f"Error in update_memory_node: {e}")
        return {"messages": state.get('messages', [])}
