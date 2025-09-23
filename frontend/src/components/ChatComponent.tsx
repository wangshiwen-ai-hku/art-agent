import React, { useState, useRef } from 'react';
import { useCanvasStore } from '../store/useCanvasStore';
import { Send, Bot, User, MessageSquare, Edit, BotIcon, Wand2 } from 'lucide-react';

type AgentStage = 'generate' | 'edit' | 'chat' | 'draw' | 'describe';

const ChatComponent: React.FC = () => {
  const { messages, addMessage, selectionBox, addSvg } = useCanvasStore();
  const [inputValue, setInputValue] = useState('');
  const [uploadedSvg, setUploadedSvg] = useState<string | null>(null);
  const [uploadedName, setUploadedName] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [currentStage, setCurrentStage] = useState<AgentStage>('chat');
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files && e.target.files[0];
    if (!file) return;
    setUploadedName(file.name);
    try {
      const text = await file.text();
      if (file.name.toLowerCase().endsWith('.svg')) {
        // Immediately add the uploaded SVG to the global store so the canvas renders it
        addSvg(text);
        // Keep a copy for potential send, but clear since we've already added to canvas
        setUploadedSvg(null);
      } else {
        setUploadedSvg(null);
      }
    } catch (err) {
      console.error('Failed to read file', err);
      setUploadedSvg(null);
    }
  };

  const handleSendMessage = async () => {
    if (!inputValue.trim() && !uploadedSvg) return;

    let content = inputValue.trim();
    if (selectionBox) {
      content += `\n[Box: x=${selectionBox.x}, y=${selectionBox.y}, width=${selectionBox.width}, height=${selectionBox.height}]`;
    }
    if (uploadedName) {
      content += `\n[Uploaded: ${uploadedName}]`;
    }

    addMessage({ author: 'user', content: `[${currentStage}] ${content}` });
    setInputValue('');
    setUploadedName(null);

    setIsLoading(true);
    try {
      const payload = {
        message: content,
        stage: currentStage,
        svg: uploadedSvg,
      };

      const res = await fetch('http://localhost:8001/api/canvas/chat', { // http://localhost:8001/api/canvas/chat
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const txt = await res.text();
        addMessage({ author: 'agent', content: `⚠️ Error: ${res.status} ${txt}` });
      } else {
        const data = await res.json();
        if (data.reply) {
          addMessage({ author: 'agent', content: data.reply });
        }
        if (data.svg) {
          addSvg(data.svg);
        }
        if (data.tool_outputs && data.tool_outputs.length > 0) {
          const toolOutputContent = `🛠️ Tools used:\n${data.tool_outputs.join('\n')}`;
          addMessage({ author: 'agent', content: toolOutputContent });
        }
      }
    } catch (err) {
      console.error(err);
      addMessage({ author: 'agent', content: '⚠️ Network error: Could not connect to the server.' });
    } finally {
      setIsLoading(false);
      setUploadedSvg(null);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  const ModeButton = ({ stage, label, icon }: { stage: AgentStage, label: string, icon: React.ReactNode }) => (
    <button
      onClick={() => setCurrentStage(stage)}
      className={`flex items-center gap-2 px-3 py-1.5 text-sm rounded-full transition-all duration-200 ${
        currentStage === stage
          ? 'bg-blue-600 text-white shadow-md'
          : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
      }`}
    >
      {icon}
      {label}
    </button>
  );

  return (
    <div className="flex flex-col h-full bg-gradient-to-b from-white to-gray-50 font-sans" style={{ fontFamily: 'Inter, Roboto, -apple-system, BlinkMacSystemFont, "Segoe UI", "Helvetica Neue", Arial' }}>
      <div className="flex-1 p-4 overflow-y-auto">
        <div className="space-y-4">
          {messages.map((msg, index) => (
            <div key={index} className={`flex items-start gap-3 ${msg.author === 'user' ? 'justify-end' : ''}`}>
              {msg.author === 'agent' && <Bot className="w-6 h-6 text-gray-500 flex-shrink-0" />}
              <div className={`px-6 py-3 rounded-2xl shadow-lg max-w-md break-words ${msg.author === 'user' ? 'bg-blue-600 text-white' : 'bg-white text-gray-800'}`} style={{ fontSize: 16, lineHeight: '1.5rem' }}>
                {msg.content}
              </div>
              {msg.author === 'user' && <User className="w-6 h-6 text-gray-500 flex-shrink-0" />}
            </div>
          ))}
        </div>
      </div>

      <div className="p-4 border-t border-gray-200 bg-white">
        <div className="mb-3 flex items-center justify-center gap-2 flex-wrap">
            <ModeButton stage="generate" label="Generate" icon={<Wand2 size={14}/>} />
            <ModeButton stage="edit" label="Edit" icon={<Edit size={14}/>} />
            <ModeButton stage="chat" label="Chat" icon={<MessageSquare size={14}/>} />
            <ModeButton stage="draw" label="Draw" icon={<BotIcon size={14}/>} />
            <ModeButton stage="describe" label="Describe" icon={<BotIcon size={14}/>} />
        </div>
        <div className="mb-3 flex items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <label htmlFor="svgUpload" className="inline-flex items-center gap-2 px-3 py-2 rounded-md bg-gray-100 hover:bg-gray-200 cursor-pointer shadow-sm">
              <span role="img" aria-label="attach">📎</span>
              <span className="text-sm text-gray-700">Upload SVG</span>
            </label>
            <input id="svgUpload" ref={fileInputRef} type="file" accept="image/svg+xml" onChange={handleFileChange} className="hidden" />
            {uploadedName && <div className="text-sm text-gray-600">File: {uploadedName}</div>}
          </div>
          <div className="text-sm text-gray-500">{isLoading ? 'Processing…' : `Mode: ${currentStage}`} </div>
        </div>

        <div className="relative">
          <input
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && !isLoading && handleSendMessage()}
            placeholder="Type your message... ✍️"
            className="w-full pl-4 pr-20 py-3 border border-gray-300 rounded-full focus:outline-none focus:ring-2 focus:ring-blue-500"
            style={{ fontSize: 16 }}
          />
          <button disabled={isLoading} onClick={handleSendMessage} className="absolute right-2 top-1/2 -translate-y-1.2 p-2 rounded-full bg-gradient-to-br from-blue-500 to-indigo-600 text-white hover:opacity-95 shadow-md disabled:opacity-50 disabled:cursor-not-allowed">
            <Send size={18} />
          </button>
        </div>
      </div>
    </div>
  );
};

export default ChatComponent; 