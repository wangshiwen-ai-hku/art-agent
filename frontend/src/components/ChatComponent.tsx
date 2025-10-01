import React, { useState, useRef, useLayoutEffect} from 'react';
import { useCanvasStore } from '../store/useCanvasStore';
import { 
  Send, 
  Bot, 
  User, 
  MessageSquare, 
  Edit, 
  Wand2, 
  Palette,
  FileText,
  Upload,
  X,
  Minimize2,
  RefreshCw,
  Settings,
  Sparkles,
  Zap,
  Heart
} from 'lucide-react';

type AgentStage = 'generate' | 'edit' | 'chat' | 'draw' | 'describe';


const ChatComponent: React.FC = () => {
  const { messages, addMessage, selectionBox, addSvg} = useCanvasStore();
  const [inputValue, setInputValue] = useState('');
  const [uploadedSvg, setUploadedSvg] = useState<string | null>(null);
  const [uploadedName, setUploadedName] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [currentStage, setCurrentStage] = useState<AgentStage>('chat');
  const [isMinimized, setIsMinimized] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const messagesEndRef = useRef<HTMLDivElement | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

  // Auto-scroll to bottom when new messages arrive
  useLayoutEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Focus textarea when stage changes
  useLayoutEffect(() => {
    if (!isMinimized) {
      textareaRef.current?.focus();
    }
  }, [currentStage, isMinimized]);

  // Auto-resize textarea based on content
  useLayoutEffect(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = 'auto';
      const newHeight = Math.min(textarea.scrollHeight, 200);
      textarea.style.height = `${newHeight}px`;
    }
  }, [inputValue]);

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files && e.target.files[0];
    if (!file) return;
    setUploadedName(file.name);
    try {
      const text = await file.text();
      if (file.name.toLowerCase().endsWith('.svg')) {
        addSvg(text);
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
      content += `\n[Selection: x=${selectionBox.x.toFixed(1)}, y=${selectionBox.y.toFixed(1)}, w=${selectionBox.width.toFixed(1)}, h=${selectionBox.height.toFixed(1)}]`;
    }
    if (uploadedName) {
      content += `\n[Uploaded: ${uploadedName}]`;
    }

    addMessage({ 
      author: 'user', 
      content: `[${currentStage}] ${content}` 
    });
    setInputValue('');
    setUploadedName(null);
    
    // Reset textarea height after sending
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }

    setIsLoading(true);
    try {
      const payload = {
        message: content,
        stage: currentStage,
        svg: uploadedSvg,
      };

      const res = await fetch('http://localhost:8001/api/canvas/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const text = await res.text();
        addMessage({ 
          author: 'agent', 
          content: `⚠️ Error: ${res.status} ${text}`
        });
      } else {
        const data = await res.json();
        if (data.reply) {
          addMessage({ 
            author: 'agent', 
            content: data.reply
          });
        }
        if (data.svg) {
          addSvg(data.svg);
        }
        if (data.tool_outputs && data.tool_outputs.length > 0) {
          const toolOutputContent = `🛠️ Tools used:\n${data.tool_outputs.join('\n')}`;
          addMessage({ 
            author: 'agent', 
            content: toolOutputContent
          });
        }
      }
    } catch (err) {
      console.error(err);
      addMessage({ 
        author: 'agent', 
        content: '⚠️ Network error: Could not connect to the server.'
      });
    } finally {
      setIsLoading(false);
      setUploadedSvg(null);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  const clearChat = () => {
    console.log('Clear chat functionality would be implemented here');
  };

  // Stage configurations with colors and icons, and save global mode_color, so that the main pannel's color will change
  // MODE_COLOR is the color of the main pannel
  const stageConfigs = {
    generate: { 
      icon: Wand2, 
      colors: { primary: '#a855f7', secondary: '#ec4899' }, 
      bg: '#a855f7'
    },
    edit: { 
      icon: Edit, 
      colors: { primary: '#3b82f6', secondary: '#06b6d4' }, 
      bg: '#3b82f6'
    },
    chat: { 
      icon: MessageSquare, 
      colors: { primary: '#10b981', secondary: '#059669' }, 
      bg: '#10b981'
    },
    draw: { 
      icon: Palette, 
      colors: { primary: '#f97316', secondary: '#ef4444' }, 
      bg: '#f97316'
    },
    describe: { 
      icon: FileText, 
      colors: { primary: '#6366f1', secondary: '#8b5cf6' }, 
      bg: '#6366f1'
    }
  };
  
  const currentConfig = stageConfigs[currentStage];


  const containerStyle: React.CSSProperties = {
    position: 'fixed',
    top: '16px',
    right: '16px',
    bottom: '16px',
    width: '420px',
    backgroundColor: 'rgba(255, 255, 255, 0.95)',
    backdropFilter: 'blur(20px)',
    borderRadius: '24px',
    boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.25), 0 0 0 1px rgba(255, 255, 255, 0.2)',
    overflow: 'hidden',
    transition: 'all 0.5s ease',
    display: 'flex',
    flexDirection: 'column',
  };

  const headerStyle: React.CSSProperties = {
    position: 'relative',
    padding: '16px 24px',
    background: `linear-gradient(135deg, ${currentConfig.colors.primary}, ${currentConfig.colors.secondary})`,
    color: 'white',
  };

  const headerOverlayStyle: React.CSSProperties = {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    backdropFilter: 'blur(8px)',
  };
  

  return (
    <div style={isMinimized ? {...containerStyle, height: '80px', bottom: '16px', top: 'auto'} : containerStyle}>
      {/* Decorative background animations */}
      <div style={{
        position: 'absolute',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        overflow: 'hidden',
        borderRadius: '24px',
        pointerEvents: 'none'
      }}>
        <div style={{
          position: 'absolute',
          top: '-50%',
          right: '-50%',
          width: '384px',
          height: '384px',
          background: 'linear-gradient(135deg, rgba(147, 197, 253, 0.2), rgba(196, 181, 253, 0.2))',
          borderRadius: '50%',
          filter: 'blur(60px)',
          animation: 'pulse 2s infinite'
        }}></div>
        <div style={{
          position: 'absolute',
          bottom: '-50%',
          left: '-50%',
          width: '384px',
          height: '384px',
          background: 'linear-gradient(45deg, rgba(252, 165, 165, 0.2), rgba(254, 215, 170, 0.2))',
          borderRadius: '50%',
          filter: 'blur(60px)',
          animation: 'pulse 2s infinite',
          animationDelay: '1s'
        }}></div>
      </div>

      {/* Header with gradient and glass effect */}
      <div style={headerStyle}>
        <div style={headerOverlayStyle}></div>
        <div style={{ position: 'relative', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <div style={{ position: 'relative' }}>
              <div style={{
                width: '40px',
                height: '40px',
                backgroundColor: 'rgba(255, 255, 255, 0.2)',
                borderRadius: '50%',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                backdropFilter: 'blur(8px)'
              }}>
                <Bot size={20} />
              </div>
              <div style={{
                position: 'absolute',
                top: '-4px',
                right: '-4px',
                width: '16px',
                height: '16px',
                backgroundColor: '#4ade80',
                borderRadius: '50%',
                border: '2px solid white',
                animation: 'pulse 2s infinite'
              }}></div>
            </div>
            <div>
              <h2 style={{ 
                fontWeight: 'bold', 
                fontSize: '18px', 
                margin: 0, 
                display: 'flex', 
                alignItems: 'center', 
                gap: '8px' 
              }}>
                AI Drawing Assistant 
                <Sparkles size={16} style={{ animation: 'spin 2s linear infinite' }} />
              </h2>
              <p style={{ fontSize: '14px', opacity: 0.9, margin: 0 }}>Ready to create magic ✨</p>
            </div>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <button 
              onClick={() => setShowSettings(!showSettings)} 
              style={{
                padding: '8px',
                backgroundColor: 'transparent',
                border: 'none',
                borderRadius: '12px',
                color: 'white',
                cursor: 'pointer',
                transition: 'all 0.2s',
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.backgroundColor = 'rgba(255, 255, 255, 0.2)';
                e.currentTarget.style.transform = 'scale(1.05)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.backgroundColor = 'transparent';
                e.currentTarget.style.transform = 'scale(1)';
              }}
            >
              <Settings size={16} />
            </button>
            <button 
              onClick={clearChat}
              style={{
                padding: '8px',
                backgroundColor: 'transparent',
                border: 'none',
                borderRadius: '12px',
                color: 'white',
                cursor: 'pointer',
                transition: 'all 0.2s',
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.backgroundColor = 'rgba(255, 255, 255, 0.2)';
                e.currentTarget.style.transform = 'scale(1.05)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.backgroundColor = 'transparent';
                e.currentTarget.style.transform = 'scale(1)';
              }}
            >
              <RefreshCw size={16} />
            </button>
            <button 
              onClick={() => setIsMinimized(!isMinimized)} 
              style={{
                padding: '8px',
                backgroundColor: 'transparent',
                border: 'none',
                borderRadius: '12px',
                color: 'white',
                cursor: 'pointer',
                transition: 'all 0.2s',
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.backgroundColor = 'rgba(255, 255, 255, 0.2)';
                e.currentTarget.style.transform = 'scale(1.05)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.backgroundColor = 'transparent';
                e.currentTarget.style.transform = 'scale(1)';
              }}
            >
              <Minimize2 size={16} />
            </button>
          </div>
        </div>
      </div>

      {/* Settings Panel */}
      {showSettings && !isMinimized && (
        <div style={{
          padding: '12px 24px',
          backgroundColor: 'rgba(255, 255, 255, 0.5)',
          backdropFilter: 'blur(8px)',
          borderBottom: '1px solid rgba(255, 255, 255, 0.3)'
        }}>
          <div style={{ 
            fontSize: '14px', 
            color: '#374151', 
            display: 'flex', 
            alignItems: 'center', 
            gap: '8px' 
          }}>
            <Zap size={16} style={{ color: '#eab308' }} />
            <span>Premium features coming soon...</span>
          </div>
        </div>
      )}

      {!isMinimized && (
        <>
          {/* Mode Selection */}
          <div style={{
            padding: '20px 24px',
            borderBottom: '1px solid rgba(255, 255, 255, 0.3)',
            backgroundColor: 'rgba(255, 255, 255, 0.3)',
            backdropFilter: 'blur(8px)'
          }}>
            <div style={{ 
              display: 'flex', 
              alignItems: 'center', 
              gap: '8px', 
              fontSize: '14px', 
              marginBottom: '16px' 
            }}>
              <span style={{ color: '#4b5563', fontWeight: 600 }}>Current Mode:</span>
              <div style={{
                padding: '6px 16px',
                borderRadius: '20px',
                background: `linear-gradient(135deg, ${currentConfig.colors.primary}, ${currentConfig.colors.secondary})`,
                color: 'white',
                fontSize: '13px',
                fontWeight: 'bold',
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
                boxShadow: '0 4px 12px rgba(0, 0, 0, 0.15)'
              }}>
                <currentConfig.icon size={14} />
                {currentStage.charAt(0).toUpperCase() + currentStage.slice(1)}
              </div>
            </div>

            <div style={{ 
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(110px, 1fr))',
              gap: '12px'
            }}>
              {Object.entries(stageConfigs).map(([stage, config]) => {
                const isActive = currentStage === stage;
                return (
                  <button 
                    key={stage}
                    onClick={() => setCurrentStage(stage as AgentStage)}
                    style={{
                      padding: '12px 16px',
                      borderRadius: '16px',
                      display: 'flex',
                      flexDirection: 'column',
                      alignItems: 'center',
                      gap: '6px',
                      fontWeight: 600,
                      fontSize: '12px',
                      border: 'none',
                      cursor: 'pointer',
                      transition: 'all 0.3s',
                      whiteSpace: 'nowrap',
                      minHeight: '70px',
                      ...(isActive ? {
                        background: `linear-gradient(135deg, ${config.colors.primary}, ${config.colors.secondary})`,
                        color: 'white',
                        boxShadow: '0 6px 20px rgba(0, 0, 0, 0.2)',
                        transform: 'translateY(-2px)'
                      } : {
                        color: '#6b7280',
                        backgroundColor: 'rgba(255, 255, 255, 0.7)',
                        backdropFilter: 'blur(8px)',
                        border: '2px solid rgba(255, 255, 255, 0.4)',
                        boxShadow: '0 2px 8px rgba(0, 0, 0, 0.1)'
                      })
                    }}
                    onMouseEnter={(e) => {
                      if (!isActive) {
                        e.currentTarget.style.backgroundColor = 'rgba(255, 255, 255, 0.9)';
                        e.currentTarget.style.borderColor = `${config.colors.primary}40`;
                        e.currentTarget.style.color = config.colors.primary;
                      }
                      e.currentTarget.style.transform = isActive ? 'translateY(-4px)' : 'translateY(-2px)';
                    }}
                    onMouseLeave={(e) => {
                      if (!isActive) {
                        e.currentTarget.style.backgroundColor = 'rgba(255, 255, 255, 0.7)';
                        e.currentTarget.style.borderColor = 'rgba(255, 255, 255, 0.4)';
                        e.currentTarget.style.color = '#6b7280';
                      }
                      e.currentTarget.style.transform = isActive ? 'translateY(-2px)' : 'translateY(0px)';
                    }}
                  >
                    <config.icon size={24} />
                    <span>{stage.charAt(0).toUpperCase() + stage.slice(1)}</span>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Chat Messages */}
          <div style={{
            flex: 1,
            overflowY: 'auto',
            padding: '20px',
            background: 'linear-gradient(to bottom, rgba(249, 250, 251, 0.5), rgba(255, 255, 255, 0.5))',
            backdropFilter: 'blur(8px)'
          }}>
            {messages.length === 0 ? (
              <div style={{ 
                textAlign: 'center', 
                color: '#6b7280', 
                paddingTop: '80px', 
                paddingBottom: '80px' 
              }}>
                <div style={{
                  width: '80px',
                  height: '80px',
                  background: 'linear-gradient(135deg, rgb(219, 234, 254), rgb(237, 233, 254))',
                  borderRadius: '50%',
                  margin: '0 auto 24px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center'
                }}>
                  <Heart size={40} style={{ color: '#ec4899', animation: 'pulse 2s infinite' }} />
                </div>
                <p style={{ fontWeight: 600, fontSize: '22px', marginBottom: '12px' }}>
                  Ready to create something amazing?
                </p>
                <p style={{ fontSize: '16px', opacity: 0.75, lineHeight: 1.5 }}>
                  Choose a mode above and start chatting!<br />
                  I can help you generate, edit, draw, and describe your creative visions.
                </p>
              </div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                {messages.map((msg, index) => (
                  <div 
                    key={index} 
                    style={{ 
                      display: 'flex', 
                      justifyContent: msg.author === 'user' ? 'flex-end' : 'flex-start',
                      animation: 'fadeIn 0.3s ease-out'
                    }}
                  >
                    <div style={{
                      maxWidth: '85%',
                      borderRadius: '16px',
                      padding: '12px 16px',
                      position: 'relative',
                      transition: 'all 0.3s',
                      ...(msg.author === 'user' ? {
                        background: `linear-gradient(135deg, ${currentConfig.colors.primary}, ${currentConfig.colors.secondary})`,
                        color: 'white',
                        borderBottomRightRadius: '4px',
                        boxShadow: '0 4px 12px rgba(0, 0, 0, 0.15)'
                      } : {
                        backgroundColor: 'rgba(255, 255, 255, 0.8)',
                        color: '#1f2937',
                        borderBottomLeftRadius: '4px',
                        boxShadow: '0 4px 12px rgba(0, 0, 0, 0.1)',
                        border: '1px solid rgba(255, 255, 255, 0.3)',
                        backdropFilter: 'blur(8px)'
                      })
                    }}>
                      {msg.author === 'agent' && (
                        <div style={{
                          width: '24px',
                          height: '24px',
                          background: 'linear-gradient(135deg, #3b82f6, #8b5cf6)',
                          borderRadius: '50%',
                          position: 'absolute',
                          top: '-8px',
                          left: '-8px',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center'
                        }}>
                          <Bot size={12} style={{ color: 'white' }} />
                        </div>
                      )}
                      {msg.author === 'user' && (
                        <div style={{
                          width: '24px',
                          height: '24px',
                          background: `linear-gradient(135deg, ${currentConfig.colors.primary}, ${currentConfig.colors.secondary})`,
                          borderRadius: '50%',
                          position: 'absolute',
                          top: '-8px',
                          right: '-8px',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center'
                        }}>
                          <User size={12} style={{ color: 'white' }} />
                        </div>
                      )}
                      <p style={{ 
                        fontSize: '14px', 
                        whiteSpace: 'pre-wrap', 
                        lineHeight: '1.5', 
                        margin: 0 
                      }}>
                        {msg.content}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            )}
            
            {isLoading && (
              <div style={{ display: 'flex', justifyContent: 'flex-start' }}>
                <div style={{
                  backgroundColor: 'rgba(255, 255, 255, 0.9)',
                  padding: '16px 24px',
                  borderRadius: '16px',
                  boxShadow: '0 4px 12px rgba(0, 0, 0, 0.1)',
                  border: '1px solid rgba(255, 255, 255, 0.3)',
                  backdropFilter: 'blur(8px)'
                }}>
                  <div style={{ 
                    display: 'flex', 
                    alignItems: 'center', 
                    gap: '12px', 
                    color: '#4b5563' 
                  }}>
                    <div style={{ display: 'flex', gap: '4px' }}>
                      {[0, 1, 2].map((i) => (
                        <div 
                          key={i}
                          style={{
                            width: '8px',
                            height: '8px',
                            background: `linear-gradient(135deg, ${currentConfig.colors.primary}, ${currentConfig.colors.secondary})`,
                            borderRadius: '50%',
                            animation: 'bounce 1s infinite',
                            animationDelay: `${i * 0.1}s`
                          }}
                        ></div>
                      ))}
                    </div>
                    <span style={{ fontSize: '14px', fontWeight: 500 }}>AI is creating magic...</span>
                    <Sparkles size={16} style={{ color: '#eab308', animation: 'spin 2s linear infinite' }} />
                  </div>
                </div>
              </div>
            )}
            
            <div ref={messagesEndRef} />
          </div>

          {/* Input Area */}
          <div style={{
            padding: '20px',
            backgroundColor: 'rgba(255, 255, 255, 0.6)',
            backdropFilter: 'blur(8px)',
            borderTop: '1px solid rgba(255, 255, 255, 0.3)',
            flexShrink: 0
          }}>
            {/* File upload and selection info */}
            {(uploadedName || selectionBox) && (
              <div style={{ 
                marginBottom: '12px', 
                display: 'flex', 
                alignItems: 'center', 
                gap: '8px', 
                fontSize: '14px',
                flexWrap: 'wrap'
              }}>
                {uploadedName && (
                  <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    color: '#1d4ed8',
                    padding: '8px 12px',
                    borderRadius: '20px',
                    backdropFilter: 'blur(8px)',
                    border: '1px solid rgba(59, 130, 246, 0.2)'
                  }}>
                    <Upload size={16} />
                    <span style={{ 
                      maxWidth: '128px', 
                      overflow: 'hidden', 
                      textOverflow: 'ellipsis', 
                      fontWeight: 500 
                    }}>
                      {uploadedName}
                    </span>
                    <button 
                      onClick={() => setUploadedName(null)}
                      style={{
                        backgroundColor: 'transparent',
                        border: 'none',
                        borderRadius: '50%',
                        padding: '2px',
                        marginLeft: '4px',
                        cursor: 'pointer',
                        transition: 'all 0.2s'
                      }}
                      onMouseEnter={(e) => {
                        e.currentTarget.style.backgroundColor = 'rgba(59, 130, 246, 0.2)';
                      }}
                      onMouseLeave={(e) => {
                        e.currentTarget.style.backgroundColor = 'transparent';
                      }}
                    >
                      <X size={12} />
                    </button>
                  </div>
                )}
                {selectionBox && (
                  <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px',
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    color: '#047857',
                    padding: '8px 12px',
                    borderRadius: '20px',
                    backdropFilter: 'blur(8px)',
                    border: '1px solid rgba(16, 185, 129, 0.2)'
                  }}>
                    <div style={{
                      width: '16px',
                      height: '16px',
                      border: '2px solid currentColor',
                      borderRadius: '2px',
                      animation: 'pulse 2s infinite'
                    }}></div>
                    <span style={{ fontWeight: 500 }}>Selection active</span>
                  </div>
                )}
              </div>
            )}

            {/* Input container */}
            <div style={{
              position: 'relative',
              backgroundColor: 'rgba(255, 255, 255, 0.8)',
              backdropFilter: 'blur(8px)',
              border: '2px solid rgba(209, 213, 219, 0.5)',
              borderRadius: '16px',
              boxShadow: '0 4px 12px rgba(0, 0, 0, 0.1)',
              overflow: 'hidden',
              transition: 'all 0.3s'
            }}>
              <textarea
                ref={textareaRef}
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey && !isLoading) {
                    e.preventDefault();
                    handleSendMessage();
                  }
                }}
                placeholder={`Ask me to ${
                  currentStage === 'generate' ? 'generate something magical' : 
                  currentStage === 'edit' ? 'edit your creation' : 
                  currentStage === 'draw' ? 'draw something beautiful' : 
                  currentStage === 'describe' ? 'describe your vision' : 
                  'chat about anything'
                }...`}
                disabled={isLoading}
                style={{
                  width: '100%',
                  padding: '16px 56px',
                  backgroundColor: 'transparent',
                  border: 'none',
                  outline: 'none',
                  fontSize: '16px',
                  resize: 'none',
                  minHeight: '60px',
                  maxHeight: '160px',
                  overflow: 'auto',
                  color: '#1f2937'
                }}
              />
              
              {/* Upload button */}
              <label 
                htmlFor="svgUpload" 
                style={{
                  position: 'absolute',
                  left: '12px',
                  bottom: '12px',
                  width: '40px',
                  height: '40px',
                  backgroundColor: 'rgba(156, 163, 175, 0.8)',
                  borderRadius: '12px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  cursor: 'pointer',
                  transition: 'all 0.2s',
                  boxShadow: '0 2px 4px rgba(0, 0, 0, 0.1)',
                  backdropFilter: 'blur(8px)',
                  border: '1px solid rgba(255, 255, 255, 0.3)'
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.backgroundColor = 'rgba(107, 114, 128, 0.8)';
                  e.currentTarget.style.transform = 'scale(1.05)';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.backgroundColor = 'rgba(156, 163, 175, 0.8)';
                  e.currentTarget.style.transform = 'scale(1)';
                }}
                title="Upload SVG file"
              >
                <Upload size={20} style={{ color: '#4b5563' }} />
              </label>
              <input 
                id="svgUpload" 
                ref={fileInputRef}
                type="file" 
                accept="image/svg+xml" 
                onChange={handleFileChange} 
                style={{ display: 'none' }}
              />
              
              {/* Send button */}
              <button 
                type="button"
                onClick={handleSendMessage}
                disabled={isLoading || (!inputValue.trim() && !uploadedSvg)}
                style={{
                  position: 'absolute',
                  right: '12px',
                  bottom: '12px',
                  width: '40px',
                  height: '40px',
                  borderRadius: '12px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  border: 'none',
                  transition: 'all 0.3s',
                  boxShadow: '0 4px 12px rgba(0, 0, 0, 0.15)',
                  ...(!inputValue.trim() && !uploadedSvg) || isLoading ? {
                    backgroundColor: 'rgba(209, 213, 219, 0.8)',
                    color: '#9ca3af',
                    cursor: 'not-allowed'
                  } : {
                    background: `linear-gradient(135deg, ${currentConfig.colors.primary}, ${currentConfig.colors.secondary})`,
                    color: 'white',
                    cursor: 'pointer'
                  }
                }}
                onMouseEnter={(e) => {
                  if (!isLoading && (inputValue.trim() || uploadedSvg)) {
                    e.currentTarget.style.transform = 'scale(1.05)';
                  }
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.transform = 'scale(1)';
                }}
                title={isLoading ? 'Sending...' : 'Send message'}
              >
                {isLoading ? (
                  <RefreshCw size={20} style={{ animation: 'spin 1s linear infinite' }} />
                ) : (
                  <Send size={20} />
                )}
              </button>
            </div>

            {/* Quick actions */}
            <div style={{
              marginTop: '12px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '12px',
              fontSize: '12px',
              color: '#6b7280'
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                <div style={{
                  width: '6px',
                  height: '6px',
                  backgroundColor: '#60a5fa',
                  borderRadius: '50%',
                  animation: 'pulse 2s infinite'
                }}></div>
                <span>Press Enter to send</span>
              </div>
              <span>•</span>
              <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                <div style={{
                  width: '6px',
                  height: '6px',
                  backgroundColor: '#34d399',
                  borderRadius: '50%',
                  animation: 'pulse 2s infinite',
                  animationDelay: '0.2s'
                }}></div>
                <span>Shift+Enter for new line</span>
              </div>
              <span>•</span>
              <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                <div style={{
                  width: '6px',
                  height: '6px',
                  backgroundColor: '#a78bfa',
                  borderRadius: '50%',
                  animation: 'pulse 2s infinite',
                  animationDelay: '0.4s'
                }}></div>
                <span>Click modes to switch</span>
              </div>
            </div>
          </div>
        </>
      )}

      {/* Add CSS animations */}
      <style>
        {`
          @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
          }
          @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
          }
          @keyframes spin {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
          }
          @keyframes bounce {
            0%, 20%, 53%, 80%, 100% { transform: translateY(0); }
            40%, 43% { transform: translateY(-8px); }
          }
        `}
      </style>
    </div>
  );
};

export default ChatComponent;