import React, { useRef, useEffect, useState } from 'react';
import { fabric } from 'fabric';
import { useCanvasStore } from '../store/useCanvasStore';

// 定义视图模式类型
type ViewMode = 'canvas' | 'png' | 'source' | 'split';

const CanvasComponent: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const fabricCanvasRef = useRef<fabric.Canvas | null>(null);
  const previewCanvasRef = useRef<HTMLCanvasElement>(null); // 独立的预览 canvas
  const previewFabricCanvasRef = useRef<fabric.Canvas | null>(null);
  
  const editorTextareaRef = useRef<HTMLTextAreaElement>(null);
  const editorHighlightRef = useRef<HTMLPreElement>(null);
  const { svgHistory, currentSvgIndex, addSvg, setSelectionBox, updateSvg } = useCanvasStore();
  const [isSpacePressed, setIsSpacePressed] = useState(false);
  const spacePressedRef = useRef(false);
  const [viewMode, setViewMode] = useState<ViewMode>('canvas');
  const [pngDataUrl, setPngDataUrl] = useState<string | null>(null);
  const [splitOrientation, setSplitOrientation] = useState<'horizontal' | 'vertical'>('horizontal');
  const [splitRatio, setSplitRatio] = useState(50); // 50% 分割比例
  const [isDraggingSplitter, setIsDraggingSplitter] = useState(false);
  const [codeTheme, setCodeTheme] = useState<'dark' | 'light'>('dark');
  const [codeFontSize, setCodeFontSize] = useState(14);
  const debounceTimerRef = useRef<number | null>(null);
  const [downloading, setDownloading] = useState<'svg' | 'png' | null>(null);
  const resizeObserverRef = useRef<ResizeObserver | null>(null);
  const [tempSvgCode, setTempSvgCode] = useState<string>(''); // 临时编辑的 SVG 代码
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false);
  const [svgError, setSvgError] = useState<string | null>(null); // SVG 加载错误信息

  // 代码主题样式
  const codeThemes = {
    dark: {
      background: '#1e1e1e',
      foreground: '#d4d4d4',
      selection: '#264f78',
      lineHighlight: '#2d2d30',
      keyword: '#569cd6',
      string: '#ce9178',
      comment: '#6a9955',
      tag: '#569cd6',
      attribute: '#9cdcfe',
      value: '#ce9178'
    },
    light: {
      background: '#ffffff',
      foreground: '#333333',
      selection: '#add6ff',
      lineHighlight: '#f0f0f0',
      keyword: '#0000ff',
      string: '#a31515',
      comment: '#008000',
      tag: '#800000',
      attribute: '#ff0000',
      value: '#a31515'
    }
  };

  // 代码字体选项
  const codeFonts = [
    'Monaco, Menlo, "Ubuntu Mono", monospace',
    '"Fira Code", monospace',
    '"Source Code Pro", monospace',
    '"Courier New", monospace',
    'Consolas, monospace'
  ];

  // 简单 SVG 语法高亮
  const escapeHtml = (str: string) => str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');

  const highlightSvg = (code: string) => {
    let html = escapeHtml(code);
    // comments
    html = html.replace(/&lt;!--[\s\S]*?--&gt;/g, (_m) => `<span style="color:#6a9955;">${_m}</span>`);
    // attribute values (strings)
    html = html.replace(/(=)(&quot;[\s\S]*?&quot;|&#39;[\s\S]*?&#39;)/g, (_m, p1, p2) => `${p1}<span style="color:#ce9178;">${p2}</span>`);
    // attribute names
    html = html.replace(/(\s)([a-zA-Z_:][-a-zA-Z0-9_:.]*)(?=\s*=)/g, (_m, p1, p2) => `${p1}<span style="color:#9cdcfe;">${p2}</span>`);
    // tags
    html = html.replace(/(&lt;\/?)([a-zA-Z_:][-a-zA-Z0-9_:.]*)(?=[\s&gt;\/])/g, (_m, p1, p2) => `${p1}<span style="color:#569cd6;">${p2}</span>`);
    return html;
  };

  // 当 history 切换时同步编辑器值
  useEffect(() => {
    const code = svgHistory[currentSvgIndex] || '';
    if (editorTextareaRef.current) {
      editorTextareaRef.current.value = code;
    }
    // 更新临时代码
    setTempSvgCode(code);
    setHasUnsavedChanges(false);
  }, [svgHistory, currentSvgIndex]);

  // 当切换到 split view 时，初始化临时代码
  useEffect(() => {
    if (viewMode === 'split') {
      setTempSvgCode(svgHistory[currentSvgIndex] || '');
      setHasUnsavedChanges(false);
    }
  }, [viewMode]);

  const fitObjectToCanvas = (canvas: fabric.Canvas) => {
    try {
      const objects = canvas.getObjects();
      if (!objects || objects.length === 0) return;
      const obj = objects[0] as any;
      // Reset scale before fitting
      try {
        obj.scale(1);
      } catch {}
      const canvasWidth = canvas.getWidth();
      const canvasHeight = canvas.getHeight();
      if (!canvasWidth || !canvasHeight) return;
      const paddingRatio = 0.85;
      let scale = Math.min(
        (canvasWidth * paddingRatio) / (obj.width || obj.getScaledWidth?.() || 1),
        (canvasHeight * paddingRatio) / (obj.height || obj.getScaledHeight?.() || 1)
      );
      if (!isFinite(scale) || scale <= 0) scale = 1;
      try {
        obj.scale(scale);
      } catch {}
      canvas.centerObject(obj);
      canvas.renderAll();
    } catch (error) {
      console.error('Error fitting object to canvas:', error);
    }
  };

  // 安全加载 SVG 到 canvas
  const loadSvgToCanvas = (canvas: fabric.Canvas, svgCode: string, onError?: (error: string) => void) => {
    if (!svgCode || !svgCode.trim()) {
      canvas.clear();
      canvas.renderAll();
      return;
    }

    try {
      fabric.loadSVGFromString(svgCode, (objects, options) => {
        try {
          if (!objects || objects.length === 0) {
            const errorMsg = 'Invalid SVG: No objects found';
            console.warn(errorMsg);
            if (onError) onError(errorMsg);
            return;
          }
          
          canvas.clear();
          const obj = fabric.util.groupSVGElements(objects, options);
          canvas.add(obj);
          fitObjectToCanvas(canvas);
          canvas.renderAll();
          
          // 清除错误
          if (svgError) setSvgError(null);
        } catch (error) {
          const errorMsg = `Error processing SVG: ${error instanceof Error ? error.message : 'Unknown error'}`;
          console.error(errorMsg);
          if (onError) onError(errorMsg);
          if (!svgError) setSvgError(errorMsg);
        }
      });
    } catch (error) {
      const errorMsg = `Error loading SVG: ${error instanceof Error ? error.message : 'Unknown error'}`;
      console.error(errorMsg);
      if (onError) onError(errorMsg);
      if (!svgError) setSvgError(errorMsg);
    }
  };

  const convertSvgToPng = async (svg: string) => {
    if (!svg) return;
    try {
      const svgBlob = new Blob([svg], { type: 'image/svg+xml;charset=utf-8' });
      const url = URL.createObjectURL(svgBlob);

      const img = new Image();
      img.crossOrigin = 'anonymous';
      const loaded: Promise<void> = new Promise((resolve, reject) => {
        img.onload = () => resolve();
        img.onerror = (e) => reject(e);
      });
      img.src = url;
      await loaded;

      const canvasEl = document.createElement('canvas');
      canvasEl.width = img.naturalWidth || 800;
      canvasEl.height = img.naturalHeight || 600;
      const ctx = canvasEl.getContext('2d');
      if (!ctx) throw new Error('Failed to get canvas context');
      
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(0, 0, canvasEl.width, canvasEl.height);
      ctx.drawImage(img, 0, 0);

      const dataUrl = canvasEl.toDataURL('image/png');
      setPngDataUrl(dataUrl);
      URL.revokeObjectURL(url);
    } catch (e) {
      console.error('Client-side SVG->PNG conversion failed', e);
      setPngDataUrl(null);
    }
  };

  // 更新预览 canvas（仅在 split view 中使用）
  const updatePreviewCanvas = (code: string) => {
    const canvas = previewFabricCanvasRef.current;
    if (canvas) {
      loadSvgToCanvas(canvas, code);
    }
  };

  // 处理代码编辑（在 split view 中）
  const handleSplitCodeChange = (newCode: string) => {
    setTempSvgCode(newCode);
    setHasUnsavedChanges(true);
    
    // 实时更新预览
    if (debounceTimerRef.current) {
      window.clearTimeout(debounceTimerRef.current);
    }
    debounceTimerRef.current = window.setTimeout(() => {
      updatePreviewCanvas(newCode);
    }, 150);
  };

  // 应用更改到主 SVG
  const applyChanges = () => {
    updateSvg(currentSvgIndex, tempSvgCode);
    setHasUnsavedChanges(false);
  };

  // 处理代码编辑（在 source view 中）
  const handleCodeChange = (newCode: string) => {
    // 更新 store 中的 SVG
    updateSvg(currentSvgIndex, newCode);
    
    // 重新加载到 canvas（实时预览）
    const canvas = fabricCanvasRef.current;
    if (canvas) {
      loadSvgToCanvas(canvas, newCode);
    }
  };

  const handleCodeInputDebounced = (newCode: string) => {
    if (debounceTimerRef.current) {
      window.clearTimeout(debounceTimerRef.current);
    }
    debounceTimerRef.current = window.setTimeout(() => {
      handleCodeChange(newCode);
    }, 150);
  };

  const syncEditorHighlight = () => {
    const code = viewMode === 'split' ? tempSvgCode : (svgHistory[currentSvgIndex] || '');
    if (editorHighlightRef.current) {
      editorHighlightRef.current.innerHTML = highlightSvg(code) + '\n';
    }
  };

  useEffect(() => {
    syncEditorHighlight();
  }, [svgHistory, currentSvgIndex, codeTheme, codeFontSize, viewMode, tempSvgCode]);

  // 分割器拖动处理
  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!isDraggingSplitter) return;
      
      const container = document.querySelector('.split-container');
      if (!container) return;

      const rect = container.getBoundingClientRect();
      let newRatio;
      
      if (splitOrientation === 'horizontal') {
        newRatio = ((e.clientX - rect.left) / rect.width) * 100;
      } else {
        newRatio = ((e.clientY - rect.top) / rect.height) * 100;
      }
      
      // 限制在 10% 到 90% 之间
      newRatio = Math.max(10, Math.min(90, newRatio));
      setSplitRatio(newRatio);
    };

    const handleMouseUp = () => {
      setIsDraggingSplitter(false);
    };

    if (isDraggingSplitter) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
      document.body.style.cursor = splitOrientation === 'horizontal' ? 'col-resize' : 'row-resize';
      document.body.style.userSelect = 'none';
    }

    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    };
  }, [isDraggingSplitter, splitOrientation]);

  // Fabric.js 画布初始化
  useEffect(() => {
    // 如果 canvas 已经存在，直接返回
    if (fabricCanvasRef.current) {
      return;
    }

    if (!canvasRef.current) {
      return;
    }

    // 延迟初始化以确保 DOM 已经渲染完成
    const timer = setTimeout(() => {
      if (!canvasRef.current || fabricCanvasRef.current) return;
      
      const parent = canvasRef.current.parentElement;
      if (!parent) return;
      
      const width = parent.clientWidth || 800;
      const height = parent.clientHeight || 600;
      
      const canvas = new fabric.Canvas(canvasRef.current, {
        width: width,
        height: height,
        backgroundColor: '#f0f0f0',
      });
      fabricCanvasRef.current = canvas;

      const resizeObserver = new ResizeObserver(entries => {
        for (let entry of entries) {
          const { width, height } = entry.contentRect;
          if (width > 0 && height > 0) {
            canvas.setWidth(width);
            canvas.setHeight(height);
            // 保持画布内容居中显示
            fitObjectToCanvas(canvas);
          }
        }
      });
      resizeObserverRef.current = resizeObserver;
      
      if (parent) {
        resizeObserver.observe(parent);
      }

      // Load current SVG into the newly created canvas (so switching views repaints)
      const currentSvg = svgHistory[currentSvgIndex];
      if (currentSvg) {
        loadSvgToCanvas(canvas, currentSvg);
      }

      // Pan functionality
      canvas.on('mouse:down', function (this: fabric.Canvas, opt: any) {
        if (spacePressedRef.current) {
          const self = this as any;
          self.isDragging = true;
          self.selection = false;
          self.lastPosX = (opt.e as MouseEvent).clientX;
          self.lastPosY = (opt.e as MouseEvent).clientY;
        }
      });
      canvas.on('mouse:move', function (this: fabric.Canvas, opt: any) {
        const self = this as any;
        if (self.isDragging) {
          const e = opt.e as MouseEvent;
          if (self.viewportTransform) {
            self.viewportTransform[4] += e.clientX - self.lastPosX;
            self.viewportTransform[5] += e.clientY - self.lastPosY;
            self.requestRenderAll();
          }
          self.lastPosX = e.clientX;
          self.lastPosY = e.clientY;
        }
      });
      canvas.on('mouse:up', function (this: fabric.Canvas) {
        const self = this as any;
        self.setViewportTransform(self.viewportTransform || [1, 0, 0, 1, 0, 0]);
        self.isDragging = false;
        self.selection = true;
      });
      
      // Zoom functionality
      canvas.on('mouse:wheel', function(this: fabric.Canvas, opt: any) {
        const delta = (opt.e as WheelEvent).deltaY;
        let zoom = this.getZoom();
        zoom *= Math.pow(0.999, delta);
        if (zoom > 20) zoom = 20;
        if (zoom < 0.01) zoom = 0.01;
        const point = new (fabric as any).Point((opt.e as WheelEvent).offsetX, (opt.e as WheelEvent).offsetY);
        this.zoomToPoint(point, zoom);
        opt.e.preventDefault();
        opt.e.stopPropagation();
      });

      // Selection box
      let selectionRect: fabric.Rect | null = null;
      let isDrawingSelection = false;
      let startX = 0, startY = 0;

      canvas.on('mouse:down', function(this: fabric.Canvas, o: any) {
        if (!isSpacePressed && (o.target == null || o.target === undefined)) {
            isDrawingSelection = true;
            const pointer = this.getPointer(o.e);
            startX = pointer.x;
            startY = pointer.y;

            selectionRect = new fabric.Rect({
                left: startX,
                top: startY,
                width: 0,
                height: 0,
                fill: 'rgba(0,102,255,0.2)',
                stroke: 'rgba(0,102,255,0.8)',
                strokeWidth: 1,
                selectable: false
            });
            this.add(selectionRect);
        }
      });

      canvas.on('mouse:move', function(this: fabric.Canvas, o: any) {
        if (isDrawingSelection && selectionRect) {
            const pointer = this.getPointer(o.e);
            const width = pointer.x - startX;
            const height = pointer.y - startY;

            selectionRect.set({
                left: width > 0 ? startX : pointer.x,
                top: height > 0 ? startY : pointer.y,
                width: Math.abs(width),
                height: Math.abs(height)
            });
            this.renderAll();
        }
      });

      canvas.on('mouse:up', function(this: fabric.Canvas) {
        if (isDrawingSelection && selectionRect) {
            setSelectionBox({
                x: selectionRect.left!,
                y: selectionRect.top!,
                width: selectionRect.width!,
                height: selectionRect.height!
            });
        }
        isDrawingSelection = false;
      });
    }, 50); // 50ms 延迟以确保 DOM 完成渲染

    return () => {
      clearTimeout(timer);
      if (resizeObserverRef.current) {
        resizeObserverRef.current.disconnect();
        resizeObserverRef.current = null;
      }
      const canvas = fabricCanvasRef.current;
      if (canvas) {
        canvas.off();
        canvas.clear();
        canvas.dispose();
        fabricCanvasRef.current = null;
      }
    };
  }, []);

  // Load SVG from state
  useEffect(() => {
    const canvas = fabricCanvasRef.current;
    console.log('🖼️ Canvas: SVG history changed. History length:', svgHistory.length, 'Current index:', currentSvgIndex);
    if (canvas) {
      const svg = svgHistory[currentSvgIndex];
      if (svg) {
        console.log('🖼️ Canvas: Loading SVG, length:', svg.length);
        loadSvgToCanvas(canvas, svg);
      } else {
        console.log('🖼️ Canvas: No SVG at current index, clearing canvas');
        canvas.clear();
        canvas.renderAll();
      }
    } else {
      console.warn('🖼️ Canvas: Fabric canvas not initialized');
    }
  }, [svgHistory, currentSvgIndex]);

  // 当视图模式改变时，重新渲染 canvas
  useEffect(() => {
    const canvas = fabricCanvasRef.current;
    if (!canvas) return;
    
    // 只在切换到 canvas 或 从其他模式回来时重新加载
    if (viewMode === 'canvas') {
      // 延迟一下以确保布局完成
      const timer = setTimeout(() => {
        const parent = canvasRef.current?.parentElement;
        if (parent && canvas) {
          const width = parent.clientWidth;
          const height = parent.clientHeight;
          if (width > 0 && height > 0) {
            canvas.setWidth(width);
            canvas.setHeight(height);
            
            // 强制重新加载当前 SVG
            const currentSvg = svgHistory[currentSvgIndex];
            if (currentSvg) {
              loadSvgToCanvas(canvas, currentSvg);
            } else {
              canvas.clear();
              canvas.renderAll();
            }
          }
        }
      }, 150); // 增加延迟确保布局完成
      
      return () => clearTimeout(timer);
    }
  }, [viewMode]);

  // 初始化预览 canvas（用于 split view）
  useEffect(() => {
    if (viewMode !== 'split') {
      // 清理预览 canvas
      if (previewFabricCanvasRef.current) {
        previewFabricCanvasRef.current.dispose();
        previewFabricCanvasRef.current = null;
      }
      return;
    }

    if (previewCanvasRef.current && !previewFabricCanvasRef.current) {
      const timer = setTimeout(() => {
        if (!previewCanvasRef.current || previewFabricCanvasRef.current) return;
        
        const parent = previewCanvasRef.current.parentElement;
        if (!parent) return;
        
        const width = parent.clientWidth || 800;
        const height = parent.clientHeight || 600;
        
        const canvas = new fabric.Canvas(previewCanvasRef.current, {
          width: width,
          height: height,
          backgroundColor: '#f0f0f0',
        });
        previewFabricCanvasRef.current = canvas;

        const resizeObserver = new ResizeObserver(entries => {
          for (let entry of entries) {
            const { width, height } = entry.contentRect;
            if (width > 0 && height > 0) {
              canvas.setWidth(width);
              canvas.setHeight(height);
              fitObjectToCanvas(canvas);
            }
          }
        });
        
        if (parent) {
          resizeObserver.observe(parent);
        }

        // 加载当前 SVG
        const currentSvg = tempSvgCode || svgHistory[currentSvgIndex];
        if (currentSvg) {
          updatePreviewCanvas(currentSvg);
        }
      }, 50);

      return () => clearTimeout(timer);
    }
  }, [viewMode, tempSvgCode, svgHistory, currentSvgIndex]);

  // Handle object modification and update history
  useEffect(() => {
    const canvas = fabricCanvasRef.current;
    if (canvas) {
      const saveState = () => {
        const newSvg = canvas.toSVG();
        addSvg(newSvg);
      };
      canvas.on('object:modified', saveState);
      return () => {
        canvas.off('object:modified', saveState);
      };
    }
  }, [addSvg]);

  // Keyboard listeners for space key (panning)
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.code === 'Space') {
        setIsSpacePressed(true);
        spacePressedRef.current = true;
        if (fabricCanvasRef.current) {
          fabricCanvasRef.current.defaultCursor = 'grab';
          fabricCanvasRef.current.selection = false;
        }
      }
    };
    const handleKeyUp = (e: KeyboardEvent) => {
      if (e.code === 'Space') {
        setIsSpacePressed(false);
        spacePressedRef.current = false;
         if (fabricCanvasRef.current) {
          fabricCanvasRef.current.defaultCursor = 'default';
          fabricCanvasRef.current.selection = true;
        }
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
    };
  }, []);

  // 渲染代码编辑器（带语法高亮与实时同步）
  const renderCodeEditor = (isSplitView: boolean = false) => {
    const code = isSplitView ? tempSvgCode : (svgHistory[currentSvgIndex] || '');
    const handleChange = isSplitView ? handleSplitCodeChange : handleCodeInputDebounced;
    const handleBlur = isSplitView ? () => {} : handleCodeChange;

    return (
      <div 
        className="h-full relative flex flex-col"
        style={{
          backgroundColor: codeThemes[codeTheme].background,
          color: codeThemes[codeTheme].foreground,
          fontFamily: codeFonts[0],
          fontSize: `${codeFontSize}px`,
          boxShadow: 'inset 0 2px 8px rgba(0, 0, 0, 0.1)',
          borderRight: '1px solid rgba(0, 0, 0, 0.1)'
        }}
      >
        {/* 代码编辑器标题栏 */}
        <div style={{
          padding: '12px 16px',
          background: codeTheme === 'dark' ? 'rgba(0, 0, 0, 0.2)' : 'rgba(0, 0, 0, 0.05)',
          borderBottom: `1px solid ${codeTheme === 'dark' ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)'}`,
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
          fontSize: '13px',
          fontWeight: 600,
          color: codeTheme === 'dark' ? 'rgba(255, 255, 255, 0.8)' : 'rgba(0, 0, 0, 0.7)',
          flexShrink: 0
        }}>
          <span>📄</span>
          <span>SVG Source Code</span>
          {isSplitView && hasUnsavedChanges && (
            <div style={{
              fontSize: '11px',
              padding: '4px 8px',
              background: 'rgba(251, 146, 60, 0.2)',
              color: '#ea580c',
              borderRadius: '6px',
              fontWeight: 600
            }}>
              UNSAVED
            </div>
          )}
          <div style={{
            marginLeft: 'auto',
            display: 'flex',
            gap: '6px'
          }}>
            <div style={{
              width: '12px',
              height: '12px',
              borderRadius: '50%',
              background: '#ef4444'
            }}></div>
            <div style={{
              width: '12px',
              height: '12px',
              borderRadius: '50%',
              background: '#f59e0b'
            }}></div>
            <div style={{
              width: '12px',
              height: '12px',
              borderRadius: '50%',
              background: '#10b981'
            }}></div>
          </div>
        </div>

        {/* Apply Changes 按钮（仅在 split view 且有未保存更改时显示） */}
        {isSplitView && hasUnsavedChanges && (
          <div style={{
            padding: '12px 16px',
            background: codeTheme === 'dark' ? 'rgba(59, 130, 246, 0.1)' : 'rgba(59, 130, 246, 0.05)',
            borderBottom: `1px solid ${codeTheme === 'dark' ? 'rgba(59, 130, 246, 0.3)' : 'rgba(59, 130, 246, 0.2)'}`,
            display: 'flex',
            alignItems: 'center',
            gap: '12px',
            flexShrink: 0
          }}>
            <button
              onClick={applyChanges}
              style={{
                padding: '8px 16px',
                borderRadius: '8px',
                fontWeight: 600,
                fontSize: '13px',
                border: 'none',
                cursor: 'pointer',
                transition: 'all 0.2s',
                background: 'linear-gradient(135deg, #3b82f6, #8b5cf6)',
                color: 'white',
                boxShadow: '0 2px 8px rgba(59, 130, 246, 0.3)'
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.transform = 'translateY(-2px)';
                e.currentTarget.style.boxShadow = '0 4px 12px rgba(59, 130, 246, 0.4)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.transform = 'translateY(0)';
                e.currentTarget.style.boxShadow = '0 2px 8px rgba(59, 130, 246, 0.3)';
              }}
            >
              ✓ Apply Changes
            </button>
            <span style={{
              fontSize: '12px',
              color: codeTheme === 'dark' ? 'rgba(255, 255, 255, 0.6)' : 'rgba(0, 0, 0, 0.6)'
            }}>
              Save changes to main canvas
            </span>
          </div>
        )}

        {/* 编辑器主体 */}
        <div style={{ flex: 1, position: 'relative', overflow: 'hidden' }}>
          {/* 高亮层 */}
          <pre
            ref={editorHighlightRef}
            className="absolute inset-0 overflow-auto p-4 select-none"
            style={{ 
              lineHeight: '1.6',
              margin: 0,
              fontFamily: codeFonts[0],
              fontSize: `${codeFontSize}px`,
              whiteSpace: 'pre-wrap',
              wordWrap: 'break-word'
            }}
          />
          {/* 输入层 */}
          <textarea
            ref={editorTextareaRef}
            value={code}
            onChange={(e) => {
              // 即时高亮同步
              if (editorHighlightRef.current) {
                editorHighlightRef.current.innerHTML = highlightSvg(e.target.value) + '\n';
              }
              handleChange(e.target.value);
            }}
            onBlur={(e) => handleBlur(e.target.value)}
            onScroll={(e) => {
              if (editorHighlightRef.current) {
                editorHighlightRef.current.scrollTop = (e.target as HTMLTextAreaElement).scrollTop;
                editorHighlightRef.current.scrollLeft = (e.target as HTMLTextAreaElement).scrollLeft;
              }
            }}
            className="absolute inset-0 w-full h-full p-4 bg-transparent outline-none resize-none"
            style={{
              color: 'transparent',
              caretColor: codeThemes[codeTheme].foreground,
              lineHeight: '1.6',
              whiteSpace: 'pre-wrap',
              wordWrap: 'break-word',
              overflow: 'auto',
              fontFamily: codeFonts[0],
              fontSize: `${codeFontSize}px`
            }}
            spellCheck={false}
          />
        </div>
      </div>
    );
  };

  // 渲染预览区域
  const renderPreview = () => {
    switch (viewMode) {
      case 'png':
        return (
          <div className="w-full h-full flex items-center justify-center p-8" style={{
            background: 'linear-gradient(135deg, #f9fafb, #ffffff)',
            overflow: 'auto'
          }}>
            {pngDataUrl ? (
              <div style={{
                maxWidth: '100%',
                maxHeight: '100%',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center'
              }}>
                <img src={pngDataUrl} alt="Converted PNG" style={{
                  display: 'block',
                  maxWidth: '100%',
                  maxHeight: '100%',
                  objectFit: 'contain',
                  boxShadow: '0 10px 40px rgba(0, 0, 0, 0.15)',
                  borderRadius: '12px',
                  background: 'white'
                }} />
              </div>
            ) : (
              <div style={{
                textAlign: 'center',
                color: '#6b7280',
                fontSize: '16px',
                fontWeight: 500
              }}>
                <div style={{
                  width: '60px',
                  height: '60px',
                  margin: '0 auto 16px',
                  border: '4px solid #e5e7eb',
                  borderTopColor: '#3b82f6',
                  borderRadius: '50%',
                  animation: 'spin 1s linear infinite'
                }}></div>
                Converting to PNG…
              </div>
            )}
          </div>
        );
      case 'source':
        return renderCodeEditor();
      default:
        return null;
    }
  };

  // 渲染分屏视图
  const renderSplitView = (canvasContainer: React.ReactNode) => {
    const isHorizontal = splitOrientation === 'horizontal';
    
    return (
      <div className="split-container w-full h-full flex relative" style={{
        flexDirection: isHorizontal ? 'row' : 'column',
        overflow: 'hidden'
      }}>
        {/* 代码编辑器区域 */}
        <div style={{
          width: isHorizontal ? `${splitRatio}%` : '100%',
          height: isHorizontal ? '100%' : `${splitRatio}%`,
          flex: `0 0 ${isHorizontal ? splitRatio : splitRatio}%`,
          overflow: 'hidden',
          position: 'relative'
        }}>
          {renderCodeEditor(true)}
        </div>
        
        {/* 分割器 */}
        <div
          style={{
            width: isHorizontal ? '8px' : '100%',
            height: isHorizontal ? '100%' : '8px',
            background: 'linear-gradient(135deg, rgba(59, 130, 246, 0.3), rgba(139, 92, 246, 0.3))',
            cursor: isHorizontal ? 'col-resize' : 'row-resize',
            position: 'relative',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            flexShrink: 0,
            transition: 'background 0.2s',
            zIndex: 10
          }}
          onMouseDown={(e) => {
            e.preventDefault();
            e.stopPropagation();
            setIsDraggingSplitter(true);
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.background = 'linear-gradient(135deg, rgba(59, 130, 246, 0.5), rgba(139, 92, 246, 0.5))';
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.background = 'linear-gradient(135deg, rgba(59, 130, 246, 0.3), rgba(139, 92, 246, 0.3))';
          }}
        >
          <div style={{
            width: isHorizontal ? '2px' : '20px',
            height: isHorizontal ? '20px' : '2px',
            background: 'rgba(255, 255, 255, 0.6)',
            borderRadius: '2px'
          }}></div>
        </div>
        
        {/* 预览区域 */}
        <div style={{
          flex: '1 1 auto',
          overflow: 'hidden',
          position: 'relative'
        }}>
          {canvasContainer}
        </div>
      </div>
    );
  };

  return (
    <div style={{
      position: 'fixed',
      top: '16px',
      left: '16px',
      bottom: '16px',
      right: '452px', // 420px (chat width) + 16px (margin) + 16px (gap)
      backgroundColor: 'rgba(255, 255, 255, 0.95)',
      backdropFilter: 'blur(20px)',
      borderRadius: '24px',
      boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.25), 0 0 0 1px rgba(255, 255, 255, 0.2)',
      overflow: 'hidden',
      transition: 'all 0.5s ease',
      display: 'flex',
      flexDirection: 'column'
    }}>
      {/* 装饰性背景动画 */}
      <div style={{
        position: 'absolute',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        overflow: 'hidden',
        borderRadius: '24px',
        pointerEvents: 'none',
        zIndex: 0
      }}>
        <div style={{
          position: 'absolute',
          top: '-25%',
          right: '-25%',
          width: '300px',
          height: '300px',
          background: 'linear-gradient(135deg, rgba(59, 130, 246, 0.1), rgba(139, 92, 246, 0.1))',
          borderRadius: '50%',
          filter: 'blur(60px)',
          animation: 'pulse 3s infinite'
        }}></div>
        <div style={{
          position: 'absolute',
          bottom: '-25%',
          left: '-25%',
          width: '300px',
          height: '300px',
          background: 'linear-gradient(45deg, rgba(168, 85, 247, 0.1), rgba(236, 72, 153, 0.1))',
          borderRadius: '50%',
          filter: 'blur(60px)',
          animation: 'pulse 3s infinite',
          animationDelay: '1.5s'
        }}></div>
      </div>

      {/* 视图模式切换和设置工具栏 */}
      <div className="relative z-10" style={{
        padding: '16px 20px',
        background: 'linear-gradient(135deg, #3b82f6, #8b5cf6)',
        backdropFilter: 'blur(12px)',
        borderBottom: '1px solid rgba(255, 255, 255, 0.2)',
        boxShadow: '0 4px 12px rgba(0, 0, 0, 0.1)'
      }}>
        <div className="flex items-center gap-3 flex-wrap">
          {/* 视图模式按钮组 */}
          <div className="flex items-center gap-2">
            <button 
              onClick={() => setViewMode('canvas')} 
              style={{
                padding: '8px 16px',
                borderRadius: '12px',
                fontWeight: 600,
                fontSize: '14px',
                border: 'none',
                cursor: 'pointer',
                transition: 'all 0.3s',
                boxShadow: viewMode === 'canvas' ? '0 4px 12px rgba(0, 0, 0, 0.2)' : 'none',
                ...(viewMode === 'canvas' ? {
                  background: 'linear-gradient(135deg, #ffffff, #f0f9ff)',
                  color: '#1e40af',
                  transform: 'translateY(-2px)'
                } : {
                  background: 'rgba(255, 255, 255, 0.2)',
                  color: 'white',
                  backdropFilter: 'blur(8px)'
                })
              }}
              onMouseEnter={(e) => {
                if (viewMode !== 'canvas') e.currentTarget.style.background = 'rgba(255, 255, 255, 0.3)';
              }}
              onMouseLeave={(e) => {
                if (viewMode !== 'canvas') e.currentTarget.style.background = 'rgba(255, 255, 255, 0.2)';
              }}
            >
              🎨 Canvas
            </button>
            <button 
              onClick={() => { setViewMode('png'); convertSvgToPng(svgHistory[currentSvgIndex]); }} 
              style={{
                padding: '8px 16px',
                borderRadius: '12px',
                fontWeight: 600,
                fontSize: '14px',
                border: 'none',
                cursor: 'pointer',
                transition: 'all 0.3s',
                boxShadow: viewMode === 'png' ? '0 4px 12px rgba(0, 0, 0, 0.2)' : 'none',
                ...(viewMode === 'png' ? {
                  background: 'linear-gradient(135deg, #ffffff, #f0f9ff)',
                  color: '#1e40af',
                  transform: 'translateY(-2px)'
                } : {
                  background: 'rgba(255, 255, 255, 0.2)',
                  color: 'white',
                  backdropFilter: 'blur(8px)'
                })
              }}
              onMouseEnter={(e) => {
                if (viewMode !== 'png') e.currentTarget.style.background = 'rgba(255, 255, 255, 0.3)';
              }}
              onMouseLeave={(e) => {
                if (viewMode !== 'png') e.currentTarget.style.background = 'rgba(255, 255, 255, 0.2)';
              }}
            >
              🖼️ PNG
            </button>
            <button 
              onClick={() => setViewMode('source')} 
              style={{
                padding: '8px 16px',
                borderRadius: '12px',
                fontWeight: 600,
                fontSize: '14px',
                border: 'none',
                cursor: 'pointer',
                transition: 'all 0.3s',
                boxShadow: viewMode === 'source' ? '0 4px 12px rgba(0, 0, 0, 0.2)' : 'none',
                ...(viewMode === 'source' ? {
                  background: 'linear-gradient(135deg, #ffffff, #f0f9ff)',
                  color: '#1e40af',
                  transform: 'translateY(-2px)'
                } : {
                  background: 'rgba(255, 255, 255, 0.2)',
                  color: 'white',
                  backdropFilter: 'blur(8px)'
                })
              }}
              onMouseEnter={(e) => {
                if (viewMode !== 'source') e.currentTarget.style.background = 'rgba(255, 255, 255, 0.3)';
              }}
              onMouseLeave={(e) => {
                if (viewMode !== 'source') e.currentTarget.style.background = 'rgba(255, 255, 255, 0.2)';
              }}
            >
              📝 Source
            </button>
            <button 
              onClick={() => setViewMode('split')} 
              style={{
                padding: '8px 16px',
                borderRadius: '12px',
                fontWeight: 600,
                fontSize: '14px',
                border: 'none',
                cursor: 'pointer',
                transition: 'all 0.3s',
                boxShadow: viewMode === 'split' ? '0 4px 12px rgba(0, 0, 0, 0.2)' : 'none',
                ...(viewMode === 'split' ? {
                  background: 'linear-gradient(135deg, #ffffff, #f0f9ff)',
                  color: '#1e40af',
                  transform: 'translateY(-2px)'
                } : {
                  background: 'rgba(255, 255, 255, 0.2)',
                  color: 'white',
                  backdropFilter: 'blur(8px)'
                })
              }}
              onMouseEnter={(e) => {
                if (viewMode !== 'split') e.currentTarget.style.background = 'rgba(255, 255, 255, 0.3)';
              }}
              onMouseLeave={(e) => {
                if (viewMode !== 'split') e.currentTarget.style.background = 'rgba(255, 255, 255, 0.2)';
              }}
            >
              ⚡ Split View
            </button>
          </div>

          {/* 下载按钮组 */}
          <div className="flex items-center gap-2 ml-auto">
            <button
              onClick={async () => {
                setDownloading('svg');
                const svg = editorTextareaRef.current?.value ?? (svgHistory[currentSvgIndex] || '');
                const blob = new Blob([svg], { type: 'image/svg+xml;charset=utf-8' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `canvas_${Date.now()}.svg`;
                a.click();
                URL.revokeObjectURL(url);
                setDownloading(null);
              }}
              style={{
                padding: '8px 16px',
                borderRadius: '12px',
                fontWeight: 600,
                fontSize: '14px',
                border: 'none',
                cursor: 'pointer',
                transition: 'all 0.3s',
                background: downloading === 'svg' ? 'rgba(255, 255, 255, 0.4)' : 'rgba(255, 255, 255, 0.2)',
                color: 'white',
                backdropFilter: 'blur(8px)',
                boxShadow: '0 2px 8px rgba(0, 0, 0, 0.1)'
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.background = 'rgba(255, 255, 255, 0.3)';
                e.currentTarget.style.transform = 'translateY(-2px)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.background = downloading === 'svg' ? 'rgba(255, 255, 255, 0.4)' : 'rgba(255, 255, 255, 0.2)';
                e.currentTarget.style.transform = 'translateY(0)';
              }}
            >
              ⬇️ SVG
            </button>
            <button
              onClick={async () => {
                setDownloading('png');
                try {
                  const svg = editorTextareaRef.current?.value ?? (svgHistory[currentSvgIndex] || '');
                  const svgBlob = new Blob([svg], { type: 'image/svg+xml;charset=utf-8' });
                  const url = URL.createObjectURL(svgBlob);
                  const img = new Image();
                  img.crossOrigin = 'anonymous';
                  await new Promise<void>((resolve, reject) => {
                    img.onload = () => resolve();
                    img.onerror = (e) => reject(e);
                    img.src = url;
                  });
                  const canvasEl = document.createElement('canvas');
                  canvasEl.width = img.naturalWidth || 800;
                  canvasEl.height = img.naturalHeight || 600;
                  const ctx = canvasEl.getContext('2d');
                  if (ctx) {
                    ctx.fillStyle = '#ffffff';
                    ctx.fillRect(0, 0, canvasEl.width, canvasEl.height);
                    ctx.drawImage(img, 0, 0);
                    const dataUrl = canvasEl.toDataURL('image/png');
                    const a = document.createElement('a');
                    a.href = dataUrl;
                    a.download = `canvas_${Date.now()}.png`;
                    a.click();
                  }
                  URL.revokeObjectURL(url);
                } catch (e) {
                  console.error('PNG download failed', e);
                } finally {
                  setDownloading(null);
                }
              }}
              style={{
                padding: '8px 16px',
                borderRadius: '12px',
                fontWeight: 600,
                fontSize: '14px',
                border: 'none',
                cursor: 'pointer',
                transition: 'all 0.3s',
                background: downloading === 'png' ? 'rgba(255, 255, 255, 0.4)' : 'rgba(255, 255, 255, 0.2)',
                color: 'white',
                backdropFilter: 'blur(8px)',
                boxShadow: '0 2px 8px rgba(0, 0, 0, 0.1)'
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.background = 'rgba(255, 255, 255, 0.3)';
                e.currentTarget.style.transform = 'translateY(-2px)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.background = downloading === 'png' ? 'rgba(255, 255, 255, 0.4)' : 'rgba(255, 255, 255, 0.2)';
                e.currentTarget.style.transform = 'translateY(0)';
              }}
            >
              ⬇️ PNG
            </button>
          </div>
        </div>

        {/* 设置行（分屏和代码编辑器设置） */}
        {(viewMode === 'split' || viewMode === 'source') && (
          <div className="flex items-center gap-4 mt-3 pt-3 border-t border-white/20">
            {viewMode === 'split' && (
              <div className="flex items-center gap-2">
                <label style={{ fontSize: '13px', color: 'white', fontWeight: 500 }}>Orientation:</label>
                <select 
                  value={splitOrientation}
                  onChange={(e) => setSplitOrientation(e.target.value as 'horizontal' | 'vertical')}
                  style={{
                    fontSize: '13px',
                    padding: '6px 12px',
                    borderRadius: '8px',
                    border: '1px solid rgba(255, 255, 255, 0.3)',
                    background: 'rgba(255, 255, 255, 0.2)',
                    color: 'white',
                    backdropFilter: 'blur(8px)',
                    cursor: 'pointer',
                    fontWeight: 500
                  }}
                >
                  <option value="horizontal" style={{ background: '#1f2937', color: 'white' }}>↔️ Horizontal</option>
                  <option value="vertical" style={{ background: '#1f2937', color: 'white' }}>↕️ Vertical</option>
                </select>
              </div>
            )}
            
            <div className="flex items-center gap-2">
              <label style={{ fontSize: '13px', color: 'white', fontWeight: 500 }}>Theme:</label>
              <select 
                value={codeTheme}
                onChange={(e) => setCodeTheme(e.target.value as 'dark' | 'light')}
                style={{
                  fontSize: '13px',
                  padding: '6px 12px',
                  borderRadius: '8px',
                  border: '1px solid rgba(255, 255, 255, 0.3)',
                  background: 'rgba(255, 255, 255, 0.2)',
                  color: 'white',
                  backdropFilter: 'blur(8px)',
                  cursor: 'pointer',
                  fontWeight: 500
                }}
              >
                <option value="dark" style={{ background: '#1f2937', color: 'white' }}>🌙 Dark</option>
                <option value="light" style={{ background: '#1f2937', color: 'white' }}>☀️ Light</option>
              </select>
            </div>
            
            <div className="flex items-center gap-2">
              <label style={{ fontSize: '13px', color: 'white', fontWeight: 500 }}>Font Size:</label>
              <select 
                value={codeFontSize}
                onChange={(e) => setCodeFontSize(Number(e.target.value))}
                style={{
                  fontSize: '13px',
                  padding: '6px 12px',
                  borderRadius: '8px',
                  border: '1px solid rgba(255, 255, 255, 0.3)',
                  background: 'rgba(255, 255, 255, 0.2)',
                  color: 'white',
                  backdropFilter: 'blur(8px)',
                  cursor: 'pointer',
                  fontWeight: 500
                }}
              >
                <option value={12} style={{ background: '#1f2937', color: 'white' }}>12px</option>
                <option value={14} style={{ background: '#1f2937', color: 'white' }}>14px</option>
                <option value={16} style={{ background: '#1f2937', color: 'white' }}>16px</option>
                <option value={18} style={{ background: '#1f2937', color: 'white' }}>18px</option>
                <option value={20} style={{ background: '#1f2937', color: 'white' }}>20px</option>
              </select>
            </div>
          </div>
        )}
      </div>

      {/* 主内容区域 */}
      <div className="flex-1 relative overflow-hidden z-10">
        {viewMode === 'split' ? (
          renderSplitView(
            <div className="w-full h-full flex flex-col" style={{
              background: 'linear-gradient(135deg, #f9fafb, #ffffff)'
            }}>
              {/* 预览标题栏 */}
              <div style={{
                padding: '12px 16px',
                background: 'rgba(0, 0, 0, 0.05)',
                borderBottom: '1px solid rgba(0, 0, 0, 0.1)',
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                fontSize: '13px',
                fontWeight: 600,
                color: 'rgba(0, 0, 0, 0.7)'
              }}>
                <span>👁️</span>
                <span>Live Preview</span>
                <div style={{
                  marginLeft: 'auto',
                  fontSize: '11px',
                  padding: '4px 8px',
                  background: 'rgba(16, 185, 129, 0.1)',
                  color: '#059669',
                  borderRadius: '6px',
                  fontWeight: 600
                }}>
                  LIVE
                </div>
              </div>
              
              {/* Canvas 区域 - 使用独立的预览 canvas */}
              <div style={{
                flex: 1,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                overflow: 'hidden',
                position: 'relative',
                width: '100%',
                height: '100%'
              }}>
                <canvas ref={previewCanvasRef} style={{ display: 'block' }} />
              </div>
            </div>
          )
        ) : (
          <>
            {/* Canvas-only mode */}
            {viewMode === 'canvas' && (
              <div className="w-full h-full" style={{
                background: 'linear-gradient(135deg, #f9fafb, #ffffff)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                position: 'relative'
              }}>
                <canvas ref={canvasRef} style={{ display: 'block' }} />
                
                {/* 错误提示 */}
                {svgError && (
                  <div style={{
                    position: 'absolute',
                    top: '20px',
                    left: '50%',
                    transform: 'translateX(-50%)',
                    padding: '12px 24px',
                    background: 'rgba(239, 68, 68, 0.95)',
                    color: 'white',
                    borderRadius: '12px',
                    boxShadow: '0 4px 12px rgba(239, 68, 68, 0.4)',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '12px',
                    zIndex: 1000,
                    backdropFilter: 'blur(8px)',
                    maxWidth: '80%'
                  }}>
                    <span style={{ fontSize: '20px' }}>⚠️</span>
                    <div style={{ flex: 1 }}>
                      <div style={{ fontWeight: 600, marginBottom: '4px' }}>SVG Error</div>
                      <div style={{ fontSize: '13px', opacity: 0.9 }}>{svgError}</div>
                    </div>
                    <button
                      onClick={() => setSvgError(null)}
                      style={{
                        background: 'rgba(255, 255, 255, 0.2)',
                        border: 'none',
                        borderRadius: '6px',
                        padding: '4px 8px',
                        color: 'white',
                        cursor: 'pointer',
                        fontSize: '12px',
                        fontWeight: 600
                      }}
                      onMouseEnter={(e) => e.currentTarget.style.background = 'rgba(255, 255, 255, 0.3)'}
                      onMouseLeave={(e) => e.currentTarget.style.background = 'rgba(255, 255, 255, 0.2)'}
                    >
                      Dismiss
                    </button>
                  </div>
                )}
              </div>
            )}

            {/* Non-canvas previews (png/source) */}
            {(viewMode === 'png' || viewMode === 'source') ? renderPreview() : null}
          </>
        )}
      </div>

      {/* CSS 动画 */}
      <style>
        {`
          @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.6; }
          }
          @keyframes spin {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
          }
        `}
      </style>
    </div>
  );
};


export default CanvasComponent;