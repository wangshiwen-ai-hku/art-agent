import React, { useRef, useEffect, useState } from 'react';
import { fabric } from 'fabric';
import { useCanvasStore } from '../store/useCanvasStore';

// 定义视图模式类型
type ViewMode = 'canvas' | 'png' | 'source' | 'split';

const CanvasComponent: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const fabricCanvasRef = useRef<fabric.Canvas | null>(null);
  const codeEditorRef = useRef<HTMLPreElement>(null);
  const { svgHistory, currentSvgIndex, addSvg, setSelectionBox, updateSvg } = useCanvasStore();
  const [isSpacePressed, setIsSpacePressed] = useState(false);
  const [viewMode, setViewMode] = useState<ViewMode>('canvas');
  const [pngDataUrl, setPngDataUrl] = useState<string | null>(null);
  const [splitOrientation, setSplitOrientation] = useState<'horizontal' | 'vertical'>('horizontal');
  const [splitRatio, setSplitRatio] = useState(50); // 50% 分割比例
  const [isDraggingSplitter, setIsDraggingSplitter] = useState(false);
  const [codeTheme, setCodeTheme] = useState<'dark' | 'light'>('dark');
  const [codeFontSize, setCodeFontSize] = useState(14);

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

  // 处理代码编辑
  const handleCodeChange = (newCode: string) => {
    // 更新 store 中的 SVG
    updateSvg(currentSvgIndex, newCode);
    
    // 重新加载到 canvas
    const canvas = fabricCanvasRef.current;
    if (canvas && newCode) {
      fabric.loadSVGFromString(newCode, (objects, options) => {
        canvas.clear();
        const obj = fabric.util.groupSVGElements(objects, options);
        canvas.add(obj).renderAll();
        canvas.centerObject(obj);
        obj.scaleToWidth(canvas.getWidth() * 0.8);
        if (obj.getScaledHeight() > canvas.getHeight() * 0.8) {
          obj.scaleToHeight(canvas.getHeight() * 0.8);
        }
        canvas.renderAll();
      });
    }
  };

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
    if (canvasRef.current) {
      const canvas = new fabric.Canvas(canvasRef.current, {
        width: canvasRef.current.parentElement?.clientWidth,
        height: canvasRef.current.parentElement?.clientHeight,
        backgroundColor: '#f0f0f0',
      });
      fabricCanvasRef.current = canvas;

      const resizeObserver = new ResizeObserver(entries => {
        for (let entry of entries) {
          const { width, height } = entry.contentRect;
          (canvas as any).setWidth(width);
          (canvas as any).setHeight(height);
        }
      });
      resizeObserver.observe(canvasRef.current.parentElement!);

      // Load current SVG into the newly created canvas (so switching views repaints)
      const currentSvg = svgHistory[currentSvgIndex];
      if (currentSvg) {
        fabric.loadSVGFromString(currentSvg, (objects, options) => {
          canvas.clear();
          const obj = fabric.util.groupSVGElements(objects, options);
          canvas.add(obj);
          canvas.centerObject(obj);
          obj.scaleToWidth(canvas.getWidth() * 0.8);
          if (obj.getScaledHeight && obj.getScaledHeight() > canvas.getHeight() * 0.8) {
            obj.scaleToHeight(canvas.getHeight() * 0.8);
          }
          canvas.renderAll();
        });
      }

      // Pan functionality
      canvas.on('mouse:down', function (this: fabric.Canvas, opt: any) {
        if (isSpacePressed) {
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

      return () => {
        resizeObserver.disconnect();
        canvas.dispose();
        // clear ref so future inits won't attempt to use disposed instance
        fabricCanvasRef.current = null;
      };
    }
  }, [isSpacePressed, setSelectionBox, viewMode, svgHistory, currentSvgIndex]);

  // Load SVG from state
  useEffect(() => {
    const canvas = fabricCanvasRef.current;
    if (canvas && svgHistory[currentSvgIndex]) {
      const svg = svgHistory[currentSvgIndex];
      fabric.loadSVGFromString(svg, (objects, options) => {
        canvas.clear();
        const obj = fabric.util.groupSVGElements(objects, options);
        canvas.add(obj).renderAll();
        canvas.centerObject(obj);
        obj.scaleToWidth(canvas.getWidth() * 0.8);
        if (obj.getScaledHeight() > canvas.getHeight() * 0.8) {
          obj.scaleToHeight(canvas.getHeight() * 0.8);
        }
        canvas.renderAll();
      });
    } else if (canvas) {
      canvas.clear();
    }
  }, [svgHistory, currentSvgIndex]);

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
        if (fabricCanvasRef.current) {
          fabricCanvasRef.current.defaultCursor = 'grab';
          fabricCanvasRef.current.selection = false;
        }
      }
    };
    const handleKeyUp = (e: KeyboardEvent) => {
      if (e.code === 'Space') {
        setIsSpacePressed(false);
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

  // 渲染代码编辑器
  const renderCodeEditor = () => (
    <div 
      className="h-full overflow-auto relative"
      style={{
        backgroundColor: codeThemes[codeTheme].background,
        color: codeThemes[codeTheme].foreground,
        fontFamily: codeFonts[0],
        fontSize: `${codeFontSize}px`
      }}
    >
      <pre
        ref={codeEditorRef}
        contentEditable
        suppressContentEditableWarning
        className="whitespace-pre-wrap break-words outline-none p-4 min-h-full"
        onBlur={(e) => handleCodeChange(e.currentTarget.textContent || '')}
        onKeyDown={(e) => {
          if (e.key === 'Tab') {
            e.preventDefault();
            document.execCommand('insertText', false, '  ');
          }
        }}
        style={{
          lineHeight: '1.5',
        }}
      >
        {svgHistory[currentSvgIndex] || ''}
      </pre>
    </div>
  );

  // 渲染预览区域
  const renderPreview = () => {
    switch (viewMode) {
      case 'png':
        return (
          <div className="w-full h-full flex items-center justify-center bg-white">
            {pngDataUrl ? (
              <img src={pngDataUrl} alt="Converted PNG" className="max-w-full max-h-full" />
            ) : (
              <div className="text-gray-500">Converting to PNG…</div>
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
        flexDirection: isHorizontal ? 'row' : 'column'
      }}>
        {/* 代码编辑器区域 */}
        <div style={{
          width: isHorizontal ? `${splitRatio}%` : '100%',
          height: isHorizontal ? '100%' : `${splitRatio}%`,
          borderRight: isHorizontal ? '1px solid #444' : 'none',
          borderBottom: !isHorizontal ? '1px solid #444' : 'none'
        }}>
          {renderCodeEditor()}
        </div>
        
        {/* 分割器 */}
        <div
          className={`absolute bg-gray-400 hover:bg-blue-500 transition-colors cursor-${
            isHorizontal ? 'col-resize' : 'row-resize'
          } z-10`}
          style={{
            width: isHorizontal ? '4px' : '100%',
            height: isHorizontal ? '100%' : '4px',
            left: isHorizontal ? `${splitRatio}%` : '0',
            top: isHorizontal ? '0' : `${splitRatio}%`,
            transform: isHorizontal ? 'translateX(-2px)' : 'translateY(-2px)'
          }}
          onMouseDown={() => setIsDraggingSplitter(true)}
        />
        
        {/* 预览区域 */}
        <div style={{
          width: isHorizontal ? `${100 - splitRatio}%` : '100%',
          height: isHorizontal ? '100%' : `${100 - splitRatio}%`
        }}>
          {canvasContainer}
        </div>
      </div>
    );
  };

  return (
    <div className="w-full h-full bg-white rounded-lg shadow-md flex flex-col relative">
      {/* 视图模式切换和设置工具栏 */}
      <div className="p-2 border-b border-gray-200 bg-gray-50 flex items-center gap-2 flex-wrap">
        <div className="flex items-center gap-1">
          <button 
            onClick={() => setViewMode('canvas')} 
            className={`px-3 py-1 rounded-md ${viewMode === 'canvas' ? 'bg-blue-600 text-white' : 'bg-gray-100'}`}
          >
            Canvas
          </button>
          <button 
            onClick={() => { setViewMode('png'); convertSvgToPng(svgHistory[currentSvgIndex]); }} 
            className={`px-3 py-1 rounded-md ${viewMode === 'png' ? 'bg-blue-600 text-white' : 'bg-gray-100'}`}
          >
            PNG
          </button>
          <button 
            onClick={() => setViewMode('source')} 
            className={`px-3 py-1 rounded-md ${viewMode === 'source' ? 'bg-blue-600 text-white' : 'bg-gray-100'}`}
          >
            Source
          </button>
          <button 
            onClick={() => setViewMode('split')} 
            className={`px-3 py-1 rounded-md ${viewMode === 'split' ? 'bg-blue-600 text-white' : 'bg-gray-100'}`}
          >
            Split View
          </button>
        </div>

        {/* 分屏设置（仅在分屏模式下显示） */}
        {viewMode === 'split' && (
          <div className="flex items-center gap-2 ml-2 pl-2 border-l border-gray-300">
            <label className="text-sm text-gray-600">Orientation:</label>
            <select 
              value={splitOrientation}
              onChange={(e) => setSplitOrientation(e.target.value as 'horizontal' | 'vertical')}
              className="text-sm border rounded px-2 py-1"
            >
              <option value="horizontal">Horizontal</option>
              <option value="vertical">Vertical</option>
            </select>
          </div>
        )}

        {/* 代码编辑器设置（在源码或分屏模式下显示） */}
        {(viewMode === 'source' || viewMode === 'split') && (
          <div className="flex items-center gap-2 ml-2 pl-2 border-l border-gray-300">
            <label className="text-sm text-gray-600">Theme:</label>
            <select 
              value={codeTheme}
              onChange={(e) => setCodeTheme(e.target.value as 'dark' | 'light')}
              className="text-sm border rounded px-2 py-1"
            >
              <option value="dark">Dark</option>
              <option value="light">Light</option>
            </select>
            
            <label className="text-sm text-gray-600 ml-2">Font Size:</label>
            <select 
              value={codeFontSize}
              onChange={(e) => setCodeFontSize(Number(e.target.value))}
              className="text-sm border rounded px-2 py-1"
            >
              <option value={12}>12px</option>
              <option value={14}>14px</option>
              <option value={16}>16px</option>
              <option value={18}>18px</option>
              <option value={20}>20px</option>
            </select>
          </div>
        )}
      </div>

      {/* 主内容区域 */}
      <div className="flex-1 relative overflow-hidden">
        {viewMode === 'split' ? (
          renderSplitView(
            <div className="w-full h-full">
              <canvas ref={canvasRef} className="w-full h-full" />
            </div>
          )
        ) : (
          <>
            {/* Canvas-only mode */}
            {viewMode === 'canvas' && (
              <div className="w-full h-full">
                <canvas ref={canvasRef} className="w-full h-full" />
              </div>
            )}

            {/* Non-canvas previews (png/source) */}
            {(viewMode === 'png' || viewMode === 'source') ? renderPreview() : null}
          </>
        )}
      </div>
    </div>
  );
};


export default CanvasComponent;