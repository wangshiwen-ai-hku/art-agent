import React, { useRef, useEffect, useState } from 'react';
// import * as fabric from 'fabric';
import { fabric } from 'fabric';
// import * as fabric from 'fabric';  
import { useCanvasStore } from '../store/useCanvasStore';

const CanvasComponent: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const fabricCanvasRef = useRef<fabric.Canvas | null>(null);
  const { svgHistory, currentSvgIndex, addSvg, setSelectionBox } = useCanvasStore();
  const [isSpacePressed, setIsSpacePressed] = useState(false);

  useEffect(() => {
    if (canvasRef.current) {
      const canvas = new fabric.Canvas(canvasRef.current, {
        width: canvasRef.current.parentElement?.clientWidth,
        height: canvasRef.current.parentElement?.clientHeight,
        backgroundColor: '#f0f0f0',
      });
      fabricCanvasRef.current = canvas;

      // Handle window resizing
      const resizeObserver = new ResizeObserver(entries => {
        for (let entry of entries) {
          const { width, height } = entry.contentRect;
          (canvas as any).setWidth(width);
          (canvas as any).setHeight(height);
        }
      });
      resizeObserver.observe(canvasRef.current.parentElement!);

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
            // Optional: remove the rectangle after selection
            // this.remove(selectionRect);
        }
        isDrawingSelection = false;
    });

      // Cleanup
      return () => {
        resizeObserver.disconnect();
        canvas.dispose();
      };
    }
  }, [isSpacePressed, setSelectionBox]);

  // Load SVG from state
  useEffect(() => {
    const canvas = fabricCanvasRef.current;
    if (canvas && svgHistory[currentSvgIndex]) {
      const svg = svgHistory[currentSvgIndex];
      fabric.loadSVGFromString(svg, (objects, options) => {
        canvas.clear();
        const obj = fabric.util.groupSVGElements(objects, options);
        canvas.add(obj).renderAll();
         // Center and zoom to fit the loaded SVG
        canvas.centerObject(obj);
        obj.scaleToWidth(canvas.getWidth() * 0.8);
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


  return (
    <div className="w-full h-full bg-white rounded-lg shadow-md">
      <canvas ref={canvasRef} />
    </div>
  );
};

export default CanvasComponent; 