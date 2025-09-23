import { create } from 'zustand';

export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

interface CanvasState {
  svgHistory: string[];
  currentSvgIndex: number;
  messages: { author: 'user' | 'agent'; content: string }[];
  selectionBox: BoundingBox | null;
  
  addSvg: (svg: string) => void;
  setCurrentSvgIndex: (index: number) => void;
  addMessage: (message: { author: 'user' | 'agent'; content: string }) => void;
  setSelectionBox: (box: BoundingBox | null) => void;
  undo: () => void;
  redo: () => void;
  updateSvg: (index: number, newSvg: string) => void; // 新增的方法
}

export const useCanvasStore = create<CanvasState>((set) => ({
  svgHistory: [],
  currentSvgIndex: -1,
  messages: [],
  selectionBox: null,
  
  addSvg: (svg) => set((state) => {
    const newHistory = state.svgHistory.slice(0, state.currentSvgIndex + 1);
    newHistory.push(svg);
    // Enforce a maximum of 10 items in history
    const limitedHistory = newHistory.slice(Math.max(0, newHistory.length - 10));
    return { 
      svgHistory: limitedHistory,
      currentSvgIndex: limitedHistory.length - 1,
     };
  }),

  setCurrentSvgIndex: (index) => set({ currentSvgIndex: index }),

  addMessage: (message) => set((state) => ({ messages: [...state.messages, message] })),
  
  setSelectionBox: (box) => set({ selectionBox: box }),

  undo: () => set((state) => ({
    currentSvgIndex: Math.max(0, state.currentSvgIndex - 1),
  })),

  redo: () => set((state) => ({
    currentSvgIndex: Math.min(state.svgHistory.length - 1, state.currentSvgIndex + 1),
  })),

  // 新增的 updateSvg 方法
  updateSvg: (index: number, newSvg: string) => set((state) => {
    if (index < 0 || index >= state.svgHistory.length) {
      console.error('Invalid index for updateSvg:', index);
      return state;
    }
    
    const newHistory = [...state.svgHistory];
    newHistory[index] = newSvg;
    
    return { 
      svgHistory: newHistory,
      // 保持当前索引不变，因为我们只是更新了当前 SVG 的内容
      currentSvgIndex: state.currentSvgIndex
    };
  }),
}));