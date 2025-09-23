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
}

export const useCanvasStore = create<CanvasState>((set) => ({
  svgHistory: [],
  currentSvgIndex: -1,
  messages: [],
  selectionBox: null,
  
  addSvg: (svg) => set((state) => {
    const newHistory = state.svgHistory.slice(0, state.currentSvgIndex + 1);
    newHistory.push(svg);
    return { 
      svgHistory: newHistory,
      currentSvgIndex: newHistory.length - 1,
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
})); 