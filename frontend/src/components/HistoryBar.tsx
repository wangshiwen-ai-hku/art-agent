import React from 'react';
import { useCanvasStore } from '../store/useCanvasStore';
import { Undo, Redo } from 'lucide-react';

const HistoryBar: React.FC = () => {
  const { svgHistory, currentSvgIndex, setCurrentSvgIndex, undo, redo } = useCanvasStore();

  // Mock data for display
  const displayHistory = svgHistory.length > 0 ? svgHistory : ['<svg viewBox="0 0 100 100"><rect width="80" height="80" x="10" y="10" fill="lightblue"/></svg>', '<svg viewBox="0 0 100 100"><circle cx="50" cy="50" r="40" fill="lightgreen"/></svg>'];
  const displayIndex = svgHistory.length > 0 ? currentSvgIndex : 0;


  return (
    <div className="bg-white p-2 border-b border-gray-200 shadow-sm">
      <div className="flex items-center space-x-4">
        <div className="flex items-center space-x-2">
          <button onClick={undo} disabled={currentSvgIndex <= 0} className="p-2 rounded-md hover:bg-gray-100 disabled:opacity-50 disabled:cursor-not-allowed">
            <Undo size={20} />
          </button>
          <button onClick={redo} disabled={currentSvgIndex >= svgHistory.length - 1} className="p-2 rounded-md hover:bg-gray-100 disabled:opacity-50 disabled:cursor-not-allowed">
            <Redo size={20} />
          </button>
        </div>
        <div className="flex-1 flex items-center space-x-2 overflow-x-auto">
          {displayHistory.map((svg, index) => (
            <button
              key={index}
              onClick={() => setCurrentSvgIndex(index)}
              className={`w-16 h-16 p-1 rounded-md border-2 ${displayIndex === index ? 'border-blue-500' : 'border-transparent'} hover:border-blue-400`}
            >
              <div className="w-full h-full bg-white rounded-sm" dangerouslySetInnerHTML={{ __html: svg }} />
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};

export default HistoryBar; 