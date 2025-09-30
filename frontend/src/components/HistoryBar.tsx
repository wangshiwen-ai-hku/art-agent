import React from 'react';
import { useCanvasStore } from '../store/useCanvasStore';
import { Undo, Redo } from 'lucide-react';

const HistoryBar: React.FC = () => {
  const { svgHistory, currentSvgIndex, setCurrentSvgIndex, undo, redo } = useCanvasStore();

  // Mock data for display
  const displayHistory = svgHistory.length > 0 ? svgHistory : [];
  const displayIndex = svgHistory.length > 0 ? currentSvgIndex : -1;


  return (
    <div className="bg-gray-800 bg-opacity-80 backdrop-blur-sm p-2 rounded-xl shadow-lg border border-white border-opacity-20">
      <div className="flex items-center space-x-4">
        <div className="flex items-center space-x-2">
          <button onClick={undo} disabled={currentSvgIndex <= 0} className="p-2 rounded-md text-white hover:bg-white hover:bg-opacity-20 disabled:opacity-50 disabled:cursor-not-allowed">
            <Undo size={20} />
          </button>
          <button onClick={redo} disabled={currentSvgIndex >= svgHistory.length - 1} className="p-2 rounded-md text-white hover:bg-white hover:bg-opacity-20 disabled:opacity-50 disabled:cursor-not-allowed">
            <Redo size={20} />
          </button>
        </div>
        <div className="flex-1 overflow-x-auto scrollbar-thin scrollbar-thumb-blue-500 scrollbar-track-gray-800">
        <div className="flex items-center gap-2 h-20 px-1">
          {displayHistory.map((svg, index) => (
            <button
              key={index}
              onClick={() => setCurrentSvgIndex(index)}
              className={`flex-shrink-0 w-20 h-20 p-1 rounded-md border-2 ${displayIndex === index ? 'border-blue-400' : 'border-transparent'} hover:border-blue-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-opacity-50`}
            >
              <div className="w-full h-full bg-white rounded-sm flex items-center justify-center overflow-hidden" dangerouslySetInnerHTML={{ __html: svg }} />
            </button>
          ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default HistoryBar; 