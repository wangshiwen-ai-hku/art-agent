import CanvasComponent from './components/CanvasComponent';
import ChatComponent from './components/ChatComponent';
import HistoryBar from './components/HistoryBar';

function App() {
  return (
    <div className="flex h-screen bg-gray-100 font-sans">
      {/* Left side: Canvas and History */}
      <div className="flex flex-col flex-1">
        <HistoryBar />
        <div className="flex-1 p-4">
          <CanvasComponent />
          <div className="w-full h-full bg-gray-200 rounded-lg shadow-md flex items-center justify-center">
            {/* <p className="text-gray-500">CanvasComponent is temporarily disabled for debugging.</p> */}
          </div>
        </div>
      </div>

      {/* Right side: Chat */}
      <div className="w-1/3 max-w-md bg-white border-l border-gray-200">
        <ChatComponent />
      </div>
    </div>
  );
}

export default App;
