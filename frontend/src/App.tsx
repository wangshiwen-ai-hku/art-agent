import CanvasComponent from './components/CanvasComponent';
import ChatComponent from './components/ChatComponent';
// import HistoryBar from './components/HistoryBar';

function App() {
  return (
    <div className="flex h-screen bg-gray-100 font-sans overflow-hidden">
      {/* Left side: Canvas and History */}
    <div className="flex flex-col flex-1">
        
        <div className="flex-1 p-4">
          <CanvasComponent />
          
        </div>
      </div>

      {/* Right side: Chat */}
      <div className="w-1/3 max-w-md bg-white shadow-lg flex flex-col">
        <ChatComponent />
      </div>
    </div>
  );
}

export default App;
