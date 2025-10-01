import CanvasComponent from './components/CanvasComponent';
// import MODE_COLOR from './components/ChatComponent'
import ChatComponent from './components/ChatComponent';

function App() {
  return (
    <div style={{
      width: '100vw',
      height: '100vh',
      background: 'linear-gradient(135deg,rgb(235, 236, 243) 0%,rgb(208, 196, 221) 100%)',
      position: 'relative',
      overflow: 'hidden',
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif'
    }}>
      {/* Canvas Component - Left side with fixed positioning */}
      <CanvasComponent />
      
      {/* Chat Component - Right side with fixed positioning */}
      <ChatComponent />
    </div>
  );
}

export default App;
