import logging
from langgraph.graph import StateGraph
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

def draw_graph(graph: StateGraph, filename: str):
    """Draws the agent graph and saves it to a file."""
    try:
        # Get the compiled graph representation
        compiled_graph = graph.compile()
        
        # Draw the graph and save as PNG
        with open(f"{filename}.png", "wb") as f:
            f.write(compiled_graph.get_graph().draw_mermaid_png())
        logger.info(f"✅ Graph visualization saved to {filename}.png")
        
    except Exception as e:
        logger.error(f"❌ Failed to draw graph: {e}")
        logger.error("Please ensure you have graphviz installed (`sudo apt-get install graphviz`) and pygraphviz (`pip install pygraphviz`).")

