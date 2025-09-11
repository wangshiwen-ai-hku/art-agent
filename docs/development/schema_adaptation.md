# Best Practice Pattern For Schema Adaptation — "Adapter Node"

## Define Explicit State Schemas

Make both the parent and subgraph state types explicit using TypedDict (or pydantic models if you want validation).

```python
from typing import TypedDict

class ParentState(TypedDict):
    query: str
    user_id: str
    history: list

class SubgraphState(TypedDict):
    prompt: str
    context: list
```

## Write a Wrapper Node for Schema Adaptation

This node:

- Extracts and transforms relevant keys from ParentState → SubgraphState.
- Calls subgraph.invoke() or subgraph.stream() with the adapted state.
- Maps results back into the ParentState.

```python
from langgraph.graph import StateGraph

def parent_to_subgraph_adapter(parent_state: ParentState, *, subgraph):
    # Step 1: Adapt parent state → subgraph state
    sub_state = {
        "prompt": parent_state["query"],
        "context": parent_state["history"]
    }

    # Step 2: Run the subgraph
    sub_result = subgraph.invoke(sub_state)

    # Step 3: Adapt subgraph output → parent state
    return {
        "history": parent_state["history"] + [sub_result["response"]]
    }
```

## Add the Adapter Node into the Parent Graph

Instead of directly adding the subgraph as a node, you add the adapter node.

```python
parent_graph = StateGraph(ParentState)

# Compile subgraph separately
compiled_subgraph = subgraph.compile()

# Add adapter node with subgraph injection
parent_graph.add_node("invoke_subgraph", lambda s: parent_to_subgraph_adapter(s, subgraph=compiled_subgraph))

# Define edges
parent_graph.set_entry_point("invoke_subgraph")
```
