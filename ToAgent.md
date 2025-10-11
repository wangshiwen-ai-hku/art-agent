# GOAL
I want to build a `SVG design agent platform`, focused on the INNOVATION on **Tools build and Memory management**.
it is a **product** for a start-up company. Agent as `Adobe Illustrator`.

# What's existing?
Now I have a initial 
- `/frontend` for web chat, which includes potraits of my design agent. 
    - chat: for inspiration of design (brainstorm and instruction)
    - generate: for from the `topic` to full svg artwork.
    - edit: for edit the uploaded or pre-generated svg artwork.
    - describe: (may not include later) for describe the svg artwork with natural language.
    ......(you can FREELY design the potraits of my product as a PROFESSIONAL PRODUCT MANGAGER, since I am not professional.)
    - start with cd 
    ```bash 
    cd frontend && npm run dev```
- `src/service/canvas_agent` is the main service running in the backend. BUT it is not working now. It can only chat and have not true agents. BUT FUTURE it will become the main agent running for ALL demands from users, route their commands to its sub-agents and chat with them.
- `src/infra/tools/draw_canvas_tools.py` `src/infra/tools/edit_canvas_tools.py` `src/infra/tools/design_tools.py` `src/infra/tools/math_tools.py` `src/infra/tools/svg_tools.py` 
    - Description: These are the EXPERIMENTAL *agents as tools* tested by me, I have a proposal that this schema can 
        - Pros: help to make sub-agents more powerful and professional by provided VARIABLE tools. help reduce the `hand-off` complexity. help to manage memory by just return the desired results.
        - Cons: the messages in sub-agents-as-tools are not accessible by main-agent.

# Core challenges
I think the MAIN logically challenges for my product are 
- How AI act and DESIGN as Human, through `iterleaved observation and action`: brainstorm, draw, observation, critique, draw......
- How AI `LOCATE and ASSOCIATE the precise path/points` in the `svg code` with `visual illustration` in each `observation, critique and draw again` loop.
- How to make open platform to `evolve with users`: maybe make the `CUSTOM TOOLS CREATION` for designers, that is more professional users can custom tools easily by adding some design rules, golden techiques, even the `shortcut` similiar to Adobe Illustrator.

The main technical challenges maybe 
- how to make sub-agents more professional and can handle ALL design sub-tasks issued to them?
- how to organize their structure to multi-agents systems and with the main agent?
- how to manage the context memory?
- how to include human-in-loop interaction? and manage the memory?

# INNVATIONS (unmatured)
- agents-as-tools: (in chinese 工具人)
- shortcut-tools: (类似于一些软件中的快捷键，比如Adobe中有一键合并路径之类的，拆分路径，选取划分好的路径的几块等等)
- math-optimization-tools: (SVG 的bezier是很好的二维几何，优化这些bezier的组合几何......)
......

# my recently learning records
`agentic_design.pdf`

    