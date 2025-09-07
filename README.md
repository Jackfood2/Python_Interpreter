```markdown
# AGI-like Orchestrator: Multi-Agent Task Automation System

**AGI-like Orchestrator** is an advanced, locally-run desktop application that simulates an Artificial General Intelligence (AGI) by orchestrating a team of specialized AI agents. These agents can plan, execute, and verify complex, multi-step tasks on your computer by interacting with the operating system, writing and running code, controlling the mouse and keyboard, browsing the web, and analyzing screenshots. The system learns from its experiences, building a long-term memory of procedures and facts to improve future performance.

Built with Python and Tkinter, it provides a user-friendly graphical interface for defining tasks and monitoring agent activity in real-time.

---

## Table of Contents
1.  [Key Features](#key-features)
2.  [System Architecture](#system-architecture)
3.  [Core Agents](#core-agents)
4.  [User Agents](#user-agents)
5.  [Memory System](#memory-system)
6.  **Getting Started**
    *   [Prerequisites](#prerequisites)
    *   [Installation](#installation)
    *   [Configuration](#configuration)
7.  [Usage Guide](#usage-guide)
8.  [LLM Providers](#llm-providers)
9.  [Advanced Features & Settings](#advanced-features--settings)
10. [Troubleshooting](#troubleshooting)
11. [Important Notes & Limitations](#important-notes--limitations)
12. [License](#license)

---

## Key Features

*   **Multi-Agent Collaboration:** Define and manage a team of specialized AI agents (e.g., Python Developer, Web Researcher) to tackle complex problems.
*   **Autonomous Task Execution:** Agents can break down your high-level goal into executable steps, including running shell commands, executing Python scripts, and automating GUI interactions via `pyautogui`.
*   **Self-Verification & Recovery:** The system doesn't just execute; it verifies if each step was successful using text analysis, computer vision (screenshot analysis), and can even ask the user for help. It can recover from failures by retrying with different methods.
*   **Persistent Memory:** Learns and stores reusable procedures and factual knowledge from past successes, making it smarter over time.
*   **Flexible LLM Backend:** Supports both cloud-based (OpenRouter) and local (LM Studio) Large Language Models for maximum flexibility and privacy.
*   **Real-Time Logging:** A dedicated window streams the agents' thoughts, actions, and outputs as they work.
*   **Extensible Design:** Easily add, edit, or remove agents and modify the core system's behavior through intuitive GUI settings.

---

## System Architecture

The application follows a modular, agent-based architecture:

1.  **User Interface (GUI):** The `TaskExecutorGUI` class handles all user interactions, input, and output display.
2.  **Agent Manager:** The `AgentManager` acts as the conductor. It selects the most suitable agents for a task, manages their execution (often in parallel threads), and synthesizes their results.
3.  **Agents:** The `Agent` class is the workhorse. Each agent instance uses an LLM to generate a step-by-step plan based on the goal, its own expertise, and retrieved memories. It then executes these steps.
4.  **Core Cognitive Agents:** Specialized agents handle critical system functions:
    *   **Router:** Selects which user agents are best for the task.
    *   **Planner:** Generates the next executable step for a user agent.
    *   **Verifier:** Determines if a step succeeded or failed.
    *   **Synthesizer:** Combines outputs from multiple agents into a final answer.
    *   **Cognitive Architect:** Reflects on completed tasks to extract and store new knowledge.
5.  **Memory Manager:** Uses ChromaDB for vector storage to manage three types of memory: Episodic (past task logs), Procedural (learned step-by-step guides), and Semantic (learned facts).
6.  **LLM Interface:** The `query_llm` and `query_vision_llm` functions handle communication with the chosen LLM provider (OpenRouter or LM Studio).

---

## Core Agents

These are the internal agents that power the system's intelligence. Their prompts and LLM IDs can be configured in the "Settings" menu.

*   **`ROUTER`**: Selects the best user agents for the given task.
*   **`PLANNER`**: The most complex agent. It generates the next single step for a user agent, choosing from tools like `python_script`, `pyautogui_script`, `cmd`, `powershell`, `browser`, or `screenshot`. It is designed to be reactive, adapting its plan based on the outcome of the previous step and employing fallback strategies (e.g., switching from Selenium to PyAutoGUI if a web element can't be found).
*   **`VERIFIER`**: Analyzes the output of an executed step to determine if it was a success, failure, or uncertain. Triggers visual verification if needed.
*   **`QUESTION_FORMULATOR`**: Generates simple yes/no questions for the Verifier to use when analyzing a screenshot.
*   **`IMAGE_DESCRIBER`**: A vision-capable LLM that answers the Verifier's questions based on a screenshot.
*   **`SYNTHESIZER`**: Combines the results from multiple user agents into a coherent final response for the user.
*   **`ARCHITECT`**: (Planned) Analyzes completed tasks to extract and store new procedures and facts into long-term memory.

---

## User Agents

These are the customizable agents you interact with directly. You can create, edit, and delete them via the GUI. Each user agent has:
*   A **Name** (e.g., "PythonAgent").
*   A **Description** defining its expertise (used by the Router).
*   An **LLM ID** specifying which model it uses.
*   A **System Prompt** that guides its overall behavior and planning style.

The application comes with two default user agents:
*   **PythonAgent**: Expert in writing and executing Python scripts for automation and data processing.
*   **ResearchAgent**: Skilled at using the browser to find information.

---

## Memory System

The system learns and retains knowledge in three ways:

1.  **Episodic Memory:** A complete log (stored as JSON files) of every step taken and its outcome for each task. This is like the system's diary.
2.  **Procedural Memory:** (Via `CognitiveArchitect`) Stores successful sequences of commands as reusable "procedures." For example, if an agent successfully automates a complex file backup, this sequence can be recalled and reused for a similar future task.
3.  **Semantic Memory:** (Via `CognitiveArchitect`) Stores discrete pieces of learned factual information.

This memory is persistent across application restarts and can be viewed or cleared via the "Cognitive Memory" viewer in the Settings menu.

---

## Getting Started

### Prerequisites

*   **Python 3.8+**: Ensure Python is installed on your system.
*   **Required Python Packages**: Install the dependencies listed in the next section.
*   **(Optional) LM Studio**: If you want to use local models, download and install [LM Studio](https://lmstudio.ai/). You will need to load a compatible model (e.g., `nvidia/nvidia-nemotron-nano-9b-v2`) and ensure its server is running on `http://127.0.0.1:1234`.
*   **(Optional) OpenRouter API Key**: If you want to use cloud models, sign up at [OpenRouter.ai](https://openrouter.ai/) and obtain an API key.

### Installation

1.  **Clone or Download:** Get the source code by cloning the repository or downloading the files.
2.  **Install Dependencies:** Open a terminal in the project directory and run:
    ```bash
    pip install -r requirements.txt
    ```
    *If a `requirements.txt` file is not provided, install the packages listed in the imports:*
    ```bash
    pip install chromadb requests pillow tk pyautogui selenium webbrowser
    ```
3.  **Configure API Key (For OpenRouter):** Open the `main.py` file (or the file containing the code) and locate the `OPENROUTER_API_KEY` variable near the top. Replace `"skxxxxx"` with your actual OpenRouter API key.
    ```python
    OPENROUTER_API_KEY = "your_actual_api_key_here"
    ```

### Configuration

*   **LLM Provider:** Choose between "OpenRouter" (cloud) and "LM Studio" (local) in the bottom-right panel of the main GUI before running a task.
*   **Core Agents:** Fine-tune the system's behavior by editing the prompts for the Router, Planner, Verifier, etc., via `Settings > Core Agents`.
*   **User Agents:** Add, edit, or remove agents via the "Agent Management" panel on the left side of the GUI.

---

## Usage Guide

1.  **Launch the Application:** Run the script with `python main.py`.
2.  **(Optional) Configure Agents:** Use the left panel to manage your team of User Agents.
3.  **Enter Your Task:** Type your high-level goal or question into the large text box at the bottom of the window. For example:
    *   "Create a Python script that lists all `.txt` files in my Documents folder and saves the list to `file_report.txt`."
    *   "Research the latest advancements in solar panel efficiency and summarize them for me."
    *   "Open Notepad, type 'Hello, World!', and save it as `test.txt` on my desktop."
4.  **Select Settings:**
    *   Choose your **LLM Provider** (OpenRouter or LM Studio).
    *   Set the number of **Active Agents** (usually 1 is sufficient for most tasks).
    *   Adjust the **Max Tokens** slider if needed (default is usually fine).
5.  **Execute:** Click the "Execute Task" button.
6.  **Monitor Execution:** A new "Execution Log" window will pop up, showing the agents' real-time thoughts, commands, outputs, and screenshots. You can minimize but not close this window while the task is running.
7.  **Human Verification (If Needed):** If the system is uncertain about a step's outcome, a dialog box will appear asking you a simple yes/no question based on a screenshot. Your answer helps it proceed.
8.  **View Results:** When the task is complete, the "Execution Log" window's "Stop Task" button changes to "Close". The final answer will appear in the "Final Answer" tab of the main window. You can also review the complete, formatted log in the "Full Execution Log" tab.
9.  **Stop a Task:** If a task is taking too long or behaving unexpectedly, click the "Stop Task" button in the Execution Log window.

---

## LLM Providers

The system can route requests to two different backends:

*   **OpenRouter (Default):**
    *   **Pros:** Access to a wide variety of powerful, state-of-the-art models. No local hardware requirements.
    *   **Cons:** Requires an internet connection and an API key. Usage may incur costs.
    *   **Setup:** Add your API key to the `OPENROUTER_API_KEY` variable in the code.

*   **LM Studio:**
    *   **Pros:** Runs entirely locally, ensuring privacy and no API costs. Works offline.
    *   **Cons:** Requires a powerful local machine (especially GPU) to run models effectively. Model quality may be lower than top-tier cloud models.
    *   **Setup:** Install LM Studio, download and load a model, and ensure its local server is running. The application is pre-configured to connect to `http://127.0.0.1:1234/v1`.

You can select the provider for each task via the radio buttons in the GUI.

---

## Advanced Features & Settings

*   **Core Agent Configuration:** Accessible via `Settings > Core Agents`, this allows you to modify the "brains" of the system. You can change the LLM used by each core agent and, more importantly, edit their system prompts to alter their decision-making logic. *Use with caution, as this can significantly change system behavior.*
*   **Memory Viewer:** Accessible via `Settings > Cognitive Memory`, this tool lets you inspect the system's learned procedures and facts, or clear all memory.
*   **Direct Answer:** The Planner agent can choose to respond directly to simple queries (e.g., "What is the capital of France?") without executing any system commands, making it function like a standard chatbot for straightforward questions.

---

## Troubleshooting

*   **"No suitable agents found."**: Ensure you have at least one User Agent configured in the left panel. Check that their descriptions are clear and relevant to the task.
*   **Task hangs or takes too long**: Use the "Stop Task" button in the Execution Log window. This often happens if an LLM call times out or a script enters an infinite loop.
*   **LM Studio not working**: Double-check that:
    1.  LM Studio is installed and running.
    2.  A model is fully downloaded and loaded.
    3.  The "Local Server" is started within LM Studio (it should show "Server running on 127.0.0.1:1234").
    4.  The model ID in the code (`LMSTUDIO_MODEL_ID`) matches the one you have loaded, or is a valid placeholder.
*   **OpenRouter API errors**: Verify your API key is correct and that you have an active internet connection. Check your OpenRouter account for rate limits or billing issues.
*   **PyAutoGUI actions not working**: The system estimates screen coordinates from screenshots. If your screen resolution changes or the UI layout is dynamic, these coordinates can be wrong. This is a known limitation of the fallback strategy.
*   **ChromaDB Errors**: If you encounter database errors, you can try clearing the memory via `Settings > Cognitive Memory > Clear ALL Memory`.

---

## Important Notes & Limitations

*   **Security Warning:** This application can execute arbitrary code and control your mouse and keyboard. **Only give it tasks you trust.** Never run it with elevated privileges unless absolutely necessary. Treat it like a powerful, but potentially mischievous, intern.
*   **Beta Software:** This is a complex prototype. It may contain bugs, produce unexpected results, or fail to complete tasks.
*   **LLM Hallucinations:** The system relies on LLMs, which can generate incorrect or nonsensical code and plans. The verification system mitigates this but doesn't eliminate it.
*   **Resource Intensive:** Running local models or complex tasks can consume significant CPU, GPU, and memory resources.
*   **UI Automation is Fragile:** Automating graphical user interfaces (GUIs) via `pyautogui` is inherently brittle. It relies on static screen coordinates and can break if windows move, applications update, or screen resolutions change.
*   **API Keys in Code:** Storing the OpenRouter API key directly in the source code is **not secure** for production or shared environments. For better security, consider loading it from an environment variable.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
```
