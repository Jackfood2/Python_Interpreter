# Multi-Agent Orchestrator

An advanced framework for orchestrating autonomous AI agents to collaboratively solve complex tasks. This application provides a graphical user interface to define, manage, and deploy a team of agents that can reason, plan, and execute actions on your computer.

## Table of Contents
1.  [Overview](#overview)
2.  [Core Concepts](#core-concepts)
3.  [System Architecture](#system-architecture)
4.  [Key Features](#key-features)
5.  [The Agent's Toolbox](#the-agents-toolbox)
6.  [Installation and Setup](#installation-and-setup)
7.  [Command Reference Table](#command-reference-table)
8.  [User Guide: Step-by-Step](#user-guide-step-by-step)
9.  [The Cognitive Memory System](#the-cognitive-memory-system)
10. [Customization and Extensibility](#customization-and-extensibility)
11. [File and Directory Structure](#file-and-directory-structure)
12. [Important Notes and Disclaimers](#important-notes-and-disclaimers)

---

## Overview

The Multi-Agent Orchestrator is more than just a chatbot; it's a dynamic problem-solving environment. It takes a high-level goal from a user and leverages a team of specialized AI agents to achieve it. These agents can write code, run commands, control your mouse and keyboard, analyze what's on the screen, and learn from their experiences to become more effective over time.

This application is ideal for automating complex digital workflows, performing advanced research, and exploring the capabilities of autonomous AI systems in a controlled environment.

## Core Concepts

*   **Orchestration:** Instead of a single AI model trying to do everything, this application acts as an orchestrator or "project manager." It intelligently selects the right agent for each part of a task.
*   **Autonomy:** Agents have the autonomy to create and adapt their own plans. The **Planner** agent is designed to be reactive, meaning it adjusts its strategy based on the success or failure of the previous step.
*   **Verification:** A critical part of autonomy is verification. The system doesn't just assume an action worked. The **Verifier** agent actively checks the outcome of each step, using command-line output, visual analysis of the screen, or even by asking the user for help.
*   **Cognitive Memory:** The system is designed to learn. Successful task executions are analyzed by the **Cognitive Architect**, which extracts reusable procedures and valuable facts. This knowledge is stored in a long-term memory database (ChromaDB) to improve future performance.

## System Architecture

The application follows a structured, cyclical workflow for every task:

1.  **User Input:** The process begins when the user provides a high-level goal.
2.  **Agent Selection (The Router):** The **Router** agent analyzes the goal and selects the best custom agent(s) for the job.
3.  **Planning (The Planner):** The selected agent's **Planner** persona takes over, formulating the single best *next step*.
4.  **Execution (The Executor):** The planned step is executed using the appropriate tool.
5.  **Verification (The Verifier):** The outcome is rigorously checked through text analysis, visual analysis, or by asking the user.
6.  **Loop and Adapt:** The verified outcome is fed back to the **Planner**, which devises the next step or a recovery action.
7.  **Synthesis (The Synthesizer):** Once the goal is achieved, the **Synthesizer** agent creates a final, comprehensive answer.
8.  **Reflection (The Cognitive Architect):** After the task, the **Cognitive Architect** reflects on the process and distills new knowledge into the long-term memory.

## Key Features

*   **Graphical User Interface:** An intuitive UI built with Tkinter for managing agents, executing tasks, and reviewing results.
*   **Custom Agent Creation:** Easily add, edit, and remove your own specialized agents with unique skills and prompts.
*   **Multi-Provider LLM Support:** Connect to OpenRouter or run models locally via LM Studio.
*   **Real-time Execution Log:** A dedicated window streams every thought, action, and result as it happens.
*   **Advanced UI Automation:** Agents can directly control the mouse and keyboard to interact with any application.
*   **Vision-Based Verification:** Agents can take screenshots and use vision-enabled LLMs to understand the visual state of your desktop.
*   **Human-in-the-Loop:** The system can pause and ask for your confirmation when it encounters ambiguity.
*   **Persistent Cognitive Memory:** Utilizes ChromaDB to give agents a long-term memory, allowing them to learn and improve.

## The Agent's Toolbox

| Tool                 | Description                                                                                             |
| -------------------- | ------------------------------------------------------------------------------------------------------- |
| `direct_answer`      | For tasks requiring only a text response (e.g., summarization).                                         |
| `cmd` / `powershell` | Provides direct access to the operating system's command line.                                          |
| `python_script`      | The most powerful tool for complex logic, data processing, web scraping, and API interactions.          |
| `pyautogui_script`   | Grants direct control over the mouse and keyboard. Acts as the agent's "hands."                           |
| `screenshot`         | Captures the current screen content. This is the agent's "eyes," used for visual analysis.                |
| `browser`            | A simple tool to open a specific URL in the user's default web browser.                                 |

# Agentic Workflow: Memory-Driven Task Execution

This document outlines how our AI agent system leverages its cognitive memory to efficiently plan and execute tasks. We will use a concrete example to illustrate the entire workflow from user request to task completion.

### The Example Task

The user provides the following prompt:
find out top 5 AI news and save in txt on desktop

---

## How Memory is Retrieved

Before the agent takes any action, it first consults its memory. This allows the system to learn from past experiences and approach new tasks more intelligently. The core concepts of this retrieval process are detailed below.

| Concept                      | Description                                                                                                                                                                                                                                                                                                                                                             |
| :--------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Vector Database for Memory** | The system uses `ChromaDB`, a vector database, to store and retrieve memories. This allows it to find relevant information based on *conceptual similarity*, not just exact keyword matches.                                                                                                                                                                             |
| **Two Core Memory Types**    | The memory is divided into two distinct types: <br> - **🧠 Procedural Memory**: Stores successful, multi-step action plans from past tasks. For our example, it might find a procedure like "How to search the web for information and save results to a file". <br> - **💡 Semantic Memory**: Stores individual, useful facts. For example, it might recall `os.path.expanduser('~/Desktop')` returns the user's desktop path. |
| **Querying Process**         | The user's entire prompt is converted into a numerical vector. This vector is then used as a search query against both procedural and semantic memory collections to find the most similar and relevant stored memories.                                                                                                                                               |
| **Informing the Planner**    | The most relevant procedures and facts are retrieved and injected directly into the prompt that is sent to the **Planner Agent**. This context is provided under a clear heading, such as `## Relevant Knowledge from Memory:`.                                                                                                                                         |
| **Making an Informed Decision**  | By receiving this context upfront, the Planner Agent gets a significant head-start. It doesn't have to figure everything out from scratch and can immediately formulate a more efficient plan based on strategies that have worked in the past.                                                                                                                           |

---

## Simple Workflow Diagram

Here is a step-by-step breakdown of the process for our example task.

**[START]**

1.  **User Input**
    *   The system receives the task: find out top 5 AI news and save in txt on desktop
    *   ➡️

2.  **Query Memory**
    *   The full prompt is vectorized and sent to the ChromaDB memory system.
    *   ➡️

3.  **Search & Retrieve**
    *   The system performs a similarity search across its two memory types to find relevant procedures and facts.
    *   ➡️

4.  **Construct Context**
    *   The top matching memories are gathered and formatted into a text block to be passed to the Planner.
    *   ➡️

5.  **Build Planner Prompt**
    *   A large prompt is assembled for the Planner LLM, containing:
        1.  Retrieved Memories (Context)
        2.  Original User Goal
        3.  Task History (empty for the first step)
    *   ➡️

6.  **LLM Planning**
    *   The complete prompt is sent to the Planner LLM, which uses the retrieved memories to generate an informed and efficient first step.
    *   ➡️

7.  **Execute Step**
    *   The system executes the generated step (e.g., runs a Python script to search for news).

**[PROCESS REPEATS UNTIL TASK IS COMPLETE]**


## Installation and Setup

Follow these steps precisely to get the application running.

#### **Step 1: Get Your Free OpenRouter API Key**
Before you can use the application, you need an API key from OpenRouter.
1.  Go to the OpenRouter website: **[https://openrouter.ai/](https://openrouter.ai/)**
2.  Sign up for a new account. OpenRouter provides new users with free credits, which allows you to experiment with many powerful models at no cost.
3.  Once logged in, navigate to your account settings and find the **Keys** page.
4.  Generate a new API key.
5.  **Copy this key and keep it ready.** You will need it in Step 6.

#### **Step 2: System Prerequisites**
Ensure you have Python 3.8 or newer installed and that your environment can install third-party packages.

#### **Step 3: Install Dependencies**
To install the required libraries, open your terminal or command prompt and execute the command listed for the **"Install Required Libraries"** action in the **Command Reference Table** below.

#### **Step 4: Save the Script**
Save the provided Python code into a file named `python_interpreter.py` in a dedicated folder.

#### **Step 5: Run the Application**
To start the application, navigate to the folder containing your script in your terminal and execute the command listed for the **"Run the Application"** action in the **Command Reference Table**.

#### **Step 6: Provide Your API Key (CRITICAL)**
*   When you run the application for the first time, a dialog box will appear.
*   **Paste your OpenRouter API Key from Step 1 into the input field and click "OK".**

## Command Reference Table

| Purpose / Action               | Command                                           |
| ------------------------------ | ------------------------------------------------- |
| **Install Required Libraries** | `pip install --upgrade pip && pip install tk chromadb requests Pillow` |
| **Run the Application**        | `python python_interpreter.py`                          |

## User Guide: Step-by-Step

#### **Step 1: Launch and Configure**
After completing the setup, launch the application. The main window will appear.

#### **Step 2: Configure Your Agents (Crucial for Success)**
The true power of this application is unlocked when you create and fine-tune agents. A well-crafted **System Prompt** is the single most important factor for an agent's success.

##### **Recommended Models**
This application has been tested and performs exceptionally well with the following OpenRouter LLM IDs. When creating or editing agents, we highly recommend using one of these:
*   `openrouter/sonoma-dusk-alpha`
*   `openrouter/sonoma-sky-alpha`

These models are notable for their massive **2 million token context windows**. This makes them excellent for complex, multi-step tasks because they can remember the entire history of actions, outputs, and verifications, leading to much better decision-making by the **Planner** agent.

##### **How to Edit Agents and Prompts**
1.  In the left panel, click "Add" to create a new agent or select an existing one and click "Edit".
2.  Fill out the fields:
    *   **Agent Name:** A short, descriptive name (e.g., `CodeWriter`).
    *   **Agent Description:** A sentence describing its skills. This is what the **Router** uses to select the agent.
    *   **LLM ID:** Enter one of the recommended models above.
    *   **System Prompt:** The detailed instructions for the agent (see the table below for high-quality examples).

##### **Core and Custom Agent Prompts Table**
Use these examples as a starting point for creating your own powerful custom agents. You can view the full prompts for the built-in Core Agents via the `Settings > Core Agents` menu.

| Agent Specialization        | Sample System Prompt                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| :-------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Python Developer**        | ```You are an expert Python developer with a specialization in scripting and automation.<br>Your primary goal is to write clean, efficient, and well-documented Python code to solve the user's request.<br><br>**Core Instructions:**<br>1. **Code Quality:** Always produce high-quality, readable Python code adhering to PEP 8 standards.<br>2. **Use Standard Libraries:** Prefer Python's built-in libraries whenever possible to ensure maximum compatibility.<br>3. **Comments:** Include comments to explain complex or non-obvious parts of your code.<br>4. **Error Handling:** Incorporate basic error handling (e.g., try-except blocks).<br>5. **Output:** Use `print()` statements to provide status updates on the script's progress.``` |
| **Web Research Specialist** | ```You are a world-class research assistant. Your mission is to find the most accurate, relevant, and up-to-date information to answer the user's query.<br><br>**Your Process:**<br>1. **Deconstruct:** Break down the user's request into key search terms and questions.<br>2. **Search:** Use available tools to search for information. Prioritize reputable sources like official documentation, academic papers, and established news organizations.<br>3. **Synthesize:** Do not just copy-paste information. Understand the material from multiple sources and synthesize it into a concise, easy-to-understand summary.<br>4. **Clarity:** Present your final answer in a clear and structured format. Use headings, lists, and bold text.<br>5. **Be Objective:** Stick to the facts found in your research.``` |
| **UI Automation Specialist**  | ```You are a meticulous UI Automation Specialist. Your purpose is to control the mouse and keyboard with extreme precision to interact with desktop applications. Your primary tool is `pyautogui_script`.<br><br>**CRITICAL INSTRUCTIONS:**<br>1. **Analyze First:** Before acting, analyze the last screenshot to determine the exact coordinates (x, y) for clicks or the sequence of key presses needed.<br>2. **Act Deliberately:** Your scripts must be simple and focused on a single action (e.g., one click, typing a single field).<br>3. **Incorporate Pauses:** Use `time.sleep()` with short durations (e.g., 0.5 to 1 second) between actions to allow the UI to respond.<br>4. **Be Cautious:** You have direct control of the user's system. Double-check your planned actions to prevent unintended consequences.``` |
| **Core Agent: PLANNER**     | `You are a meticulous, reactive, and persistent AGI Planner. Your job is to devise the single best next step to achieve a goal, adapting your plan based on the outcome of the previous step...`<br>*(The full, detailed prompt is available in the Core Agents settings and explains the agent's philosophy on tool selection, UI automation best practices, and reactive planning with fallback strategies.)* |

#### **Step 3: Define a Task and Execute**
In the large text box, write a clear, detailed prompt. Click "Execute Task".

#### **Step 4: Monitor and Interact**
Watch the "Execution Log" window. The application may pause and show a pop-up asking for "Human Verification" if it is uncertain. Your input is crucial for keeping the task on track.

#### **Step 5: Review the Final Result**
When the task is complete, close the log window. The final answer will be in the "Final Answer" tab, and a detailed log will be available in the "Full Execution Log" tab.

## The Cognitive Memory System

The application features a long-term memory to help agents learn from experience.

*   **Episodic Memory:** A detailed record of every action and outcome from completed tasks.
*   **Procedural Memory:** Reusable, successful sequences of steps extracted from episodic memory (a "skill").
*   **Semantic Memory:** A collection of discrete facts and pieces of information learned during tasks.
*   **Viewing Memory:** You can inspect the agent's memory via the `Settings > Cognitive Memory` menu item.

## Customization and Extensibility

*   **Add New Agents:** Create new agents with specialized skills via the UI.
*   **Modify Core Agents:** Advanced users can modify the core agent prompts via the `Settings` menu.
*   **Integrate New Tools:** The application's code can be extended to include new tools for agents to use.

## File and Directory Structure

When first run, the application will create a `data` directory:

*   `data/`
    *   `agents.json`: Stores your custom agent configurations.
    *   `core_agents_config.json`: Stores the core agent prompts.
    *   `memory/`: Contains all cognitive memory data.
        *   `chroma_db/`: The persistent database for procedural and semantic memory.
        *   `episodes/`: JSON files logging each completed task.

## Important Notes and Disclaimers

*   **Security:** This application gives AI agents the ability to execute code and control your computer. Only run tasks from trusted sources and be aware of the commands being executed. The user is responsible for the actions taken by the agents.
*   **API Costs:** Using the OpenRouter API will incur costs based on your usage. Monitor your usage and set spending limits on your OpenRouter account. The initial free credits are generous but not unlimited.
