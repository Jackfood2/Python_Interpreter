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

1.  **User Input:** The process begins when the user provides a high-level goal (e.g., "Find the current price of Bitcoin and save it to a file on my desktop named btc_price.txt").
2.  **Agent Selection (The Router):** The **Router** agent analyzes the user's goal and consults the list of available custom agents. It selects the agent(s) whose skills and descriptions are most relevant to the task.
3.  **Planning (The Planner):** The selected agent's **Planner** persona takes over. It formulates the single best *next step* to make progress.
4.  **Execution (The Executor):** The planned step is executed using the appropriate tool from the agent's toolbox.
5.  **Verification (The Verifier):** The outcome of the execution is rigorously checked through text analysis, visual analysis, or by asking the user for help.
6.  **Loop and Adapt:** The verified outcome is fed back to the **Planner**, which then devises the next logical step or a recovery action.
7.  **Synthesis (The Synthesizer):** Once the goal is achieved, the **Synthesizer** agent reviews the entire process to create a single, comprehensive final answer.
8.  **Reflection (The Cognitive Architect):** After the task is complete, the **Cognitive Architect** reflects on the successful steps and distills new knowledge, storing it in the long-term memory.

## Key Features

*   **Graphical User Interface:** An intuitive UI built with Tkinter for managing agents, executing tasks, and reviewing results.
*   **Custom Agent Creation:** Easily add, edit, and remove your own specialized agents with unique skills, models, and system prompts.
*   **Multi-Provider LLM Support:** Connect to OpenRouter for a wide variety of models or run models locally via LM Studio.
*   **Real-time Execution Log:** A dedicated window streams every thought, action, command output, and verification result as it happens.
*   **Advanced UI Automation:** Agents can directly control the mouse and keyboard to interact with any application on your desktop.
*   **Vision-Based Verification:** Agents can take screenshots and use vision-enabled LLMs to understand the visual state of your desktop.
*   **Human-in-the-Loop:** The system can pause and ask for your confirmation when it encounters ambiguity.
*   **Persistent Cognitive Memory:** Utilizes ChromaDB to give agents a long-term memory, allowing them to learn and improve over time.

## The Agent's Toolbox

| Tool                 | Description                                                                                                                              |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `direct_answer`      | For tasks that only require a text response, like summarization or answering a direct question.                                           |
| `cmd` / `powershell` | Provides direct access to the operating system's command line.                                                                           |
| `python_script`      | The most powerful tool for complex logic, data processing, web scraping, and API interactions.                                           |
| `pyautogui_script`   | Grants direct control over the mouse and keyboard. Used for automating applications that don't have APIs. Acts as the agent's "hands."   |
| `screenshot`         | Captures the current screen content. This is the agent's "eyes," used for visual analysis.                                                |
| `browser`            | A simple tool to open a specific URL in the user's default web browser.                                                                    |

## Installation and Setup

Follow these steps precisely to get the application running.

#### **Step 1: Prerequisites**
Ensure you have Python 3.8 or newer installed. The application also requires several third-party Python libraries to function correctly.

#### **Step 2: Install Dependencies**
To install these required libraries, open your terminal or command prompt and execute the command listed for the **"Install Required Libraries"** action in the **Command Reference Table** below.

#### **Step 3: Save the Script**
Save the provided Python code into a file named `orchestrator.py`. Place it in a dedicated folder for this project.

#### **Step 4: Run the Application**
To start the application, navigate to the folder containing your script in your terminal and execute the command listed for the **"Run the Application"** action in the **Command Reference Table**.

#### **Step 5: Provide Your API Key (CRITICAL)**
*   When you run the application for the first time, a dialog box will appear.
*   **You must provide a valid OpenRouter API Key to proceed.**
*   Paste your key into the input field and click "OK".

## Command Reference Table

This table contains all the necessary commands for installing dependencies and running the application. Copy and paste them into your terminal or command prompt.

| Purpose / Action               | Command                                           |
| ------------------------------ | ------------------------------------------------- |
| **Install Required Libraries** | `pip install --upgrade pip && pip install tk chromadb requests Pillow` |
| **Run the Application**        | `python orchestrator.py`                          |

## User Guide: Step-by-Step

#### **Step 1: Launch the Application**
After completing the setup, launch the application using the relevant command from the table above.

#### **Step 2: Explore the UI**
*   **Left Panel (Agent Management):** View your list of custom agents.
*   **Bottom Panel (Task Input):** Write your commands for the agents here.
*   **Controls (Bottom Right):** Configure the LLM provider, number of active agents, and token limits.

#### **Step 3: Configure Your Agents (Crucial for Success)**
The default agents are only examples. The true power of this application is unlocked when you create and fine-tune agents for your specific needs. **A well-crafted system prompt is the single most important factor for an agent's success.**

**How to Edit Agents:**
1.  In the left panel, either click "Add" to create a new agent or select an existing agent and click "Edit".
2.  Fill out the fields:
    *   **Agent Name:** A short, descriptive name (e.g., `CodeWriter`, `WebResearcher`).
    *   **Agent Description:** A sentence describing its skills. This is what the **Router** uses to select the agent.
    *   **LLM ID:** The model from OpenRouter you want this agent to use (e.g., `openrouter/sonoma-sky-alpha`).
    *   **System Prompt:** The detailed instructions for the agent (see the table below for high-quality examples).

#### **Sample System Prompts Table**

Use these examples as a starting point for creating your own powerful agents.

| Agent Specialization        | Sample System Prompt                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| :-------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Python Developer**        | ```You are an expert Python developer with a specialization in scripting and automation.<br>Your primary goal is to write clean, efficient, and well-documented Python code to solve the user's request.<br><br>**Core Instructions:**<br>1. **Code Quality:** Always produce high-quality, readable Python code adhering to PEP 8 standards.<br>2. **Use Standard Libraries:** Prefer Python's built-in libraries whenever possible to ensure maximum compatibility.<br>3. **Comments:** Include comments to explain complex or non-obvious parts of your code.<br>4. **Error Handling:** Incorporate basic error handling (e.g., try-except blocks).<br>5. **Output:** Use `print()` statements to provide status updates on the script's progress.``` |
| **Web Research Specialist** | ```You are a world-class research assistant. Your mission is to find the most accurate, relevant, and up-to-date information to answer the user's query.<br><br>**Your Process:**<br>1. **Deconstruct:** Break down the user's request into key search terms and questions.<br>2. **Search:** Use available tools to search for information. Prioritize reputable sources like official documentation, academic papers, and established news organizations.<br>3. **Synthesize:** Do not just copy-paste information. Understand the material from multiple sources and synthesize it into a concise, easy-to-understand summary.<br>4. **Clarity:** Present your final answer in a clear and structured format. Use headings, lists, and bold text.<br>5. **Be Objective:** Stick to the facts found in your research.``` |
| **UI Automation Specialist**  | ```You are a meticulous UI Automation Specialist. Your purpose is to control the mouse and keyboard with extreme precision to interact with desktop applications. Your primary tool is `pyautogui_script`.<br><br>**CRITICAL INSTRUCTIONS:**<br>1. **Analyze First:** Before acting, analyze the last screenshot to determine the exact coordinates (x, y) for clicks or the sequence of key presses needed.<br>2. **Act Deliberately:** Your scripts must be simple and focused on a single action (e.g., one click, typing a single field).<br>3. **Incorporate Pauses:** Use `time.sleep()` with short durations (e.g., 0.5 to 1 second) between actions to allow the UI to respond.<br>4. **Be Cautious:** You have direct control of the user's system. Double-check your planned actions to prevent unintended consequences.``` |

#### **Step 4: Define a Task and Execute**
In the large text box, write a clear, detailed prompt. Click "Execute Task".

#### **Step 5: Monitor and Interact**
Watch the "Execution Log" window. The application may pause and show a pop-up asking for "Human Verification" if it is uncertain about a step's success. Your input is crucial for keeping the task on track.

#### **Step 6: Review the Final Result**
When the task is complete, close the log window to return to the main application screen. The final answer will be in the "Final Answer" tab, and a detailed log will be available in the "Full Execution Log" tab.

## The Cognitive Memory System

The application features a long-term memory to help agents learn from experience.

*   **Episodic Memory:** A detailed record of every action and outcome from completed tasks.
*   **Procedural Memory:** Reusable, successful sequences of steps extracted from episodic memory. This is like a "skill" the agent learns.
*   **Semantic Memory:** A collection of discrete facts and pieces of information learned during tasks.
*   **Viewing Memory:** You can inspect the contents of the agent's memory by navigating to the `Settings > Cognitive Memory` menu item.

## Customization and Extensibility

*   **Add New Agents:** The easiest way to customize the system is by creating new agents with specialized skills via the UI.
*   **Modify Core Agents:** Advanced users can modify the prompts and behavior of the core agents by editing the `data/core_agents_config.json` file.
*   **Integrate New Tools:** The application's code can be extended to include new tools for agents to use.

## File and Directory Structure

When first run, the application will create a `data` directory with the following structure:

*   `data/`
    *   `agents.json`: Stores the configurations for all your custom agents.
    *   `core_agents_config.json`: Contains the advanced settings and prompts for the core agents (Planner, Verifier, etc.).
    *   `memory/`: The directory containing all cognitive memory data.
        *   `chroma_db/`: The persistent database for procedural and semantic memory.
        *   `episodes/`: JSON files logging each completed task.

## Important Notes and Disclaimers

*   **Security:** This application gives AI agents the ability to execute code and control your computer. Only run tasks from trusted sources and be aware of the commands being executed. The user is responsible for the actions taken by the agents.
*   **API Costs:** Using the OpenRouter API will incur costs based on your usage of their models. Monitor your usage and set spending limits on your Open
