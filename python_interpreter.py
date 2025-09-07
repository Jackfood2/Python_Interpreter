import base64
import ctypes
import contextlib
import traceback
import io
import json
import os
import platform
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
import webbrowser
from pathlib import Path
from tkinter import (END, DISABLED, NORMAL, Button, Entry, Frame, Label,
                     Listbox, Menu, Spinbox, Toplevel, scrolledtext,
                     simpledialog, messagebox)
from tkinter import ttk
import tkinter as tk
import chromadb
import requests
from chromadb.utils import embedding_functions
from PIL import Image, ImageGrab, ImageTk

# --- Global Configuration ---
# Note: Storing API keys directly in code is not recommended for production.
# Consider using environment variables or a secure configuration file.
OPENROUTER_API_KEY = "skxxxxxxxxxxxxxxxxxxxxxx"
API_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_ENDPOINT = f"{API_BASE_URL}/chat/completions"  # Changed from API_ENDPOINT
HTTP_REFERER = "http://localhost"
SITE_TITLE = "AGI-like Orchestrator"
DATA_DIR = Path("data")
AGENTS_FILE = DATA_DIR / "agents.json"
CORE_AGENTS_CONFIG_FILE = DATA_DIR / "core_agents_config.json"
MEMORY_DIR = DATA_DIR / "memory"
LMSTUDIO_API_BASE_URL = "http://127.0.0.1:1234/v1"
LMSTUDIO_API_ENDPOINT = f"{LMSTUDIO_API_BASE_URL}/chat/completions"
LMSTUDIO_MODEL_ID = "nvidia/nvidia-nemotron-nano-9b-v2"

# --- Core Application Logic (Largely Unchanged, but included for completeness) ---

class ConfigManager:
    """Manages loading and saving of agent configurations."""
    def __init__(self):
        DATA_DIR.mkdir(exist_ok=True)
        MEMORY_DIR.mkdir(exist_ok=True)
        self.initialize_user_agents()
        self.initialize_core_agents_config()

    def initialize_user_agents(self):
        if not AGENTS_FILE.exists():
            default_agents = [
                {"name": "PythonAgent",
                 "description": "Expert in Python scripting for automation, data processing, and web interaction using libraries like Selenium and BeautifulSoup.",
                 "llm_id": "openrouter/sonoma-sky-alpha",
                 "system_prompt": "You are an expert Python developer. Write clean, efficient, and well-documented Python code."},
                {"name": "ResearchAgent",
                 "description": "Skilled at using the browser tool to find information and answer questions.",
                 "llm_id": "openrouter/sonoma-sky-alpha",
                 "system_prompt": "You are a world-class research assistant. Find the most accurate and relevant information to answer the user's query."},
            ]
            self.save_json(AGENTS_FILE, default_agents)

    def initialize_core_agents_config(self):
        if not CORE_AGENTS_CONFIG_FILE.exists():
            default_config = {
                "ROUTER": {
                    "llm_id": "openrouter/sonoma-sky-alpha",
                    "prompt": "You are a silent agent router. Respond ONLY with a JSON object. Example: {\"ranked_agents\": [\"PythonAgent\"]}"
                },
                "PLANNER": {
                    "llm_id": "openrouter/sonoma-sky-alpha",
                    "prompt": """You are a meticulous, reactive, and persistent AGI Planner. Your job is to devise the single best next step to achieve a goal, adapting your plan based on the outcome of the previous step.

**YOUR CAPABILITIES & TOOLS:**
You have full autonomy to select the most logical and efficient tool for each step. Your primary responsibility is to analyze the task and choose the best approach from your available capabilities.
*   **Direct Answer (`direct_answer`):** Your primary tool for summarization, answering direct questions, creative writing, or any task that requires a text-based response without interacting with the OS or external tools. The "command" for this type should be the full, complete answer you want to provide to the user.
*   **OS Shell Access (`cmd`, `powershell`):** For all command-line operations. Use `cmd` for basic tasks (file listing, moving files) and `powershell` for advanced Windows scripting, system management, or complex commands that require more power.
*   **Advanced Scripting & Automation (`python_script`):** Your most versatile tool for complex logic, file creation/manipulation, data processing, API calls, and high-level, structured web automation using libraries like Selenium. If a task requires logic, loops, or external libraries, this is the correct choice.
*   **Direct UI Control (`pyautogui_script`):** For direct and complete control of the user interface (mouse and keyboard). Use this to automate any desktop application, perform actions in web browsers that Selenium cannot reliably handle, type text, press keys (`enter`, `tab`), and execute hotkeys (`ctrl+s`). This tool acts as your hands.
*   **Visual Analysis (`screenshot`):** To capture the screen's current state. Use this when you need to "see" the screen to understand the context, verify an outcome, locate an element for a future `pyautogui_script` step, or debug a failed UI interaction.
*   **Simple Web Navigation (`browser`):** For simply opening a URL in the user's default browser. Use this *only* when you need to navigate to a page without requiring any further automated interaction on that page.

**Tool Selection Philosophy:**
- **Prioritize `direct_answer`:** If the user's request can be fulfilled with a text response, use `direct_answer`. Only use other tools when system interaction is required.
- **Be Pragmatic:** Choose the simplest, most direct tool that will reliably accomplish the sub-task. Switch if the direct tool is not able to accomplish sub-task.
- **Differentiate:** Understand if a task can be done in the background via the shell/script or if it requires direct interaction with the GUI.
- **Chain Your Actions:** You can and should sequence tools logically. For example, use a `python_script` to prepare data, then a `powershell` command to use that data to perform a system action.
## ENHANCEMENT ##
**CRITICAL INSTRUCTIONS FOR UI AUTOMATION:**

**1. Selenium Best Practices & Pausing:**
   - Selenium always opens a new, clean browser session. For tasks requiring login, you must handle it within a single script.
   - **MANDATORY:** Any script performing UI automation that needs to be verified **MUST** end with `input("Pause for verification...")` to keep the window open for the screenshot.

**2. State-Aware Reactive Planning & Fallback Strategy:**
   - Your primary goal is to react to the `last_outcome`. **Do not restart the plan from the beginning.** Your next step should be a direct response to the success or failure of the previous one.
   - **If a Selenium step fails (e.g., `TimeoutException`):**
     a. **Analyze the error and the screenshot.** The verifier provides a screenshot in the log.
     b. **First Fallback (Better Selector):** Try the action again with a more robust Selenium selector. Instead of a complex XPath, try finding the element by its visible text (e.g., `By.XPATH, "//*[text()='Send']"`).
     c. **Second Fallback (PyAutoGUI):** If robust selectors also fail, use a `pyautogui_script`. You will need to estimate the (x, y) coordinates of the button you want to click based on the screenshot from the previous failed step.

**EXAMPLE `pyautogui_script`:**

import pyautogui
import time

# Failsafe: moving the mouse to the top-left corner will stop the script.
pyautogui.FAILSAFE = True

# Coordinates estimated from the screenshot in the previous step's log
x_coord = 1250
y_coord = 850

print(f"Moving mouse to ({x_coord}, {y_coord}) and clicking.")
pyautogui.moveTo(x_coord, y_coord, duration=0.5)
pyautogui.click()

time.sleep(1) # Wait a moment for the UI to react
print("Click action performed.")

YOUR CORE LOGIC:
ANALYZE THE LAST ACTION'S OUTCOME FIRST. This is your new starting point. Your goal is to make progress from this specific state.
FORMULATE THE NEXT STEP. If the last step failed, devise a recovery action (e.g., try a new selector, switch to pyautogui). If it succeeded, devise the next logical step toward the goal.
ASSESS COMPLETION. Set "is_conclusive": true ONLY if you are absolutely certain this step will fully achieve the user's ultimate goal.
RESPONSE FORMAT: Respond ONLY with a single, valid JSON object.
{
    "thought": "Your detailed reasoning. First, I will analyze the last action's outcome. Based on that success/failure, I will devise the next step. If Selenium failed, I will explain my fallback strategy (e.g., trying a new selector or switching to pyautogui with estimated coordinates).",
    "step": {
        "description": "A clear, concise description of this single action.",
        "type": "cmd | powershell | python_script | browser | pyautogui_script",
        "command": "The exact and complete command to execute."
    },
    "is_conclusive": false
}"""
                },
                "SYNTHESIZER": {"llm_id": "openrouter/sonoma-sky-alpha", "prompt": "You are a Manager Agent. Synthesize the provided agent outputs into a single, conclusive answer for the user."},
                "VERIFIER": {"llm_id": "openrouter/sonoma-sky-alpha", "prompt": """You are an expert AGI Verifier. Your role is to determine if a given action was successful based on its command output.
You will be given the user's overall goal, the description of the specific action taken, and the output (stdout, stderr, exit code) from the command.

Analyze the provided information to determine the outcome.

**Your decision process:**
1.  **Check the Exit Code:** A non-zero exit code (`code` != 0) almost always indicates a failure.
2.  **Analyze `stderr`:** The presence of errors, exceptions, or "command not found" messages in stderr is a strong indicator of failure.
3.  **Analyze `stdout`:** Does the output in stdout align with the expected result of the action? For example, if the action was to 'list files', does stdout contain a file listing?
4.  **Identify Uncertainty:** If the command output is empty or provides no useful information to confirm success or failure (e.g., for UI actions that don't produce text output), you MUST classify the outcome as 'UNCERTAIN'. This will trigger a visual verification.

**Response Format:**
Respond ONLY with a single, valid JSON object with 'status' and 'reasoning'.
- `status`: Must be one of "PASSED", "FAILED", or "UNCERTAIN".
- `reasoning`: A brief explanation for your decision."""},
                "QUESTION_FORMULATOR": {"llm_id": "openrouter/sonoma-sky-alpha", "prompt": """You are an expert at creating verification questions. Based on the user's goal and the specific action an agent took, generate a short list of simple, factual, yes/no questions that can be answered by looking at a screenshot. If the answer to all questions is 'yes', the action was successful.
Respond ONLY with a valid JSON object containing a list of strings. Example:
{
"questions": [
"Is the 'File Explorer' window open?",
"Is the file named 'report.docx' visible and selected?"
]
}"""},
                "ARCHITECT": {"llm_id": "openrouter/sonoma-sky-alpha", "prompt": ""},
                "IMAGE_DESCRIBER": {"llm_id": "openrouter/sonoma-sky-alpha"},
            }
            self.save_json(CORE_AGENTS_CONFIG_FILE, default_config)

    def load_json(self, file_path, default=None):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return default if default is not None else {}

    def save_json(self, file_path, data):
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)

    def load_user_agents(self): return self.load_json(AGENTS_FILE, [])
    def save_user_agents(self, agents): self.save_json(AGENTS_FILE, agents)
    def load_core_agents_config(self): return self.load_json(CORE_AGENTS_CONFIG_FILE)
    def save_core_agents_config(self, config): self.save_json(CORE_AGENTS_CONFIG_FILE, config)

class MemoryManager:
    """Handles the agent's long-term memory using ChromaDB and file storage."""
    def __init__(self, log_callback):
        self.log_callback = log_callback
        self.client = chromadb.PersistentClient(path=str(MEMORY_DIR / "chroma_db"))
        st_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        self.procedural = self.client.get_or_create_collection(name="procedural_memory", embedding_function=st_ef)
        self.semantic = self.client.get_or_create_collection(name="semantic_memory", embedding_function=st_ef)
        self.episodic_path = MEMORY_DIR / "episodes"
        self.episodic_path.mkdir(exist_ok=True)

    def save_episode(self, goal, transcript):
        timestamp = int(time.time())
        episode_id = f"episode_{timestamp}"
        episode_data = {"goal": goal, "transcript": transcript}
        file_path = self.episodic_path / f"{episode_id}.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(episode_data, f, indent=2)
        return episode_id

    def add_procedure(self, description, steps, source_episode):
        doc_id = f"proc_{int(time.time())}_{hash(description)}"
        self.procedural.add(documents=[description], metadatas=[{"steps": json.dumps(steps), "source_episode": source_episode}], ids=[doc_id])
        self.log_callback(f"[Memory] New procedure learned: {description}", "success")

    def add_fact(self, fact, source_episode):
        doc_id = f"fact_{int(time.time())}_{hash(fact)}"
        self.semantic.add(documents=[fact], metadatas=[{"source_episode": source_episode}], ids=[doc_id])
        self.log_callback(f"[Memory] New fact learned: {fact}", "success")

    def search_procedures(self, query, n_results=3):
        return self.procedural.query(query_texts=[query], n_results=n_results) if self.procedural.count() > 0 else None

    def search_facts(self, query, n_results=5):
        return self.semantic.query(query_texts=[query], n_results=n_results) if self.semantic.count() > 0 else None

class CognitiveArchitect:
    """Reflects on completed tasks to generate new memories."""
    def __init__(self, memory_manager, log_callback, config):
        self.memory = memory_manager
        self.log_callback = log_callback
        self.config = config

    def reflect_on_episode(self, goal, transcript, episode_id, provider):
        self.log_callback("\n[CognitiveArchitect] Reflecting on the completed task...", "header")
        summary = f"Goal: {goal}\n\nTranscript of Actions and Outcomes:\n"
        for entry in transcript:
            step_info = entry.get('step')
            description = step_info.get('description', 'N/A') if step_info else "N/A"
            summary += f"- Step: {description}\n  - Outcome: {json.dumps(entry.get('outcome', {}))}\n"

        system_prompt = """You are a Cognitive Architect. Your purpose is to analyze an AI agent's performance on a task and distill generalizable knowledge. From the provided goal and transcript, extract Procedures (reusable, successful command sequences) and Facts (new, useful information). Respond ONLY with a valid JSON object. Use empty lists [] if nothing was learned. Example: {"procedures": [{"description": "Title", "steps": ["cmd1", "cmd2"]}], "facts": ["A distilled fact."]}"""
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": summary}]
        # --- FIX: Pass the provider argument ---
        response = query_llm(messages, self.config['llm_id'], self.log_callback, threading.Event(), provider)
        if not response: return

        try:
            match = re.search(r'{.*}', response, re.DOTALL)
            if not match:
                self.log_callback(f"[CognitiveArchitect] Could not find JSON in reflection response.", "warning")
                return
            reflections = json.loads(match.group(0))
            if reflections.get("procedures") or reflections.get("facts"):
                self.log_callback(f"\n[CognitiveArchitect] Generated the following memories for storage:", "info")
                # Pretty-print the JSON to make it readable in the log
                pretty_reflections = json.dumps(reflections, indent=2)
                self.log_callback(pretty_reflections, "output") # Use 'output' tag for monospaced font
            for proc in reflections.get("procedures", []): self.memory.add_procedure(proc['description'], proc['steps'], episode_id)
            for fact in reflections.get("facts", []): self.memory.add_fact(fact, episode_id)
            self.log_callback("[CognitiveArchitect] Reflection complete.", "info")
        except Exception as e:
            self.log_callback(f"[CognitiveArchitect] Error during reflection: {e}", "error")

class StepVerifier:
    """Verifies the outcome of an agent's action, using text, vision, or human input."""
    def __init__(self, log_callback, user_prompt, core_agents_config, question_queue, answer_queue, provider):
        self.log_callback = log_callback
        self.user_prompt = user_prompt
        self.core_agents_config = core_agents_config
        self.question_queue = question_queue
        self.answer_queue = answer_queue
        self.provider = provider # --- Store the provider

    def log(self, message, tag=None): self.log_callback(message, tag)

    def _verify_with_text(self, step_description, command_output, cancel_event):
        self.log("[Verifier] Attempting text-based verification...", "info")
        config = self.core_agents_config.get("VERIFIER", {})
        prompt = f'**Goal:** "{self.user_prompt}"\n**Action:** "{step_description}"\n**Output:**\n{json.dumps(command_output)}'
        messages = [{"role": "system", "content": config.get("prompt")}, {"role": "user", "content": prompt}]
        # --- FIX: Pass self.provider to the LLM query ---
        response = query_llm(messages, config.get("llm_id"), self.log_callback, cancel_event, self.provider)
        if not response: return None
        try:
            match = re.search(r'{.*}', response, re.DOTALL)
            if not match: return None
            result = json.loads(match.group(0))
            if result.get("status", "").upper() in ["PASSED", "FAILED", "UNCERTAIN"]:
                self.log(f"[Verifier] Text-based result: {result.get('status')}. Reason: {result.get('reasoning', 'N/A')}", "info")
                return result
            return None
        except (json.JSONDecodeError, AttributeError) as e:
            self.log(f"[Verifier] Failed to parse text-based verification response: {e}", "error")
            return None

    def _verify_with_vision(self, step_description, cancel_event):
        self.log("[Verifier] Falling back to vision-based verification.", "info")
        questions = self._generate_verification_questions(step_description, cancel_event)
        if not questions: return {"status": "UNCERTAIN", "reasoning": "Could not generate verification questions."}

        self.log_callback("HIDE_WINDOWS_FOR_SCREENSHOT", "system_command")
        time.sleep(1.5)

        if cancel_event.is_set():
            self.log_callback("SHOW_WINDOWS_AFTER_SCREENSHOT", "system_command")
            return {"status": "FAILED", "reasoning": "Task cancelled during verification."}

        screenshot = ImageGrab.grab()
        self.log_callback("SHOW_WINDOWS_AFTER_SCREENSHOT", "system_command")
        self.log(screenshot, "image_display")

        buffered = io.BytesIO()
        screenshot.save(buffered, format="JPEG"); img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        del screenshot, buffered

        vision_prompt = f'Based on the image, answer each question with only "yes", "no", or "uncertain".\nQuestions:\n{json.dumps(questions, indent=2)}\nRespond ONLY with a JSON object like: {{"answers": ["yes", "no"]}}'
        config = self.core_agents_config.get("IMAGE_DESCRIBER", {})
        response_text = query_vision_llm(vision_prompt, img_base64, config.get("llm_id"), self.log_callback, cancel_event, self.provider)
        if not response_text: return {"status": "FAILED", "reasoning": "Vision model failed to respond."}

        try:
            match = re.search(r'{.*}', response_text, re.DOTALL)
            answers = json.loads(match.group(0)).get("answers", []) if match else []
        except (json.JSONDecodeError, AttributeError): answers = []

        if len(answers) != len(questions): return {"status": "UNCERTAIN", "reasoning": "Vision model returned malformed answers."}

        final_reasoning = "[Vision-based] Q&A:\n"
        for q, a in zip(questions, answers):
            answer_str = str(a).lower().strip()
            final_reasoning += f"- Q: {q} -> A: {answer_str.upper()}\n"
            if answer_str == 'no': return {"status": "FAILED", "reasoning": final_reasoning}
            if answer_str == 'uncertain':
                human_answer = self._request_human_verification(q)
                final_reasoning += f"  - User Confirmation -> {'YES' if human_answer else 'NO'}\n"
                if not human_answer: return {"status": "FAILED", "reasoning": final_reasoning}
        return {"status": "PASSED", "reasoning": final_reasoning}

    def _generate_verification_questions(self, step_description, cancel_event):
        config = self.core_agents_config.get("QUESTION_FORMULATOR", {})
        prompt = f'Goal: "{self.user_prompt}"\nAction: "{step_description}"'
        messages = [{"role": "system", "content": config.get("prompt")}, {"role": "user", "content": prompt}]
        # --- FIX: Pass self.provider to the LLM query ---
        response = query_llm(messages, config.get("llm_id"), self.log_callback, cancel_event, self.provider)
        if not response: return []
        try:
            match = re.search(r'{.*}', response, re.DOTALL)
            return json.loads(match.group(0)).get("questions", []) if match else []
        except (json.JSONDecodeError, AttributeError): return []

    def _request_human_verification(self, question):
        self.log(f"[Verifier] Asking user for help: '{question}'", "warning")
        self.question_queue.put(question)
        while self.answer_queue.empty(): time.sleep(0.2)
        return self.answer_queue.get()

    def verify_step_outcome(self, step_description, command_output, cancel_event):
        text_result = self._verify_with_text(step_description, command_output, cancel_event)
        if text_result and text_result.get('status') in ['PASSED', 'FAILED']:
            return text_result
        return self._verify_with_vision(step_description, cancel_event)

def query_llm(messages, model_id, log_callback, cancel_event, provider, max_tokens=10000):
    log_callback(f"\n[LLM:{model_id} via {provider.upper()}] Querying...", "info")

    if provider == "lmstudio":
        url = LMSTUDIO_API_ENDPOINT
        headers = {"Content-Type": "application/json"}
        data = {"model": LMSTUDIO_MODEL_ID, "messages": messages, "temperature": 0.2, "max_tokens": max_tokens, "stream": True}
    else: # Default to OpenRouter
        url = OPENROUTER_API_ENDPOINT
        headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json", "HTTP-Referer": HTTP_REFERER, "X-Title": SITE_TITLE}
        data = {"model": model_id, "messages": messages, "temperature": 0.2, "max_tokens": max_tokens, "stream": True}

    response_text = ""
    
    for attempt in range(30): 
        if cancel_event.is_set():
            log_callback("\n[LLM] Task cancelled before request.", "warning")
            return None
        try:
            with requests.post(url, headers=headers, json=data, timeout=10, stream=True) as resp:
                resp.raise_for_status()
                if resp.status_code != 200:
                    log_callback(f"[LLM] Unexpected status code: {resp.status_code}", "error")
                    log_callback(f"[LLM] Response: {resp.text}", "error")
                log_callback("", "llm_stream_start")
                
                stream_completed = True  # Track if stream completed normally
                
                for chunk in resp.iter_lines():
                    if cancel_event.is_set():
                        log_callback("\n[LLM] Stream cancelled.", "warning")
                        stream_completed = False
                        break 
                    if chunk and (line := chunk.decode('utf-8').strip()).startswith('data: ') and line != 'data: [DONE]':
                        try:
                            content = json.loads(line[6:])["choices"][0]["delta"].get("content", "")
                            if content: 
                                response_text += content
                                log_callback(content, "llm_stream")
                        except (json.JSONDecodeError, KeyError): 
                            continue
                
                log_callback("", "llm_stream_end")
                
                # Only return if we successfully completed the stream
                if stream_completed:
                    return response_text
                else:
                    return None  # Cancelled

        except requests.exceptions.Timeout:
            log_callback(f"[LLM] Request timed out, retrying... ({attempt+1}/30)", "warning")
            continue # Go to the next attempt
        except requests.exceptions.RequestException as e:
            log_callback(f"\n[LLM ERROR on {provider.upper()}] {e}", "error")
            return None
    
    # This part is reached if all retry attempts time out
    log_callback(f"\n[LLM ERROR on {provider.upper()}] Request failed after multiple timeouts.", "error")
    return None

def query_vision_llm(prompt, base64_image, model_id, log_callback, cancel_event, provider, max_tokens=1024):
    log_callback(f"\n[VisionLLM:{model_id} via {provider.upper()}] Querying with image...", "info")
    if cancel_event.is_set(): return None

    payload = {"model": model_id, "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}]}], "max_tokens": max_tokens, "temperature": 0.2}

    if provider == "lmstudio":
        url = LMSTUDIO_API_ENDPOINT
        headers = {"Content-Type": "application/json"}
        # Use the placeholder model ID for LM Studio
        payload["model"] = LMSTUDIO_MODEL_ID
    else: # Default to OpenRouter
        url = OPENROUTER_API_ENDPOINT
        headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json", "HTTP-Referer": HTTP_REFERER, "X-Title": SITE_TITLE}

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=180)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e:
        log_callback(f"\n[LLM ERROR on {provider.upper()}] Vision model API error: {e}", "error")
    except (KeyError, IndexError):
        log_callback(f"\n[LLM ERROR] Could not parse vision model response.", "error")
    return None

class Agent:
    """Represents an autonomous agent that can plan and execute steps to achieve a goal."""
    def __init__(self, name, description, llm_id, system_prompt, log_callback, config_manager, memory_manager, question_queue, answer_queue, subprocess_list):
        self.name, self.description, self.llm_id, self.system_prompt = name, description, llm_id, system_prompt
        self.log_callback, self.config_manager, self.memory_manager = log_callback, config_manager, memory_manager
        self.core_agents_config = config_manager.load_core_agents_config()
        self.temp_scripts_dir = Path(tempfile.gettempdir()) / 'multi_agent_scripts'
        self.temp_scripts_dir.mkdir(exist_ok=True)
        self.user_prompt, self.question_queue, self.answer_queue = None, question_queue, answer_queue
        self.active_subprocesses = subprocess_list

    def log(self, message, tag=None): self.log_callback(f"[{self.name}] {message}", tag)

    def generate_next_step(self, prompt, history, last_outcome, cancel_event, provider):
        # Memory retrieval logic (unchanged)
        relevant_procs = self.memory_manager.search_procedures(prompt)
        relevant_facts = self.memory_manager.search_facts(prompt)
        memory_context = "## Relevant Knowledge from Memory:\n"
        if relevant_procs and relevant_procs.get('documents'):
            for doc, meta in zip(relevant_procs['documents'][0], relevant_procs['metadatas'][0]): memory_context += f"- Procedure: {doc}\n  - Steps: {meta['steps']}\n"
        if relevant_facts and relevant_facts.get('documents'):
            for doc in relevant_facts['documents'][0]: memory_context += f"- Fact: {doc}\n"
        if "Procedure:" not in memory_context and "Fact:" not in memory_context: memory_context += "None\n"

        history_ctx = "## History:\n" + ("\n".join([f"Step {i + 1}: {json.dumps(h, indent=2)}" for i, h in enumerate(history)]) if history else "None")
        planner_core_prompt = self.core_agents_config.get("PLANNER", {}).get('prompt', '')

        sys_prompt = f"{self.system_prompt}\n\n---\n\n{planner_core_prompt}" if self.system_prompt else planner_core_prompt
        last_outcome_str = json.dumps(last_outcome, indent=2) if last_outcome else "N/A (This is the first step)"
        user_prompt_content = f"{memory_context}\n---\n**Ultimate Goal:** \"{prompt}\"\n{history_ctx}\n**Last Action's Outcome:**\n{last_outcome_str}\n---\nDetermine the single next step."

        messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt_content}]
        response = query_llm(messages, self.llm_id, self.log_callback, cancel_event, provider)
        if response and (match := re.search(r'{.*}', response, re.DOTALL)):
            try: return json.loads(match.group(0))
            except json.JSONDecodeError as e: self.log(f"Failed to parse plan: {e}\nResponse: {response}", "error")
        return None

    def execute_step(self, step_data):
        description, step_type, command = step_data.get("description", "N/A"), step_data.get("type"), step_data.get("command", "")
        self.log(f"Executing: {description}", "info")

        # --- In-Process Execution for Standard Python Scripts ---
        if step_type == "python_script":
            self.log("[Execution] Running script in-process...", "info")
            stdout_buffer = io.StringIO()
            stderr_buffer = io.StringIO()
            return_code = 0
            
            try:
                # Redirect stdout and stderr to capture all output
                with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
                    # Use a restricted globals dict for a little extra safety
                    restricted_globals = {"__builtins__": __builtins__}
                    exec(command, restricted_globals)
            except Exception:
                # If any error occurs, capture the full traceback
                # and set a non-zero exit code.
                stderr_buffer.write(traceback.format_exc())
                return_code = 1
            finally:
                # Get the captured output
                stdout = stdout_buffer.getvalue()
                stderr = stderr_buffer.getvalue()
                stdout_buffer.close()
                stderr_buffer.close()

            return {"stdout": stdout, "stderr": stderr, "code": return_code}

        # --- Subprocess Execution for Shell Commands and UI Automation ---
        flags = subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0
        try:
            if step_type in ["cmd", "powershell"]:
                is_shell = True if step_type == "cmd" else False
                cmd_list = ['powershell.exe', '-NoProfile', '-Command', command] if step_type == "powershell" else command
                
                # Use Popen and communicate() to prevent I/O deadlocks
                proc = subprocess.Popen(
                    cmd_list, 
                    shell=is_shell, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE, 
                    text=True, 
                    encoding='utf-8', 
                    errors='ignore', 
                    creationflags=flags
                )
                
                try:
                    # communicate() safely reads output and waits for process to end
                    stdout, stderr = proc.communicate(timeout=300) # 5-minute timeout
                    return_code = proc.returncode
                except subprocess.TimeoutExpired:
                    # If the command genuinely hangs, kill it and report an error
                    proc.kill()
                    # Try to get any output it produced before being killed
                    stdout, stderr = proc.communicate()
                    stderr += "\n[Execution Error] Command timed out after 5 minutes and was terminated."
                    return_code = 1 # A non-zero code indicates failure

                return {"stdout": stdout, "stderr": stderr, "code": return_code}
            
            # PyAutoGUI scripts still run in a separate process
            elif step_type == "pyautogui_script":
                script_path = self.temp_scripts_dir / f"agent_script_{int(time.time())}.py"
                script_path.write_text(command.replace('\\n', '\n'), encoding='utf-8')
                proc = subprocess.Popen([sys.executable, str(script_path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='ignore', creationflags=flags)
                self.active_subprocesses.append(proc)
                try:
                    stdout, stderr = proc.communicate(timeout=300)
                    return {"stdout": stdout, "stderr": stderr, "code": proc.returncode}
                except subprocess.TimeoutExpired:
                    proc.kill()
                    self.log("Script execution timed out after 5 minutes and was terminated.", "error")
                    return {"stdout": "", "stderr": "TimeoutExpired: The script took too long to execute.", "code": 1}
                finally:
                    if proc.poll() is None:
                        try:
                            proc.kill()
                            proc.wait(timeout=5)
                        except (ProcessLookupError, subprocess.TimeoutExpired): pass
                    if proc in self.active_subprocesses: self.active_subprocesses.remove(proc)

            elif step_type == "screenshot":
                self.log_callback("HIDE_WINDOWS_FOR_SCREENSHOT", "system_command")
                time.sleep(1.5)
                screenshot = ImageGrab.grab()
                self.log_callback("SHOW_WINDOWS_AFTER_SCREENSHOT", "system_command")
                self.log(screenshot, "image_display")
                return {"stdout": f"Screenshot taken. Reason: {command}", "stderr": "", "code": 0}

            elif step_type == "browser":
                webbrowser.open(command)
                return {"stdout": f"Opened URL: {command}", "stderr": "", "code": 0}
            
            else: 
                return {"stdout": "", "stderr": f"Unknown step type '{step_type}'", "code": 1}

        except Exception as e:
            return {"stdout": "", "stderr": f"Critical execution error: {str(e)}", "code": 1}

    def run(self, prompt, cancel_event, provider):
        self.user_prompt = prompt
        history, last_outcome = [], None
        step_count, max_steps = 0, 15
        # THIS IS THE KEY FIX: Pass 'provider' to the verifier's constructor
        verifier = StepVerifier(self.log_callback, prompt, self.core_agents_config, self.question_queue, self.answer_queue, provider)
    
        while step_count < max_steps:
            if cancel_event.is_set():
                return {"success": False, "error": "Task cancelled.", "history": history}
            step_count += 1
            self.log(f"\n--- Step {step_count}/{max_steps} ---", "header")
    
            # FIX #2: Pass 'provider' to the planner
            plan = self.generate_next_step(prompt, history, last_outcome, cancel_event, provider)
            if not plan or "step" not in plan:
                last_outcome = {"status": "FAILED", "reasoning": "Planner failed to produce a valid plan."}
                history.append({"step": plan, "outcome": last_outcome})
                continue
    
            if plan.get('step', {}).get('type') == 'direct_answer':
                self.log("Planner chose a direct answer.", "success")
                final_answer = plan['step'].get('command', 'No answer provided.')
                history.append({"step": plan.get("step"), "outcome": {"status": "PASSED", "reasoning": "Direct answer provided."}})
                return {"success": True, "output": final_answer, "history": history}
    
            exec_result = self.execute_step(plan.get("step", {}))

            # --- NEW: Improved Output Logging ---
            stdout = exec_result.get('stdout', '').strip()
            stderr = exec_result.get('stderr', '').strip()

            if stdout or stderr:
                self.log("--- PROCESS OUTPUT ---", "output")
                if stdout:
                    # Log standard output
                    self.log(stdout, "output")
                if stderr:
                    # Log standard error with a distinct error tag
                    self.log(f"[STDERR]:\n{stderr}", "error")
                self.log("--- END OUTPUT ---", "output")
            else:
                # If there's no output, confirm that it ran silently.
                self.log("[OUTPUT] Command produced no output.", "info")
            # --- END of NEW BLOCK ---
    
            if exec_result["code"] != 0:
                last_outcome = {"status": "FAILED", "reasoning": f"Execution failed with code {exec_result['code']}.", "output": exec_result}
    
            if exec_result["code"] != 0 or stderr:
                # Construct a more detailed reason for the failure.
                reasoning = ""
                if exec_result["code"] != 0:
                    reasoning += f"Execution failed with a non-zero exit code ({exec_result['code']}).\n"
                if stderr:
                    reasoning += f"The command produced the following error output:\n{stderr}"

                last_outcome = {
                    "status": "FAILED",
                    "reasoning": reasoning.strip(),
                    "output": exec_result
                }
                history.append({"step": plan.get("step"), "outcome": last_outcome})
                # This 'continue' is crucial - it forces the agent to stop and replan.
                continue
    
            if plan.get("is_conclusive"):
                last_outcome = {"status": "PASSED", "reasoning": "Marked as conclusive by planner.", "output": exec_result}
                history.append({"step": plan.get("step"), "outcome": last_outcome})
                break
    
            verification = verifier.verify_step_outcome(plan['step']['description'], exec_result, cancel_event)
            last_outcome = {"status": verification.get("status", "FAILED"), "reasoning": verification.get("reasoning", "N/A"), "output": exec_result}
            history.append({"step": plan.get("step"), "outcome": last_outcome})
    
        final_result = history[-1] if history else {}
        is_success = final_result.get("outcome", {}).get("status") == "PASSED"
        if step_count >= max_steps:
            return {"success": False, "error": f"Max steps ({max_steps}) reached.", "history": history}
        return {"success": is_success, "output": json.dumps(final_result, indent=2), "history": history}

class AgentManager:
    """Manages the lifecycle of agents, including selection, execution, and synthesis."""
    def __init__(self, log_callback, on_finish_callback, question_queue, answer_queue):
        self.log_callback = log_callback
        self.on_finish_callback = on_finish_callback
        self.config_manager = ConfigManager()
        self.memory_manager = MemoryManager(log_callback)
        self.architect = CognitiveArchitect(self.memory_manager, log_callback, self.config_manager.load_core_agents_config().get("ARCHITECT", {}))
        self.cancel_event = threading.Event()
        self.active_subprocesses = []
        self.reload_agents(question_queue, answer_queue)

    def log(self, message, tag=None): self.log_callback(message, tag)

    def reload_agents(self, question_queue, answer_queue):
        self.user_agents_data = self.config_manager.load_user_agents()
        self.core_agents_config = self.config_manager.load_core_agents_config()
        self.agents = [Agent(a['name'], a['description'], a.get('llm_id'), a.get('system_prompt', ''), self.log_callback, self.config_manager, self.memory_manager, question_queue, answer_queue, self.active_subprocesses) for a in self.user_agents_data]
        self.log("Agents and configurations reloaded.", "info")

    def cleanup_processes(self):
        if not self.active_subprocesses: return
        self.log(f"Cleaning up {len(self.active_subprocesses)} active subprocess(es)...", "info")
        for proc in self.active_subprocesses[:]:
            if proc.poll() is None:
                try: proc.kill(); self.log(f"Force-killed process {proc.pid}.", "warning")
                except Exception as e: self.log(f"Failed to kill process {proc.pid}: {e}", "error")
        self.active_subprocesses.clear()

    def select_agents(self, prompt, num_to_select, max_tokens, provider):
        # This logic is sound and remains unchanged.
        self.log(f"\n[Router] Selecting the best {num_to_select} agent(s)...", "header")
        if not self.agents: return []
        agent_list = "\n".join([f"- {a.name}: {a.description}" for a in self.agents])
        router_cfg = self.core_agents_config.get("ROUTER", {})
        user_prompt = f'User Prompt: "{prompt}"\nAgents:\n{agent_list}\nTask: Rank and select exactly {num_to_select} agent(s).'
        messages = [{"role": "system", "content": router_cfg.get("prompt", "")}, {"role": "user", "content": user_prompt}]
        response = query_llm(messages, router_cfg.get("llm_id"), self.log_callback, self.cancel_event, provider, max_tokens=256)
        if not response: return []
        try:
            match = re.search(r'{.*}', response, re.DOTALL)
            if not match: return []
            names = json.loads(match.group(0)).get("ranked_agents", [])
            self.log(f"[Router] Selected agents: {names}", "info")
            return [agent for name in names for agent in self.agents if agent.name == name][:num_to_select]
        except json.JSONDecodeError: return []
        
    def synthesize_results(self, prompt, agent_outputs, max_tokens, provider):
        self.log("\n[Synthesizer] Synthesizing final answer...", "header")
        inputs = "\n".join([f"--- Output from {name} ---\n{res.get('output', 'N/A')}" for name, res in agent_outputs.items()])
        config = self.core_agents_config.get("SYNTHESIZER", {})
        user_prompt = f'Original Prompt: "{prompt}"\nAgent Outputs:\n{inputs}'
        messages = [{"role": "system", "content": config.get("prompt")}, {"role": "user", "content": user_prompt}]
        # --- FIX: Pass the provider argument ---
        return query_llm(messages, config.get("llm_id"), self.log_callback, self.cancel_event, provider, max_tokens=max_tokens) or "Synthesizer failed."

    def execute_task(self, prompt, num_agents_to_activate, max_tokens, provider):
        """
        Public method called by the GUI. Starts the entire task in a background thread
        to keep the GUI responsive.
        """
        # The GUI now calls this method, which immediately returns after starting the thread.
        threading.Thread(
            target=self._execute_task_thread, 
            args=(prompt, num_agents_to_activate, max_tokens, provider), 
            daemon=True
        ).start()

    def _execute_task_thread(self, prompt, num_agents_to_activate, max_tokens, provider):
        """
        The actual task execution logic, running in a background thread.
        """
        self.cancel_event.clear()
        self.cleanup_processes()
        self.log("\n[Manager] Starting new task...", "header")
        
        active_agents = self.select_agents(prompt, num_agents_to_activate, max_tokens, provider)
        if not active_agents:
            self.log("[Manager] No suitable agents were selected. Aborting.", "error")
            self.on_finish_callback({"final_answer": "Task aborted: No suitable agents found."})
            return

        self.log(f"\n[Manager] Activating: {[a.name for a in active_agents]}", "success")
        agent_outputs, threads = {}, []
        def run_agent_and_store(agent): agent_outputs[agent.name] = agent.run(prompt, self.cancel_event, provider)
        
        for agent in active_agents:
            thread = threading.Thread(target=run_agent_and_store, args=(agent,)); threads.append(thread); thread.start()
        for thread in threads: thread.join()

        self.cleanup_processes()
        
        if self.cancel_event.is_set():
            self.on_finish_callback({"final_answer": "Task was cancelled by the user."})
            return

        transcript = [h for res in agent_outputs.values() for h in res.get('history', [])]
        if transcript:
            self.log("\n[Manager] Saving episode and starting cognitive reflection in background...", "info")
            episode_id = self.memory_manager.save_episode(prompt, transcript)
            # Reflection is still in its own daemon thread, which is fine.
            threading.Thread(target=self.architect.reflect_on_episode, args=(prompt, transcript, episode_id, provider), daemon=True).start()

        if all(result.get('success') for result in agent_outputs.values()):
            self.log("\n[Manager] All agents succeeded.", "success")
            if len(agent_outputs) == 1:
                final_answer = next(iter(agent_outputs.values())).get('output', "Task complete.")
            else:
                self.log("\n[Manager] Multiple agents succeeded. Synthesizing results...", "info")
                final_answer = self.synthesize_results(prompt, agent_outputs, max_tokens, provider)
        else:
            self.log("\n[Manager] One or more agents failed.", "warning")
            final_answer = "One or more agents failed. Review the log for details."

        self.log(f"\n{'=' * 25} TASK COMPLETE {'=' * 25}", "header")
        self.on_finish_callback({"final_answer": final_answer})

    def execute_task(self, prompt, num_agents_to_activate, max_tokens, provider):
        self.cancel_event.clear()
        self.cleanup_processes()
        self.log("\n[Manager] Starting new task...", "header")
        
        active_agents = self.select_agents(prompt, num_agents_to_activate, max_tokens, provider)
        if not active_agents:
            self.log("[Manager] No suitable agents were selected. Aborting.", "error")
            self.on_finish_callback({"final_answer": "Task aborted: No suitable agents found."})
            return

        self.log(f"\n[Manager] Activating: {[a.name for a in active_agents]}", "success")
        agent_outputs, threads = {}, []
        def run_agent_and_store(agent): agent_outputs[agent.name] = agent.run(prompt, self.cancel_event, provider)
        
        for agent in active_agents:
            thread = threading.Thread(target=run_agent_and_store, args=(agent,)); threads.append(thread); thread.start()
        for thread in threads: thread.join()

        self.cleanup_processes()
        
        if self.cancel_event.is_set():
            self.on_finish_callback({"final_answer": "Task was cancelled by the user."})
            return

        transcript = [h for res in agent_outputs.values() for h in res.get('history', [])]
        if transcript:
            episode_id = self.memory_manager.save_episode(prompt, transcript)
            threading.Thread(target=self.architect.reflect_on_episode, args=(prompt, transcript, episode_id, provider), daemon=True).start()

        if all(result.get('success') for result in agent_outputs.values()):
            self.log("\n[Manager] All agents succeeded.", "success")
            final_answer = next(iter(agent_outputs.values())).get('output', "Task complete.") if len(agent_outputs) == 1 else self.synthesize_results(prompt, agent_outputs, max_tokens, provider)
        else:
            self.log("\n[Manager] One or more agents failed.", "warning")
            final_answer = "One or more agents failed. Review the log for details."

        self.log(f"\n{'=' * 25} TASK COMPLETE {'=' * 25}", "header")
        self.on_finish_callback({"final_answer": final_answer})

    def cancel_task(self):
        """Forcefully stops the current task."""
        self.log("\n[Manager] Cancellation signal received. Terminating processes...", "warning")
        self.cancel_event.set()
        self.cleanup_processes()

class StyleManager:
    """Manages the visual theme and styles for the Tkinter application."""
    def __init__(self):
        self.colors = {"bg": "#2E3440", "bg_light": "#3B4252", "bg_lighter": "#434C5E", "fg": "#D8DEE9", "fg_muted": "#A3B3D3", "accent": "#88C0D0", "accent_active": "#8FBCBB", "green": "#A3BE8C", "red": "#BF616A", "yellow": "#EBCB8B"}
        self.fonts = {"main": ("Segoe UI", 10), "bold": ("Segoe UI", 10, "bold"), "log": ("Consolas", 10)}
        self.style = ttk.Style()
        if "clam" in self.style.theme_names(): self.style.theme_use('clam')
        self.configure_styles()

    def configure_styles(self):
        self.style.configure('.', background=self.colors["bg"], foreground=self.colors["fg"], font=self.fonts["main"])
        self.style.configure('TFrame', background=self.colors["bg"])
        self.style.configure('TLabel', background=self.colors["bg"], foreground=self.colors["fg"])
        self.style.configure('TLabelFrame.Label', background=self.colors["bg"], foreground=self.colors["accent"], font=self.fonts["bold"])
        self.style.configure('TNotebook', background=self.colors['bg'], borderwidth=0)
        self.style.configure('TNotebook.Tab', background=self.colors['bg_light'], foreground=self.colors['fg_muted'], padding=[8, 4], font=self.fonts['main'])
        self.style.map('TNotebook.Tab', background=[('selected', self.colors['accent'])], foreground=[('selected', self.colors['bg'])])
        self.configure_button_style('TButton', self.colors["bg_lighter"], self.colors["fg"], self.colors["accent"])
        self.configure_button_style('Primary.TButton', self.colors["accent"], self.colors["bg"], self.colors["accent_active"])
        self.configure_button_style('Destructive.TButton', self.colors["red"], self.colors["fg"], self.colors["red"])
        self.style.configure('TSpinbox', arrowsize=12, fieldbackground=self.colors["bg_light"], foreground=self.colors["fg"])
        self.style.configure('Horizontal.TScale', troughcolor=self.colors["bg_light"], background=self.colors["accent"])

    def configure_button_style(self, name, bg, fg, active_bg):
        self.style.configure(name, background=bg, foreground=fg, font=self.fonts["bold"], borderwidth=1, relief="raised", padding=(10, 8))
        self.style.map(name, background=[('active', active_bg)], relief=[('pressed', 'sunken')])

    def get_widget_styles(self, widget_type):
        if widget_type == "text":
            return {"background": self.colors["bg_light"], "foreground": self.colors["fg"], "font": self.fonts["log"], "relief": "flat", "borderwidth": 0, "highlightthickness": 1, "highlightbackground": self.colors["accent"], "insertbackground": self.colors["fg"]}
        if widget_type == "listbox":
            return {"background": self.colors["bg_light"], "foreground": self.colors["fg"], "font": self.fonts["main"], "relief": "flat", "borderwidth": 0, "highlightthickness": 0, "selectbackground": self.colors["accent"], "selectforeground": self.colors["bg"]}
        return {}

class ExecutionLogWindow(Toplevel):
    """A pop-up window to display the live execution log of an agent task."""
    def __init__(self, parent, cancel_callback):
        super().__init__(parent)
        self.parent = parent
        self.cancel_callback = cancel_callback
        self.is_task_running = True
        self.protocol("WM_DELETE_WINDOW", self.on_close_attempt)
        self.title("Execution Log")
        self.geometry("900x700")
        self.wm_attributes("-topmost", True)
        self.configure(bg=StyleManager().colors['bg'])
        
        self.log_text = scrolledtext.ScrolledText(self, wrap="word", state=DISABLED, **StyleManager().get_widget_styles("text"))
        self.log_text.pack(fill="both", expand=True, padx=10, pady=(10, 0))
        
        self.stop_button = ttk.Button(self, text="Stop Task", command=cancel_callback, style="Destructive.TButton")
        self.stop_button.pack(pady=10, padx=10, fill="x")

    def on_close_attempt(self):
        if self.is_task_running:
            messagebox.showwarning("Task in Progress", "Cannot close this window while a task is running. Please use the 'Stop Task' button.", parent=self)
        else:
            self.destroy()

class TaskExecutorGUI:
    """The main graphical user interface for the agent orchestrator."""
    def __init__(self, root):
        self.root = root
        self.root.title("Multi-Agent Orchestrator")
        self.root.geometry("1400x900")
        self.style_manager = StyleManager()
        self.root.configure(bg=self.style_manager.colors['bg'])

        self.is_task_running = False
        self.log_photo_images = []
        self.execution_log_window = None
        self.full_task_log = [] # To store the log for post-task review

        self.log_queue = queue.Queue()
        self.question_queue = queue.Queue()
        self.answer_queue = queue.Queue()
        self.agent_manager = AgentManager(self.log_to_gui, self.on_task_finish, self.question_queue, self.answer_queue)
        self.provider_var = tk.StringVar(value="openrouter")

        self.create_widgets()
        self.populate_agent_list()
        self.process_log_queue()

    def create_widgets(self):
        self.create_menu()
        main_pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_pane.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Left Pane: Agent Management
        agent_panel = ttk.LabelFrame(main_pane, text="Agent Management", padding=10)
        main_pane.add(agent_panel, weight=1)
        self.create_agent_panel_widgets(agent_panel)
        
        # Right Pane: Output and Controls
        right_frame = ttk.Frame(main_pane)
        main_pane.add(right_frame, weight=3)
        right_frame.rowconfigure(0, weight=1)
        right_frame.columnconfigure(0, weight=1)

        # --- NEW: Notebook for Final Answer and Full Log ---
        self.output_notebook = ttk.Notebook(right_frame)
        self.output_notebook.grid(row=0, column=0, sticky="nsew")

        answer_frame = ttk.Frame(self.output_notebook)
        log_frame = ttk.Frame(self.output_notebook)
        self.output_notebook.add(answer_frame, text="Final Answer")
        self.output_notebook.add(log_frame, text="Full Execution Log")

        self.answer_text = scrolledtext.ScrolledText(answer_frame, wrap="word", state=DISABLED, **self.style_manager.get_widget_styles("text"), padx=10, pady=10)
        self.answer_text.pack(fill="both", expand=True)
        self.answer_text.insert(END, "Ready to execute a new task. The final answer will appear here.")
        self.answer_text.config(state=DISABLED)

        self.log_review_text = scrolledtext.ScrolledText(log_frame, wrap="word", state=DISABLED, **self.style_manager.get_widget_styles("text"), padx=10, pady=10)
        self.log_review_text.pack(fill="both", expand=True)
        self.setup_text_tags(self.log_review_text)

        # Bottom Frame for Input and Controls
        bottom_frame = ttk.Frame(right_frame, padding=(0, 10, 0, 0))
        bottom_frame.grid(row=1, column=0, sticky="nsew")
        bottom_frame.columnconfigure(0, weight=1); bottom_frame.columnconfigure(1, minsize=200)

        self.prompt_text = scrolledtext.ScrolledText(bottom_frame, height=4, wrap="word", **self.style_manager.get_widget_styles("text"))
        self.prompt_text.grid(row=0, column=0, sticky="nsew")
        self.prompt_text.bind("<Return>", lambda e: self.start_task() if not self.is_task_running else "break")
        self.prompt_text.bind("<Shift-Return>", lambda e: self.prompt_text.insert(tk.INSERT, '\n') or "break")

        controls_frame = ttk.Frame(bottom_frame)
        controls_frame.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        self.create_controls_widgets(controls_frame)

    def create_menu(self):
        menubar = Menu(self.root, bg=self.style_manager.colors["bg_light"], fg=self.style_manager.colors["fg"])
        self.root.config(menu=menubar)
        settings_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Settings", menu=settings_menu)
        settings_menu.add_command(label="Core Agents", command=lambda: CoreAgentsSettingsWindow(self.root, self.agent_manager.config_manager))
        settings_menu.add_command(label="Cognitive Memory", command=lambda: MemoryViewerWindow(self.root, self.agent_manager.memory_manager))

    def create_agent_panel_widgets(self, parent):
        self.agent_listbox = Listbox(parent, **self.style_manager.get_widget_styles("listbox"))
        self.agent_listbox.pack(fill="both", expand=True, pady=(0, 10))
        self.agent_listbox.bind("<<ListboxSelect>>", self.on_agent_select)
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill="x", pady=(0, 10)); btn_frame.columnconfigure((0, 1, 2), weight=1)
        self.add_button = ttk.Button(btn_frame, text="Add", command=self.add_agent); self.add_button.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        self.edit_button = ttk.Button(btn_frame, text="Edit", command=self.edit_agent); self.edit_button.grid(row=0, column=1, sticky="ew", padx=5)
        self.remove_button = ttk.Button(btn_frame, text="Remove", command=self.remove_agent, style="Destructive.TButton"); self.remove_button.grid(row=0, column=2, sticky="ew", padx=(5, 0))
        self.agent_desc_text = scrolledtext.ScrolledText(parent, height=8, wrap="word", state=DISABLED, **self.style_manager.get_widget_styles("text")); self.agent_desc_text.pack(fill="both", expand=True)

    def create_controls_widgets(self, parent):
        self.execute_button = ttk.Button(parent, text="Execute Task", command=self.start_task, style="Primary.TButton")
        self.execute_button.pack(fill="x")
        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill="x", pady=10)
        provider_frame = ttk.LabelFrame(parent, text="LLM Provider")
        provider_frame.pack(fill="x", pady=(0, 10))
        ttk.Radiobutton(provider_frame, text="OpenRouter", variable=self.provider_var, value="openrouter").pack(side="left", padx=10, pady=5)
        ttk.Radiobutton(provider_frame, text="LM Studio", variable=self.provider_var, value="lmstudio").pack(side="left", padx=10, pady=5)
        ttk.Label(parent, text="Active Agents:").pack(anchor="w")
        self.num_agents_var = tk.IntVar(value=1)
        self.num_agents_spinbox = ttk.Spinbox(parent, from_=1, to=1, width=5, textvariable=self.num_agents_var, state=DISABLED, style="TSpinbox"); self.num_agents_spinbox.pack(fill="x")
        self.max_tokens_label = ttk.Label(parent, text="Max Tokens: 10000"); self.max_tokens_label.pack(anchor="w", pady=(10, 0))
        self.max_tokens_var = tk.IntVar(value=10000)
        self.max_tokens_slider = ttk.Scale(parent, from_=256, to=100000, variable=self.max_tokens_var, orient=tk.HORIZONTAL, command=self.update_max_tokens_label); self.max_tokens_slider.pack(fill="x")

    def start_task(self):
        prompt = self.prompt_text.get("1.0", END).strip()
        if not prompt or self.is_task_running: return
        provider = self.provider_var.get()
        if provider == "lmstudio":
            # Remind user to have LM Studio running
            if not messagebox.askokcancel("LM Studio Check", "You have selected LM Studio as the provider.\n\nPlease ensure the LM Studio server is running and a model is fully loaded before continuing.", parent=self.root):
                return
        
        self.is_task_running = True
        self.set_ui_state(running=True)
        self.log_photo_images.clear()
        self.full_task_log.clear()

        # Clear previous results
        for widget in [self.answer_text, self.log_review_text]:
            widget.config(state=NORMAL); widget.delete("1.0", END); widget.config(state=DISABLED)
        
        # Create log window and hide main window
        self.execution_log_window = ExecutionLogWindow(self.root, self.cancel_task)
        self.setup_text_tags(self.execution_log_window.log_text)
        self.root.withdraw()
        
        self.prompt_text.delete("1.0", END)
        
        threading.Thread(target=self.agent_manager.execute_task, args=(prompt, self.num_agents_var.get(), self.max_tokens_var.get(), provider), daemon=True).start()

    def on_task_finish(self, result):
        self.root.after(0, self._on_task_finish_ui, result)
        
    def _on_task_finish_ui(self, result):
        # Don't close the log window yet - just update button state
        if self.execution_log_window:
            # Change the button text to indicate completion
            for widget in self.execution_log_window.winfo_children():
                if isinstance(widget, ttk.Button) and widget['text'] == "Stop Task":
                    widget.config(text="Close", style="TButton")
                    widget.config(command=self.close_execution_log)  # <-- This line is the problem
                    break
        
        # Display final answer
        final_answer = result.get("final_answer", "Task finished with no conclusive answer.")
        self.answer_text.config(state=NORMAL)
        self.answer_text.delete("1.0", END)
        self.answer_text.insert(END, final_answer)
        self.answer_text.config(state=DISABLED)
        
        # Display full log for review
        self.log_review_text.config(state=NORMAL)
        self.log_review_text.delete("1.0", END)
        for item, tag in self.full_task_log:
            self.log_item_to_widget(self.log_review_text, item, tag)
        self.log_review_text.config(state=DISABLED)
    
        self.is_task_running = False
        self.set_ui_state(running=False)
        
        # Log completion message
        self.log_to_gui("\n[TASK COMPLETE] You can now close this window.", "success")

    def cancel_task(self):
        if not self.is_task_running: return
        self.agent_manager.cancel_task()

    def close_execution_log(self):
        """Close the execution log window and return to main window."""
        if self.execution_log_window and self.execution_log_window.winfo_exists():
            self.execution_log_window.destroy()
            self.execution_log_window = None
        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()
        
    def set_ui_state(self, running: bool):
        state = DISABLED if running else NORMAL
        self.prompt_text.config(state=state)
        self.execute_button.config(state=state)
        for widget in [self.add_button, self.edit_button, self.remove_button, self.max_tokens_slider]:
            widget.config(state=state)
        self.num_agents_spinbox.config(state='readonly' if not running and self.agent_listbox.size() > 0 else DISABLED)

    def process_log_queue(self):
        try:
            while not self.log_queue.empty():
                item, tag = self.log_queue.get_nowait()
                self.full_task_log.append((item, tag))
                
                if tag == "system_command":
                    if item == "HIDE_WINDOWS_FOR_SCREENSHOT": 
                        self.hide_windows_for_screenshot()
                    elif item == "SHOW_WINDOWS_AFTER_SCREENSHOT": 
                        self.show_windows_after_screenshot()
                    continue
    
                # Log to execution window if it exists and is valid
                if self.execution_log_window and self.execution_log_window.winfo_exists():
                    self.log_item_to_widget(self.execution_log_window.log_text, item, tag)
    
            while not self.question_queue.empty():
                question = self.question_queue.get_nowait()
                parent = self.execution_log_window or self.root
                parent.lift()
                self.answer_queue.put(messagebox.askyesno("Human Verification Needed", question, parent=parent))
        finally:
            self.root.after(100, self.process_log_queue)

    def log_item_to_widget(self, widget, item, tag):
        try:
            widget.config(state=NORMAL)
            if isinstance(item, str):
                if tag and tag.startswith("llm_stream"):
                    widget.insert(END, item, ("llm_stream",)); 
                    if tag == "llm_stream_end": widget.insert(END, '\n')
                else: widget.insert(END, item + "\n", (tag,) if tag else ())
            elif isinstance(item, Image.Image):
                max_width = widget.winfo_width() - 40 if widget.winfo_width() > 40 else 600
                w, h = item.size; new_w = min(w, max_width); new_h = int(new_w * (h / w))
                photo = ImageTk.PhotoImage(item.resize((new_w, new_h), Image.Resampling.LANCZOS))
                self.log_photo_images.append(photo) # Keep reference
                widget.image_create(END, image=photo, padx=10, pady=10); widget.insert(END, '\n\n')
                if len(self.log_photo_images) > 20: self.log_photo_images = self.log_photo_images[-10:]
            widget.see(END)
        except Exception as e: print(f"Error logging to GUI: {e}")
        finally:
            if widget.winfo_exists(): widget.config(state=DISABLED)
    
    def log_to_gui(self, item, tag=None): self.log_queue.put((item, tag))

    def hide_windows_for_screenshot(self):
        if self.execution_log_window: self.execution_log_window.withdraw()
        # Main window is already withdrawn during task execution

    def show_windows_after_screenshot(self):
        if self.execution_log_window: self.execution_log_window.deiconify()
        # Main window remains withdrawn

    def setup_text_tags(self, widget):
        colors, fonts = self.style_manager.colors, self.style_manager.fonts
        tags = {"info": (colors["fg_muted"], False), "header": (colors["accent"], True), "success": (colors["green"], True), "error": (colors["red"], False), "warning": (colors["yellow"], False), "output": (colors["fg"], False), "llm_stream": (colors["accent_active"], False)}
        for tag, (color, is_bold) in tags.items():
            widget.tag_config(tag, foreground=color, font=fonts["bold"] if is_bold else fonts["main"])

    def populate_agent_list(self):
        # Unchanged from previous version
        self.agent_listbox.delete(0, END)
        self.agent_manager.reload_agents(self.question_queue, self.answer_queue)
        agents = self.agent_manager.user_agents_data
        for agent in agents: self.agent_listbox.insert(END, agent['name'])
        max_agents = len(agents) if agents else 0
        if max_agents > 0:
            self.num_agents_spinbox.config(from_=1, to=max_agents, state="readonly")
            self.num_agents_var.set(min(1, max_agents))
        else:
            self.num_agents_spinbox.config(from_=1, to=1, state=DISABLED)

    def update_max_tokens_label(self, value):
        val = int(float(value) // 128 * 128)
        self.max_tokens_var.set(val)
        self.max_tokens_label.config(text=f"Max Tokens: {val}")

    def on_agent_select(self, event):
        # Unchanged from previous version
        selections = self.agent_listbox.curselection()
        if not selections: return
        name = self.agent_listbox.get(selections[0])
        data = next((a for a in self.agent_manager.user_agents_data if a['name'] == name), None)
        self.agent_desc_text.config(state=NORMAL)
        self.agent_desc_text.delete("1.0", END)
        if data:
            desc = f"Description:\n{data.get('description', 'N/A')}\n\nLLM ID: {data.get('llm_id', 'N/A')}"
            prompt = data.get('system_prompt')
            if prompt: desc += f"\n\n---\n\nSystem Prompt:\n{prompt}"
            self.agent_desc_text.insert("1.0", desc)
        self.agent_desc_text.config(state=DISABLED)

    def add_agent(self): self._edit_agent_dialog()
    def edit_agent(self): self._edit_agent_dialog(edit_mode=True)
        
    def _edit_agent_dialog(self, edit_mode=False):
        agent_to_edit, name = None, None
        if edit_mode:
            selections = self.agent_listbox.curselection()
            if not selections: return messagebox.showerror("Selection Error", "Please select an agent to edit.")
            name = self.agent_listbox.get(selections[0])
            agent_to_edit = next((a for a in self.agent_manager.user_agents_data if a['name'] == name), None)
        
        title = f"Edit Agent: {name}" if edit_mode else "Add New Agent"
        dialog = AgentDialog(self.root, title, agent_to_edit)
        if not dialog.result: return

        if not edit_mode and any(a['name'].lower() == dialog.result['name'].lower() for a in self.agent_manager.user_agents_data):
            return messagebox.showerror("Duplicate Agent", "An agent with this name already exists.")

        if edit_mode:
            self.agent_manager.user_agents_data = [dialog.result if a['name'] == name else a for a in self.agent_manager.user_agents_data]
        else:
            self.agent_manager.user_agents_data.append(dialog.result)
        
        self.agent_manager.config_manager.save_user_agents(self.agent_manager.user_agents_data)
        self.populate_agent_list()

    def remove_agent(self):
        selections = self.agent_listbox.curselection()
        if not selections: return messagebox.showerror("Selection Error", "Please select an agent to remove.")
        name = self.agent_listbox.get(selections[0])
        if messagebox.askyesno("Confirm Removal", f"Are you sure you want to remove agent '{name}'?"):
            self.agent_manager.user_agents_data = [a for a in self.agent_manager.user_agents_data if a['name'] != name]
            self.agent_manager.config_manager.save_user_agents(self.agent_manager.user_agents_data)
            self.populate_agent_list()

# --- Utility & Settings Dialogs (Unchanged) ---
class AgentDialog(simpledialog.Dialog):
    def __init__(self, parent, title, agent_data=None): self.agent_data = agent_data or {}; super().__init__(parent, title)
    def body(self, master):
        Label(master, text="Agent Name:").grid(row=0, sticky="w", padx=5, pady=2); self.name_entry = Entry(master, width=50); self.name_entry.grid(row=0, column=1, padx=5, pady=2)
        Label(master, text="Agent Description:").grid(row=1, sticky="nw", padx=5, pady=2); self.desc_text = scrolledtext.ScrolledText(master, width=50, height=5, wrap="word"); self.desc_text.grid(row=1, column=1, padx=5, pady=2)
        Label(master, text="LLM ID:").grid(row=2, sticky="w", padx=5, pady=2); self.llm_id_entry = Entry(master, width=50); self.llm_id_entry.grid(row=2, column=1, padx=5, pady=2)
        Label(master, text="System Prompt:").grid(row=3, sticky="nw", padx=5, pady=2); self.system_prompt_text = scrolledtext.ScrolledText(master, width=50, height=10, wrap="word"); self.system_prompt_text.grid(row=3, column=1, padx=5, pady=2)
        self.name_entry.insert(0, self.agent_data.get('name', '')); self.desc_text.insert("1.0", self.agent_data.get('description', '')); self.llm_id_entry.insert(0, self.agent_data.get('llm_id', '')); self.system_prompt_text.insert("1.0", self.agent_data.get('system_prompt', ''))
        if self.agent_data.get('name'): self.name_entry.config(state=DISABLED)
        return self.name_entry
    def apply(self):
        name = self.name_entry.get().strip(); description = self.desc_text.get("1.0", "end-1c").strip(); llm_id = self.llm_id_entry.get().strip(); system_prompt = self.system_prompt_text.get("1.0", "end-1c").strip()
        if not name or not description or not llm_id: messagebox.showerror("Input Error", "Name, Description, and LLM ID are required.", parent=self); self.result = None; return
        self.result = {"name": name, "description": description, "llm_id": llm_id, "system_prompt": system_prompt}

class CoreAgentsSettingsWindow(Toplevel):
    """A Toplevel window for viewing and editing core agent configurations."""
    def __init__(self, parent, config_manager):
        super().__init__(parent)
        self.config_manager = config_manager
        self.style_manager = StyleManager()  # Use the app's style manager

        # --- Window Configuration ---
        self.title("Core Agent Settings")
        self.geometry("900x700")
        self.configure(bg=self.style_manager.colors['bg'])
        self.transient(parent)  # Keep this window on top of the main one
        self.grab_set()         # Modal behavior: block interaction with the main window

        # --- Widgets ---
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)

        # A dictionary to hold references to the Entry and Text widgets for saving later
        self.agent_widgets = {}

        self.load_settings()

    def load_settings(self):
        """Loads the current configuration and builds the UI tabs and widgets."""
        config = self.config_manager.load_core_agents_config()

        # Create a tab for each core agent in the configuration
        for name, data in config.items():
            frame = ttk.Frame(self.notebook, padding=10)
            self.notebook.add(frame, text=name.replace("_", " ").title())

            # --- LLM ID Widgets ---
            ttk.Label(frame, text="LLM ID:").pack(anchor="w")
            llm_id_entry = ttk.Entry(frame, font=self.style_manager.fonts['main'])
            llm_id_entry.pack(fill="x", pady=(2, 10))
            llm_id_entry.insert(0, data.get("llm_id", ""))

            # --- System Prompt Widgets (BUG FIX APPLIED HERE) ---
            # Always create the label and text box, regardless of whether a prompt exists.
            ttk.Label(frame, text="System Prompt:").pack(anchor="w")
            
            # Use a ScrolledText widget for the prompt, applying consistent styling
            prompt_text = scrolledtext.ScrolledText(
                frame,
                wrap="word",
                height=20,
                **self.style_manager.get_widget_styles("text")
            )
            prompt_text.pack(fill="both", expand=True)
            
            # Insert the prompt content. If it doesn't exist, .get() returns an empty string.
            prompt_text.insert("1.0", data.get("prompt", ""))

            # Store references to the input widgets for saving later
            self.agent_widgets[name] = {
                "llm_id": llm_id_entry,
                "prompt": prompt_text
            }

        # --- Save and Cancel Buttons ---
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill="x", padx=10, pady=(0, 10))
        
        # Place buttons on the right side of the frame
        save_button = ttk.Button(btn_frame, text="Save Settings", command=self.save_settings, style="Primary.TButton")
        save_button.pack(side="right")
        
        cancel_button = ttk.Button(btn_frame, text="Cancel", command=self.destroy)
        cancel_button.pack(side="right", padx=10)

    def save_settings(self):
        """Reads data from the widgets, constructs a new config, and saves it."""
        new_config = {}
        # Iterate through the agent names and their corresponding widgets
        for name, widgets in self.agent_widgets.items():
            # Get the values from the UI, stripping any extra whitespace
            llm_id_val = widgets["llm_id"].get().strip()
            prompt_val = widgets["prompt"].get("1.0", "end-1c").strip()
            
            # Build the new configuration entry for this agent
            new_config[name] = {"llm_id": llm_id_val}
            
            # IMPORTANT: Only add the "prompt" key if the user has actually entered text.
            # This prevents saving empty prompts to the JSON file.
            if prompt_val:
                new_config[name]["prompt"] = prompt_val

        try:
            # Save the newly constructed configuration dictionary
            self.config_manager.save_core_agents_config(new_config)
            messagebox.showinfo("Success", "Core agent settings have been saved.", parent=self)
            self.destroy()  # Close the window on successful save
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save settings: {e}", parent=self)

class MemoryViewerWindow(Toplevel):
    def __init__(self, parent, memory_manager):
        super().__init__(parent)
        self.memory_manager = memory_manager
        self.title("Cognitive Memory Viewer")
        self.geometry("900x700")
        self.transient(parent)
        self.grab_set()

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)

        self.episodic_text = self.create_tab("Episodic")
        self.procedural_text = self.create_tab("Procedural")
        self.semantic_text = self.create_tab("Semantic")

        self.refresh_all()

        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill="x", padx=10, pady=(0, 10))
        ttk.Button(btn_frame, text="Clear ALL Memory", command=self.clear_all_memory, style="Destructive.TButton").pack(side="right")
        ttk.Button(btn_frame, text="Refresh", command=self.refresh_all).pack(side="right", padx=10)

    def create_tab(self, title):
        frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(frame, text=title)
        text_widget = scrolledtext.ScrolledText(frame, wrap="word", state=DISABLED, font=StyleManager().fonts['log'])
        text_widget.pack(fill="both", expand=True)
        text_widget.tag_config("bold", font=StyleManager().fonts['bold'])
        return text_widget

    def load_memory(self, widget, loader_func, error_msg):
        widget.config(state=NORMAL)
        widget.delete("1.0", END)
        try:
            loader_func(widget)
        except Exception as e:
            widget.insert("1.0", f"{error_msg}: {e}")
        finally:
            # Ensure the widget is always disabled after operation
            widget.config(state=DISABLED)

    def load_episodic(self, widget):
        files = sorted(self.memory_manager.episodic_path.glob("*.json"), reverse=True)
        if not files:
            widget.insert("1.0", "No episodes recorded yet.")
            return
        for f in files:
            with open(f, 'r', encoding='utf-8') as file:
                data = json.load(file)
            widget.insert(END, f"--- Episode: {f.name} ---\n", "bold")
            widget.insert(END, f"Goal: {data.get('goal')}\n\n")

    def load_procedural(self, widget):
        data = self.memory_manager.procedural.get(include=["metadatas", "documents"])
        if not data or not data['ids']:
            widget.insert("1.0", "No procedures learned yet.")
            return
        for doc, meta in zip(data['documents'], data['metadatas']):
            widget.insert(END, f"Procedure: {doc}\n", "bold")
            widget.insert(END, f" Steps: {json.dumps(json.loads(meta['steps']), indent=2)}\n")
            widget.insert(END, f" Source: {meta['source_episode']}\n\n")

    def load_semantic(self, widget):
        data = self.memory_manager.semantic.get(include=["metadatas", "documents"])
        if not data or not data['ids']:
            widget.insert("1.0", "No facts learned yet.")
            return
        for doc, meta in zip(data['documents'], data['metadatas']):
            widget.insert(END, f"Fact: {doc}\n", "bold")
            widget.insert(END, f" Source: {meta['source_episode']}\n\n")

    def refresh_all(self):
        self.load_memory(self.episodic_text, self.load_episodic, "Error loading episodes")
        self.load_memory(self.procedural_text, self.load_procedural, "Error loading procedures")
        self.load_memory(self.semantic_text, self.load_semantic, "Error loading facts")

    def clear_all_memory(self):
        if messagebox.askyesno("Confirm", "Permanently delete ALL cognitive memory?", parent=self):
            try:
                self.memory_manager.client.delete_collection("procedural_memory")
                self.memory_manager.client.delete_collection("semantic_memory")
                for f in self.memory_manager.episodic_path.glob("*.json"):
                    f.unlink()
                messagebox.showinfo("Success", "Memory cleared. Restart to re-initialize.", parent=self)
                self.destroy()
            except Exception as e:
                messagebox.showerror("Error", f"Could not clear memory: {e}", parent=self)

# --- Application Entry Point ---

if __name__ == "__main__":
    try:
        if platform.system() == "Windows": ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception: pass
    
    root = tk.Tk()
    app = TaskExecutorGUI(root)
    root.mainloop()
