"""
Core Agent Implementation using LangGraph and MCP
"""
import os
import asyncio
from typing import Annotated, Sequence, Literal

from pydantic import BaseModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.tree import Tree
from rich.prompt import Prompt, Confirm
from rich.syntax import Syntax
from rich.table import Table
from pathlib import Path
import subprocess
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from tool.local_tools import get_local_tools
from tool.mcp_tools import get_mcp_tools
from tool.github_direct import get_github_tools
from tool.github_list import get_github_list_tools
from tool.web_tools import get_web_tools



console = Console()



load_dotenv()




class AgentState(BaseModel):
    """State management for the agent workflow"""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    current_file: str = ""
    error_context: str = ""


class CodeAssistantAgent:
    """
    Minimalist AI Coding Assistant using LangGraph and MCP
    
    Architecture:
    - StateGraph with 3 nodes: user_input, model_response, tool_use
    - Persistent state using SQLite checkpointing
    - Tool integration: local tools + MCP servers
    """
    
    def __init__(self):
        self.console = console
        self._checkpointer_ctx = None
        self.checkpointer = None
        self.agent = None
        self.thread_id = "default_session"
        self.last_options = {}  # Store numbered options from bullet points
        self.mcp_tools = []  # Initialize mcp_tools attribute
        self.current_file = None
        self.file_content = None
        self.max_retries = 3
        
        # Display welcome banner
        self._display_welcome()
        
        # Initialize LLM with rate limit handling
        self.llm  = ChatGroq(
              model="openai/gpt-oss-120b",
              temperature=0,
)
        
        # Initialize tools
        self.console.print("[cyan]Loading tools...[/cyan]")
        self.tools = []
        
        # Load local tools
        local_tools = get_local_tools()
        self.tools.extend(local_tools)
        self.console.print(f"[green]Loaded {len(local_tools)} local tools[/green]")
        
        # Load direct GitHub tools
        github_tools = get_github_tools()
        self.tools.extend(github_tools)
        github_list_tools = get_github_list_tools()
        self.tools.extend(github_list_tools)
        self.console.print(f"[green]Loaded {len(github_tools) + len(github_list_tools)} GitHub tools[/green]")
        
        # Load web tools
        web_tools = get_web_tools()
        self.tools.extend(web_tools)
        self.console.print(f"[green]Loaded {len(web_tools)} web tools[/green]")
        
        # Build workflow
        self.workflow = StateGraph(AgentState)
        self._setup_workflow()
    
    def _display_welcome(self):
        """Display welcome banner"""
        banner = """
=============================================================

   🤖 CODE ASSISTANT - Advanced Edition
   
   • Auto Error Fixing  • Smart File Editing
   • GitHub Integration  • Syntax Highlighting
   
   Type '!help' for advanced commands

=============================================================
        """
        self.console.print(banner, style="bold cyan")
    
    def _setup_workflow(self):
        """Setup the StateGraph workflow"""
        # Register nodes
        self.workflow.add_node("model_response", self.model_response)
        self.workflow.add_node("tool_use", self.tool_use)
        
        # Define edges
        self.workflow.set_entry_point("model_response")
        self.workflow.add_edge("tool_use", "model_response")
        
        # Conditional routing
        self.workflow.add_conditional_edges(
            "model_response",
            self.check_tool_use,
            {
                "tool_use": "tool_use",
                END: END,
            },
        )
    
    async def initialize(self):
        """Async initialization for checkpointer and MCP tools"""
        # Initialize SQLite checkpointer
        db_path = os.path.join(os.getcwd(), "checkpoints.db")
        self.console.print(f"[cyan]Initializing checkpoint database: {db_path}[/cyan]")
        
        self._checkpointer_ctx = AsyncSqliteSaver.from_conn_string(db_path)
        self.checkpointer = await self._checkpointer_ctx.__aenter__()
        
        # Load MCP tools
        try:
            self.mcp_tools = await get_mcp_tools()
            if self.mcp_tools:
                self.tools.extend(self.mcp_tools)
                self.console.print(f"[green]Loaded {len(self.mcp_tools)} GitHub MCP tools[/green]")
        except Exception as e:
            self.console.print(f"[yellow]Warning: Could not load MCP tools: {e}[/yellow]")
        
        # Bind all tools
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        # Compile the workflow with recursion limit
        self.agent = self.workflow.compile(
            checkpointer=self.checkpointer
        )
        self.console.print("[green]Agent initialized successfully![/green]\n")
        
        # Show quick start guide
        self._display_quick_start()
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            if self._checkpointer_ctx:
                await self._checkpointer_ctx.__aexit__(None, None, None)
        except Exception:
            pass  # Ignore cleanup errors on exit
    
    def detect_file_from_context(self, text: str) -> str:
        """Smart file detection from user input"""
        words = text.split()
        for word in words:
            if '.' in word and any(word.endswith(ext) for ext in ['.py', '.js', '.txt', '.md', '.json']):
                if os.path.exists(word):
                    return word
        return None
    
    def show_file_with_syntax(self, file_path: str, content: str):
        """Display file with syntax highlighting"""
        ext = Path(file_path).suffix[1:]
        syntax = Syntax(content, ext or "text", theme="monokai", line_numbers=True)
        self.console.print(Panel(syntax, title=f"📄 {file_path}", border_style="cyan"))
    
    def run_code(self, file_path: str) -> tuple[str, str, int]:
        """Execute code and capture output"""
        try:
            if file_path.endswith('.py'):
                result = subprocess.run(
                    ['python', file_path],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                return result.stdout, result.stderr, result.returncode
            return "", "Unsupported file type", 1
        except subprocess.TimeoutExpired:
            return "", "Execution timeout", 1
        except Exception as e:
            return "", str(e), 1
    
    async def fix_errors_loop(self, file_path: str):
        """Auto-fix errors in loop"""
        for attempt in range(1, self.max_retries + 1):
            self.console.print(f"\n[yellow]🔄 Attempt {attempt}/{self.max_retries}[/yellow]")
            
            stdout, stderr, code = self.run_code(file_path)
            
            if code == 0:
                self.console.print("[green]✓ Code executed successfully![/green]")
                self.console.print(Panel(stdout, title="Output", border_style="green"))
                return True
            
            self.console.print(Panel(stderr, title="Error", border_style="red"))
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            fix_prompt = f"""Fix this error:\n\nERROR:\n{stderr}\n\nCODE:\n{content}\n\nProvide ONLY corrected code."""
            
            with self.console.status("[cyan]Fixing...", spinner="dots"):
                response = self.llm.invoke([HumanMessage(content=fix_prompt)])
            
            fixed_code = response.content
            if '```python' in fixed_code:
                fixed_code = fixed_code.split('```python')[1].split('```')[0].strip()
            elif '```' in fixed_code:
                fixed_code = fixed_code.split('```')[1].split('```')[0].strip()
            
            # Backup
            with open(f"{file_path}.backup", 'w', encoding='utf-8') as f:
                f.write(content)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(fixed_code)
            
            self.console.print("[green]✓ Applied fix[/green]")
        
        self.console.print("[red]❌ Max attempts reached[/red]")
        return False
    
    def edit_file_interactive(self, file_path: str):
        """Interactive file editing"""
        if not os.path.exists(file_path):
            self.console.print(f"[red]File not found: {file_path}[/red]")
            return
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        self.show_file_with_syntax(file_path, content)
        
        self.console.print("\n[cyan]Edit:[/cyan] 1=Replace all 2=Replace lines 3=Insert 4=Delete")
        choice = Prompt.ask("Choose", choices=["1", "2", "3", "4"])
        
        if choice == "1":
            new_content = Prompt.ask("New content")
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            self.console.print("[green]✓ Updated[/green]")
        
        elif choice == "2":
            start = int(Prompt.ask("Start line"))
            end = int(Prompt.ask("End line"))
            new_lines = Prompt.ask("New content")
            lines = content.split('\n')
            lines[start-1:end] = [new_lines]
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
            self.console.print("[green]✓ Updated[/green]")
    
    def scan_project_table(self):
        """Scan and display project as table"""
        table = Table(title="📁 Project Files")
        table.add_column("File", style="cyan")
        table.add_column("Size", style="green")
        table.add_column("Type", style="yellow")
        
        for root, dirs, files in os.walk('.'):
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            for file in files:
                if not file.startswith('.'):
                    path = os.path.join(root, file)
                    size = os.path.getsize(path)
                    ext = Path(file).suffix
                    table.add_row(path, f"{size}B", ext)
        
        self.console.print(table)
    
    def _format_with_numbers(self, text) -> str:
        """Convert bullet points to numbered list and store mapping"""
        import re
        
        # Handle if text is a list of content blocks
        if isinstance(text, list):
            # Extract text from content blocks
            text_parts = []
            for block in text:
                if hasattr(block, 'text'):
                    text_parts.append(block.text)
                elif isinstance(block, dict) and 'text' in block:
                    text_parts.append(block['text'])
                elif isinstance(block, str):
                    text_parts.append(block)
            text = '\n'.join(text_parts)
        
        # Ensure text is a string
        if not isinstance(text, str):
            text = str(text)
        
        # Reset options
        self.last_options = {}
        
        # Find bullet points (• or -)
        lines = text.split('\n')
        option_num = 1
        formatted_lines = []
        
        for line in lines:
            # Match bullet points with various formats
            bullet_match = re.match(r'^(\s*)[•\-\*]\s+(.+)$', line)
            if bullet_match:
                indent = bullet_match.group(1)
                content = bullet_match.group(2)
                
                # Store the mapping
                self.last_options[str(option_num)] = content
                
                # Replace with number
                formatted_lines.append(f"{indent}**{option_num}.** {content}")
                option_num += 1
            else:
                formatted_lines.append(line)
        
        result = '\n'.join(formatted_lines)
        
        # Add helper text if options were found
        if self.last_options:
            result += f"\n\n*Tip: Type a number (1-{len(self.last_options)}) to select an option*"
        
        return result
    
    def model_response(self, state: AgentState) -> dict:
        """Node: Generate model response"""
        messages = list(state.messages)
        
        # Trim messages to prevent token limit - keep only last 5 exchanges
        if len(messages) > 10:
            messages = messages[-10:]
        
        # Always add system message with tool instructions
        system_message = SystemMessage(content="""You MUST call tools. For web search: use search_web(query). For docs: use search_langchain_docs, search_langgraph_docs, search_mcp_docs. For GitHub: use push_folder or quick_push_file. CALL TOOLS DIRECTLY.""")
        
        if len(messages) == 1:
            messages = [system_message] + messages
        else:
            # Replace first message if it's a system message
            if isinstance(messages[0], SystemMessage):
                messages[0] = system_message
            else:
                messages = [system_message] + messages
        
        # Display thinking indicator
        with self.console.status("[bold cyan]Thinking...", spinner="dots"):
            # Check if user wants GitHub action - force tool use
            last_msg = messages[-1].content if hasattr(messages[-1], 'content') else ""
            if any(word in last_msg.lower() for word in ['push', 'upload', 'github', 'repo']):
                # Add explicit instruction to use tool
                messages[-1] = HumanMessage(content=f"{last_msg}\n\n[SYSTEM: You MUST call the appropriate tool NOW. Do not respond with text.]")
            
            response = self.llm_with_tools.invoke(messages)
        
        # Display AI response
        if response.content:
            # Convert bullet points to numbered list
            formatted_content = self._format_with_numbers(response.content)
            
            self.console.print(Panel(
                Markdown(formatted_content),
                title="[bold cyan]Assistant[/bold cyan]",
                border_style="cyan"
            ))
        
        return {"messages": [response]}
    
    async def tool_use(self, state: AgentState) -> dict:
        """Node: Execute tool calls"""
        import sys
        from io import StringIO
        
        messages = state.messages
        last_message = messages[-1]
        
        tool_calls = last_message.tool_calls
        tool_messages = []
        
        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            
            # Display tool execution
            self.console.print(f"\n[bold yellow]Executing tool:[/bold yellow] [magenta]{tool_name}[/magenta]")
            self.console.print(f"[dim]Arguments: {tool_args}[/dim]\n")
            
            # Find and execute the tool
            tool = next((t for t in self.tools if t.name == tool_name), None)
            
            if tool:
                try:
                    # Suppress MCP stderr warnings during tool execution
                    old_stderr = sys.stderr
                    sys.stderr = StringIO()
                    
                    try:
                        # MCP tools are always async
                        if tool in self.mcp_tools:
                            result = await tool.ainvoke(tool_args)
                        else:
                            result = tool.invoke(tool_args)
                    finally:
                        sys.stderr = old_stderr
                    
                    # Display tool result
                    self.console.print(Panel(
                        str(result),
                        title=f"[bold green]Tool Result: {tool_name}[/bold green]",
                        border_style="green"
                    ))
                    
                    tool_messages.append(
                        ToolMessage(
                            content=str(result),
                            tool_call_id=tool_call["id"]
                        )
                    )
                except Exception as e:
                    error_msg = f"Tool error: {str(e)}"
                    self.console.print(f"[bold red]{error_msg}[/bold red]")
                    
                    tool_messages.append(
                        ToolMessage(
                            content=error_msg,
                            tool_call_id=tool_call["id"]
                        )
                    )
            else:
                error_msg = f"Tool {tool_name} not found"
                self.console.print(f"[bold red]{error_msg}[/bold red]")
                
                tool_messages.append(
                    ToolMessage(
                        content=error_msg,
                        tool_call_id=tool_call["id"]
                    )
                )
        
        return {"messages": tool_messages}
    
    def check_tool_use(self, state: AgentState) -> Literal["tool_use", END]:
        """Conditional edge: Check if tools should be used"""
        messages = state.messages
        last_message = messages[-1]
        
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tool_use"
        return END
    
    async def run(self):
        """Main interactive loop"""
        config = {"configurable": {"thread_id": self.thread_id}}
        
        while True:
            try:
                # Get user input with rich prompt
                self.console.print()
                try:
                    user_input = Prompt.ask(
                        "[bold green]Your request[/bold green] [dim](or type 'help')[/dim]",
                        default=""
                    )
                except (EOFError, KeyboardInterrupt):
                    self.console.print("\n[bold cyan]Goodbye![/bold cyan]\n")
                    break
                
                # Skip empty input
                if not user_input or not user_input.strip():
                    continue
                
                # Check for exit commands
                if user_input.lower() in ['exit', 'quit', 'q']:
                    self.console.print("\n[bold cyan]Goodbye![/bold cyan]\n")
                    break
                
                # Special commands
                if user_input.lower() == 'help':
                    self._display_help()
                    continue
                
                if user_input.lower() == 'tools':
                    self._display_tools()
                    continue
                
                if user_input.lower() == 'clear':
                    # Clear conversation history
                    config = {"configurable": {"thread_id": f"session_{os.urandom(4).hex()}"}}
                    self.console.print("[green]✓ Conversation history cleared[/green]")
                    continue
                
                # Check if user typed a number to select an option
                if user_input.strip().isdigit() and user_input.strip() in self.last_options:
                    selected_option = self.last_options[user_input.strip()]
                    self.console.print(f"[green]Selected:[/green] {selected_option}\n")
                    user_input = selected_option
                
                # Direct commands with !
                if user_input.startswith('!'):
                    await self.handle_direct_command(user_input[1:])
                    continue
                
                # Quick command for GitHub repos
                if 'my' in user_input.lower() and 'repo' in user_input.lower() and 'list' in user_input.lower():
                    from tool.github_list import list_my_repos
                    self.console.print("\n[bold yellow]Fetching your repositories...[/bold yellow]\n")
                    result = list_my_repos.invoke({})
                    continue
                
                # Direct command handler for GitHub push
                if 'push' in user_input.lower() and 'folder' in user_input.lower():
                    # Extract info and call tool directly
                    from tool.github_direct import push_folder
                    
                    # Get folder path from current directory
                    folder_name = 'codeAssistent' if 'codeassistent' in user_input.lower() else '.'
                    folder_path = os.path.join(os.getcwd(), folder_name) if folder_name != '.' else os.getcwd()
                    
                    self.console.print(f"\n[bold yellow]Executing:[/bold yellow] push_folder")
                    self.console.print(f"[dim]Folder: {folder_path}[/dim]\n")
                    
                    with self.console.status("[bold cyan]Pushing to GitHub...", spinner="dots"):
                        result = push_folder.invoke({
                            'owner': 'Abhitheshek',
                            'repo': 'terminalcodeAssistant',
                            'folder_path': folder_path,
                            'branch': 'main',
                            'message': 'Upload codeAssistent folder'
                        })
                    
                    self.console.print(Panel(
                        str(result),
                        title="[bold green]Result[/bold green]",
                        border_style="green"
                    ))
                    continue
                
                # Create human message
                human_message = HumanMessage(content=user_input)
                
                # Invoke the workflow (it will run until END)
                await self.agent.ainvoke(
                    {"messages": [human_message]},
                    config=config
                )
                
            except KeyboardInterrupt:
                self.console.print("\n[bold cyan]Goodbye![/bold cyan]\n")
                break
            except Exception as e:
                import traceback
                self.console.print(f"[bold red]Error: {e}[/bold red]")
                self.console.print(f"[dim]{traceback.format_exc()}[/dim]")
                self.console.print("[yellow]Continuing... Type 'exit' to quit[/yellow]")
    
    async def handle_direct_command(self, cmd: str):
        """Handle direct commands"""
        parts = cmd.split()
        if not parts:
            return
        
        action = parts[0].lower()
        
        if action == 'read' and len(parts) > 1:
            file_path = parts[1]
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                self.show_file_with_syntax(file_path, content)
                self.current_file = file_path
            else:
                self.console.print(f"[red]File not found: {file_path}[/red]")
        
        elif action == 'run' and len(parts) > 1:
            stdout, stderr, code = self.run_code(parts[1])
            if code == 0:
                self.console.print(Panel(stdout, title="✓ Output", border_style="green"))
            else:
                self.console.print(Panel(stderr, title="❌ Error", border_style="red"))
        
        elif action == 'fix' and len(parts) > 1:
            await self.fix_errors_loop(parts[1])
        
        elif action == 'edit' and len(parts) > 1:
            self.edit_file_interactive(parts[1])
        
        elif action == 'scan':
            self.scan_project_table()
        
        elif action == 'repos':
            from tool.github_list import list_my_repos
            list_my_repos.invoke({})
        
        elif action == 'help':
            self._display_advanced_help()
        
        else:
            self.console.print(f"[yellow]Unknown command: {action}[/yellow]")
            self.console.print("[dim]Try: !help[/dim]")
    
    def _display_advanced_help(self):
        """Display advanced help"""
        help_text = """
# Advanced Commands

## Direct Commands (use ! prefix):
- **!read <file>** - Display file with syntax highlighting
- **!run <file>** - Execute Python file
- **!fix <file>** - Auto-fix errors in loop (max 3 attempts)
- **!edit <file>** - Interactive file editing
- **!scan** - Show project structure as table
- **!repos** - List your GitHub repositories

## Web & Documentation:
- "Search web for Python decorators"
- "Search LangChain docs for agents"
- "Search LangGraph docs for state management"
- "Search MCP docs for tools"
- "Get info about langchain library"
- "Search Stack Overflow for async errors"

## Examples:
```
!read agent.py
!run test.py
!fix buggy_code.py
"Search LangChain docs for LCEL"
"Get library info for langgraph"
```
        """
        self.console.print(Panel(
            Markdown(help_text),
            title="[bold cyan]Advanced Help[/bold cyan]",
            border_style="cyan"
        ))
    
    def _display_quick_start(self):
        """Display quick start guide on startup"""
        # Set up initial numbered options
        self.last_options = {
            "1": "List all files in current directory",
            "2": "Show available tools",
            "3": "Scan entire project",
            "4": "Read the README.md file",
            "5": "Show advanced help"
        }
        
        quick_start = """
[bold cyan]🤖 Code Assistant - Ready![/bold cyan]

**Quick Actions:**
**1.** List all files in current directory
**2.** Show available tools
**3.** Scan entire project
**4.** Read the README.md file
**5.** Show advanced help

[yellow]⚡ Direct Commands:[/yellow]
[dim]!read <file>  !run <file>  !fix <file>  !edit <file>  !scan  !repos[/dim]

[blue]🌐 Web Features:[/blue]
[dim]Search web, docs (LangChain, LangGraph, MCP, Python), Stack Overflow[/dim]

[dim]Commands: [green]help[/green] | [green]tools[/green] | [green]clear[/green] | [green]!help[/green] | [green]exit[/green][/dim]
"""
        self.console.print(Panel(quick_start, border_style="cyan", padding=(1, 2)))
    
    def _display_help(self):
        """Display help information"""
        help_text = """
# Help

## Available Commands:
- **help**: Display this help message
- **tools**: List all available tools
- **clear**: Clear conversation history (fixes token limit errors)
- **exit/quit/q**: Exit the assistant

## Example Queries:
- "Show me the content of main.py"
- "Search web for Python async best practices"
- "Search LangChain docs for agents"
- "Get info about langchain library"
- "Push agent.py to GitHub"
- "Fix errors in test.py"

## Tips:
- Be specific in your requests
- The assistant can read files, run tests, search the web, and more
- All interactions are saved in checkpoints.db for debugging
        """
        self.console.print(Panel(
            Markdown(help_text),
            title="[bold cyan]Help[/bold cyan]",
            border_style="cyan"
        ))
    
    def _display_tools(self):
        """Display available tools in a tree structure"""
        tree = Tree("[bold cyan]Available Tools[/bold cyan]")
        
        # Group tools by type
        local_branch = tree.add("[yellow]Local Tools[/yellow]")
        web_branch = tree.add("[blue]Web & Docs Tools[/blue]")
        github_branch = tree.add("[green]GitHub Tools[/green]")
        mcp_branch = tree.add("[magenta]MCP Tools[/magenta]")
        
        for tool in self.tools:
            if tool in self.mcp_tools:
                mcp_branch.add(f"[magenta]• {tool.name}[/magenta]: {tool.description}")
            elif 'github' in tool.name.lower() or 'repo' in tool.name.lower():
                github_branch.add(f"[green]• {tool.name}[/green]: {tool.description}")
            elif 'search' in tool.name.lower() or 'web' in tool.name.lower() or 'docs' in tool.name.lower() or 'library' in tool.name.lower():
                web_branch.add(f"[blue]• {tool.name}[/blue]: {tool.description}")
            else:
                local_branch.add(f"[yellow]• {tool.name}[/yellow]: {tool.description}")
        
        self.console.print(tree)