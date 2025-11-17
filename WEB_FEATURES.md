# 🌐 Web Search & Documentation Features

## 🔍 Web Search Tools

### 1. **General Web Search**
```python
"Search web for Python async best practices"
"Search web for LangChain tutorials"
```
- Uses DuckDuckGo
- Returns top 5 results
- Shows live search status

### 2. **Read Webpage**
```python
"Read webpage https://python.langchain.com/docs/introduction"
```
- Extracts text content
- Removes scripts/styles
- Returns first 3000 chars

## 📚 Documentation Search

### 3. **LangChain Docs**
```python
"Search LangChain docs for agents"
"Search LangChain docs for LCEL"
"Search LangChain docs for tools"
```
- Searches official LangChain docs
- Returns top 3 results with links

### 4. **LangGraph Docs**
```python
"Search LangGraph docs for state management"
"Search LangGraph docs for checkpoints"
```
- Searches LangGraph documentation
- Returns relevant docs

### 5. **MCP Docs**
```python
"Search MCP docs for tools"
"Search MCP docs for servers"
```
- Searches Model Context Protocol docs
- Returns official documentation

### 6. **Python Docs**
```python
"Search Python docs for asyncio"
"Search Python docs for decorators"
```
- Searches official Python docs
- Returns relevant documentation

## 💬 Stack Overflow

### 7. **Stack Overflow Search**
```python
"Search Stack Overflow for async errors"
"Search Stack Overflow for LangChain issues"
```
- Searches Stack Overflow
- Returns top 5 Q&A results

## 📦 Library Information

### 8. **PyPI Library Info**
```python
"Get library info for langchain"
"Get library info for langgraph"
"Get info about fastapi library"
```
- Fetches from PyPI
- Shows version, author, description
- Includes links

## 🎯 Live Status Updates

All web tools show real-time status:
```
🔍 Searching web for: Python async
→ Fetching results...
✓ Found 5 results

📚 Searching LangChain docs: agents
→ Querying documentation...
✓ Retrieved docs

📦 Getting info for: langchain
→ Fetching from PyPI...
✓ Retrieved library info
```

## 🚀 Usage Examples

### Example 1: Learn about LangChain
```
User: "Search LangChain docs for how to create agents"
Agent: 🔍 Searches docs → Returns relevant pages
```

### Example 2: Debug Error
```
User: "Search Stack Overflow for ZeroDivisionError in Python"
Agent: 💬 Searches SO → Returns solutions
```

### Example 3: Library Research
```
User: "Get info about langgraph library"
Agent: 📦 Fetches PyPI → Shows version, description, links
```

### Example 4: Web Research
```
User: "Search web for Python best practices 2024"
Agent: 🔍 Searches web → Returns top articles
```

## 🎨 Terminal Output

Beautiful formatted output with:
- 🔍 Search icons
- 📚 Documentation icons
- ✓ Success indicators
- → Progress indicators
- Color-coded status messages

## 🔧 Technical Details

- **Search Engine**: DuckDuckGo (no API key needed)
- **Web Scraping**: BeautifulSoup4
- **HTTP Requests**: requests library
- **Timeout**: 10-15 seconds per request
- **Rate Limiting**: Built-in delays
- **Error Handling**: Graceful fallbacks

## 💡 Tips

1. Be specific in search queries
2. Use quotes for exact phrases
3. Combine with other tools (e.g., search then read webpage)
4. Check multiple sources for accuracy
5. Use library info before installing packages
