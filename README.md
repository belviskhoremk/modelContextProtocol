# MCP Python Learning Project

A comprehensive collection of Python examples and implementations for learning the Model Context Protocol (MCP). This educational project demonstrates various MCP concepts, patterns, and integrations through practical code examples.

## 🎯 Features

- **Multiple Learning Examples**: Various Python scripts showcasing different MCP implementation patterns
- **Tool Integration Demos**: Examples of MCP tool usage with different LLM providers
- **Async & Sync Patterns**: Both synchronous and asynchronous MCP implementations
- **LangChain Integration**: Examples using LangChain with MCP tools
- **Fireworks AI Support**: Demonstrations using Fireworks AI as the LLM provider
- **Educational Focus**: Clear, well-commented code for learning purposes

## 🛠️ Prerequisites

- Python 3.9 or higher
- [uv](https://github.com/astral-sh/uv) package manager

## 📦 Installation

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd modelContextProtocol
```

### 2. Set Up Environment with uv
```bash
# Initialize project (if not already done)
uv init

# Create virtual environment
uv venv

# Activate virtual environment
# On Unix/macOS:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate

# Install dependencies
uv sync
```

### 3. Environment Configuration
Create a `.env` file in the project root:
```env
FIREWORKS_API_KEY=your_fireworks_api_key_here
# Add other API keys as needed
```

## 🚀 Getting Started

### Basic Usage
1. **Explore the Examples**: Browse through the Python files to understand different MCP patterns
2. **Run Individual Scripts**: Execute specific examples to see MCP in action
3. **Modify and Experiment**: Adapt the code to test your own MCP implementations


## 📚 Learning Path

1. **Start with Basics**: Begin with simple MCP examples
2. **Understand Tool Integration**: Learn how MCP tools work with different providers
3. **Explore Async Patterns**: Study asynchronous MCP implementations
4. **Practice with Real APIs**: Use actual API keys to test live integrations
5. **Build Your Own**: Create custom MCP tools and integrations

## 🔧 Key Concepts Covered

- **MCP Protocol Fundamentals**: Understanding the core MCP specification
- **Tool Definition and Registration**: How to define and use MCP tools
- **Session Management**: Handling MCP sessions and connections
- **Error Handling**: Proper error management in MCP implementations
- **Integration Patterns**: Various ways to integrate MCP with LLM providers

## 📖 Additional Resources

- [MCP Official Documentation](https://modelcontextprotocol.io/)
- [LangChain MCP Integration](https://docs.langchain.com/)
- [Fireworks AI Documentation](https://docs.fireworks.ai/)

## 🤝 Contributing

This is an educational project! Feel free to:
- Add new examples
- Improve existing code
- Fix bugs or typos
- Enhance documentation

## 📄 License

This project is for educational purposes. Feel free to use, modify, and distribute the code for learning and teaching MCP concepts.

## 🆘 Troubleshooting

### Common Issues

**uv not found**: Install uv following the [official installation guide](https://github.com/astral-sh/uv#installation)

**API Key Issues**: Ensure your API keys are properly set in the `.env` file

**Import Errors**: Make sure you've activated the virtual environment and run `uv sync`

### Getting Help

- Check the example code comments for detailed explanations
- Review the MCP official documentation
- Examine error messages for specific guidance

---

Happy learning with MCP! 🚀