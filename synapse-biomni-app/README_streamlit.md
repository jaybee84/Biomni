# Biomni Streamlit Application

A user-friendly web interface for interacting with the Biomni biomedical AI agent. This application provides an intuitive UI for entering research prompts and viewing AI responses.

## Features

- 🔑 **API Key Management**: Secure setup for Anthropic Claude API
- 🤖 **Agent Initialization**: Easy configuration of Biomni agent
- 🔗 **Synapse Integration**: Connect to Synapse.org for data access
- 💬 **Interactive Chat**: User-friendly prompt interface
- 📋 **Chat History**: View and manage conversation history
- 💾 **Export Options**: Save conversations as text or JSON
- 📊 **Status Monitoring**: Real-time system status indicators

## Prerequisites

1. **Anthropic API Key**: Get your API key from [Anthropic Console](https://console.anthropic.com/)
2. **Synapse Account**: Register at [Synapse.org](https://www.synapse.org/) if using Synapse features
3. **Python Environment**: Python 3.8+ with required packages

## Installation

1. **Install dependencies**:
   ```bash
   pip install -r requirements_streamlit.txt
   ```

2. **Set up your API key** (choose one method):
   
   **Option A: Environment Variable**
   ```bash
   export ANTHROPIC_API_KEY="your-api-key-here"
   ```
   
   **Option B: .env File**
   Create a `.env` file in the same directory:
   ```
   ANTHROPIC_API_KEY=your-api-key-here
   ```

3. **Configure Synapse** (optional, for Synapse features):
   ```bash
   # Create/edit ~/.synapseConfig
   [profile biomni-agent-test]
   username=your-username
   authtoken=your-personal-access-token
   ```

## Running the Application

1. **Start the Streamlit app**:
   ```bash
   streamlit run biomni_streamlit_app.py
   ```

2. **Open your browser** to the URL shown (typically `http://localhost:8501`)

3. **Configure the application**:
   - Set up your API key in the sidebar
   - Initialize the Biomni agent
   - Optionally connect to Synapse

## Usage Guide

### Basic Usage

1. **Setup Phase**:
   - Enter your Anthropic API key
   - Configure data path for Biomni (default: `~/biomni_data_test`)
   - Select your preferred Claude model
   - Click "Initialize Agent"

2. **Research Phase**:
   - Choose from example prompts or write your own
   - Click "Submit" to send your prompt to the agent
   - View results in the chat history panel

3. **Advanced Features**:
   - Connect to Synapse for data access
   - Export chat history for later reference
   - Clear history to start fresh

### Example Prompts

The app includes several pre-built example prompts:

- **CRISPR Screen Planning**: Design genetic screens for specific research questions
- **Drug Discovery**: Identify therapeutic targets and compounds
- **Synapse Data Analysis**: Download and analyze datasets from Synapse
- **Biomarker Discovery**: Find molecular signatures in omics data

### System Status

The interface shows real-time status for:
- ✅ **API Key**: Anthropic authentication status
- ✅ **Agent**: Biomni agent initialization status
- ✅ **Synapse**: Synapse.org connection status

## Troubleshooting

### Common Issues

1. **API Key Error**:
   ```
   "Could not resolve authentication method"
   ```
   - Solution: Ensure your Anthropic API key is correctly set

2. **Agent Initialization Error**:
   ```
   "Permission denied: '/dfs'"
   ```
   - Solution: Use a local path like `~/biomni_data_test` instead of system paths

3. **Synapse Connection Error**:
   ```
   "DuplicateSectionError"
   ```
   - Solution: Check your `~/.synapseConfig` file for duplicate profile entries

4. **Import Error**:
   ```
   "No module named 'biomni'"
   ```
   - Solution: Ensure you're in the correct environment and biomni is installed

### Performance Tips

- **First Run**: Initial agent setup downloads ~20GB of data lake files
- **Memory**: Large datasets may require significant RAM
- **Network**: Stable internet connection recommended for downloads

## File Structure

```
biomni_streamlit_app.py     # Main Streamlit application
requirements_streamlit.txt  # Python dependencies
README_streamlit.md        # This documentation
```

## Security Notes

- Never commit API keys to version control
- Use environment variables or .env files for sensitive data
- The app runs locally by default (localhost:8501)

## Support

For issues related to:
- **Biomni Package**: See [Biomni GitHub](https://github.com/snap-stanford/Biomni)
- **Streamlit**: See [Streamlit Documentation](https://docs.streamlit.io/)
- **Anthropic Claude**: See [Anthropic Documentation](https://docs.anthropic.com/)

## License

This application follows the same license as the Biomni package.
