#!/usr/bin/env python3
"""
Biomni Streamlit Application

A user-friendly web interface for interacting with the Biomni biomedical AI agent.
This app provides an intuitive UI for entering prompts and viewing AI responses.
"""

import subprocess
import streamlit as st
import sys
import os
import time
from datetime import datetime
import traceback
from pathlib import Path
import logging
import io
import threading
from queue import Queue
import re
import base64
import json
import pandas as pd

# Activate conda environment at the start
try:
    # Run conda activate command
    result = subprocess.run(
        ["conda", "activate", "biomni_e1"], 
        shell=True, 
        capture_output=True, 
        text=True
    )
    print("Conda environment activation attempted")
except Exception as e:
    print(f"Warning: Could not activate conda environment: {e}")

from biomni.agent import A1

# Add parent directory to path for importing biomni
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

# Try to import and load dotenv
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ python-dotenv loaded successfully")
except ImportError:
    print("⚠️ python-dotenv not available, skipping .env file loading")
    def load_dotenv(*args, **kwargs):
        """Dummy function when dotenv is not available"""
        pass

# Function to load .env file from custom path
def load_env_from_path(env_file_path):
    """Load environment variables from a custom .env file path"""
    try:
        env_path = Path(env_file_path)
        if env_path.exists():
            with open(env_path, 'r') as f:
                content = f.read()
                loaded_vars = []
                for line in content.split('\n'):
                    if '=' in line and not line.strip().startswith('#'):
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        if key and value:
                            os.environ[key] = value
                            loaded_vars.append(key)
                
                # Show which variables were loaded
                vars_msg = ", ".join(loaded_vars) if loaded_vars else "No variables"
                return True, f"✅ Successfully loaded .env file from: {env_path}\n   Variables loaded: {vars_msg}"
        else:
            return False, f"❌ No .env file found at: {env_path}"
    except Exception as e:
        return False, f"❌ Error loading .env file: {str(e)}"

# Configure Streamlit page
st.set_page_config(
    page_title="Biomni AI Agent Interface",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.chat-message {
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}
.user-message {
    background-color: #e3f2fd;
    border-left: 4px solid #2196f3;
}
.ai-message {
    background-color: #f3e5f5;
    border-left: 4px solid #9c27b0;
}
.status-success {
    color: #4caf50;
    font-weight: bold;
}
.status-error {
    color: #f44336;
    font-weight: bold;
}
.status-warning {
    color: #ff9800;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'agent' not in st.session_state:
        st.session_state.agent = None
    if 'api_key_configured' not in st.session_state:
        st.session_state.api_key_configured = False
    if 'auth_method' not in st.session_state:
        st.session_state.auth_method = None  # 'anthropic' or 'aws'
    if 'aws_bearer_token' not in st.session_state:
        st.session_state.aws_bearer_token = None
    if 'synapse_connected' not in st.session_state:
        st.session_state.synapse_connected = False
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'agent_status' not in st.session_state:
        st.session_state.agent_status = "Not initialized"
    if 'log_messages' not in st.session_state:
        st.session_state.log_messages = []
    if 'log_queue' not in st.session_state:
        st.session_state.log_queue = Queue()
    if 'processing_task' not in st.session_state:
        st.session_state.processing_task = False
    if 'last_rerun_time' not in st.session_state:
        st.session_state.last_rerun_time = 0
    if 'execution_counter' not in st.session_state:
        st.session_state.execution_counter = 0

# class StreamlitLogHandler(logging.Handler):
#     """Custom logging handler that sends logs to Streamlit session state"""
    
#     def __init__(self, log_queue):
#         super().__init__()
#         self.log_queue = log_queue
#         self.setFormatter(logging.Formatter(
#             '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
#         ))
    
#     def emit(self, record):
#         try:
#             log_entry = {
#                 'timestamp': datetime.fromtimestamp(record.created).strftime("%H:%M:%S"),
#                 'level': record.levelname,
#                 'name': record.name,
#                 'message': record.getMessage()
#             }
#             self.log_queue.put(log_entry)
#         except Exception:
#             self.handleError(record)

# def setup_logging():
#     """Setup logging configuration for real-time streaming"""
#     if 'logging_configured' not in st.session_state:
#         # Create a custom handler for Streamlit
#         handler = StreamlitLogHandler(st.session_state.log_queue)
        
#         # Configure root logger
#         root_logger = logging.getLogger()
#         root_logger.setLevel(logging.INFO)
        
#         # Remove existing handlers to avoid duplicates
#         for existing_handler in root_logger.handlers[:]:
#             root_logger.removeHandler(existing_handler)
        
#         # Add our custom handler
#         root_logger.addHandler(handler)
        
#         # Configure specific loggers that might be used by Biomni
#         loggers_to_configure = ['biomni', 'synapseclient', 'urllib3', 'requests']
#         for logger_name in loggers_to_configure:
#             logger = logging.getLogger(logger_name)
#             logger.setLevel(logging.INFO)
#             logger.addHandler(handler)
        
#         st.session_state.logging_configured = True

# def get_log_level_color(level):
#     """Get color for log level"""
#     colors = {
#         'DEBUG': '#6c757d',
#         'INFO': '#17a2b8', 
#         'WARNING': '#ffc107',
#         'ERROR': '#dc3545',
#         'CRITICAL': '#721c24'
#     }
#     return colors.get(level, '#000000')

# def display_log_message(log_entry):
#     """Display a single log message with proper styling and formatting"""
#     color = get_log_level_color(log_entry['level'])
    
#     # Format the message for better readability
#     message = format_log_message(log_entry['message'])
    
#     # Choose appropriate icon based on log level
#     icon = {
#         'INFO': 'ℹ️',
#         'WARNING': '⚠️', 
#         'ERROR': '❌',
#         'CRITICAL': '🚨',
#         'DEBUG': '🔍'
#     }.get(log_entry['level'], '📝')
    
#     st.markdown(f"""
#     <div style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; font-size: 0.85em; 
#                 padding: 0.4rem; margin: 0.2rem 0; border-left: 4px solid {color}; 
#                 background-color: {color}10; border-radius: 4px;">
#         <div style="display: flex; align-items: center; margin-bottom: 0.2rem;">
#             <span style="margin-right: 0.5rem;">{icon}</span>
#             <span style="color: #666; font-size: 0.8em;">{log_entry['timestamp']}</span>
#             <span style="color: {color}; font-weight: bold; margin-left: 0.5rem;">{log_entry['level']}</span>
#             <span style="color: #888; margin-left: 0.5rem; font-size: 0.8em;">({log_entry['name']})</span>
#         </div>
#         <div style="color: #333; margin-left: 1.5rem;">{message}</div>
#     </div>
#     """, unsafe_allow_html=True)

# def format_log_message(message):
#     """Format log message for better readability"""
#     # Handle file paths and make them more readable
#     if 'syn' in message.lower() and ('download' in message.lower() or 'upload' in message.lower()):
#         message = f"🔄 <strong>Synapse Operation:</strong> {message}"
#     elif 'plot' in message.lower() or 'chart' in message.lower() or 'visualization' in message.lower():
#         message = f"📊 <strong>Visualization:</strong> {message}"
#     elif 'error' in message.lower() or 'failed' in message.lower():
#         message = f"❌ <strong>Error:</strong> {message}"
#     elif 'success' in message.lower() or 'completed' in message.lower():
#         message = f"✅ <strong>Success:</strong> {message}"
#     elif 'starting' in message.lower() or 'initializing' in message.lower():
#         message = f"🚀 <strong>Starting:</strong> {message}"
#     elif 'file' in message.lower() and ('save' in message.lower() or 'write' in message.lower()):
#         message = f"💾 <strong>File Operation:</strong> {message}"
#     elif 'data' in message.lower() and ('load' in message.lower() or 'read' in message.lower()):
#         message = f"📁 <strong>Data Loading:</strong> {message}"
    
#     # Check for file extensions and highlight them
#     import re
#     file_pattern = r'([^\s]+\.(csv|json|png|jpg|jpeg|pdf|xlsx|txt|py|ipynb))'
#     message = re.sub(file_pattern, r'<code style="background:#e9ecef; padding:1px 3px; border-radius:2px;">\1</code>', message, flags=re.IGNORECASE)
    
#     return message

def setup_api_key():
    """Setup Anthropic API key or AWS Bedrock token"""
    st.sidebar.subheader("🔑 API Configuration")
    
    # Check if API key is already in environment
    if "ANTHROPIC_API_KEY" in os.environ and os.environ["ANTHROPIC_API_KEY"]:
        st.sidebar.success("✅ ANTHROPIC_API_KEY already configured")
        # Ensure it's available for langchain-anthropic
        if not os.environ["ANTHROPIC_API_KEY"].startswith("sk-ant-"):
            st.sidebar.warning("⚠️ API key format doesn't look standard (should start with 'sk-ant-' for Anthropic)")
        st.session_state.api_key_configured = True
        st.session_state.auth_method = 'anthropic'
        return True
    elif "AWS_BEARER_TOKEN_BEDROCK" in os.environ and os.environ["AWS_BEARER_TOKEN_BEDROCK"]:
        # Store AWS token separately without mapping to ANTHROPIC_API_KEY
        st.session_state.aws_bearer_token = os.environ["AWS_BEARER_TOKEN_BEDROCK"]
        st.sidebar.success("✅ AWS_BEARER_TOKEN_BEDROCK found and configured")
        st.sidebar.info("💡 AWS token will be used with AWS Bedrock client")
        st.session_state.api_key_configured = True
        st.session_state.auth_method = 'aws'
        return True
    
    # API key input options
    api_option = st.sidebar.radio(
        "Choose API key setup method:",
        ["Load from custom .env file", "Enter API key", "Use python-dotenv"]
    )
    
    if api_option == "Load from custom .env file":
        # Custom .env file path input
        default_path = str(Path(__file__).parent.parent / ".env")
        env_file_path = st.sidebar.text_input(
            "Path to .env file:",
            value=default_path,
            help="Enter the full path to your .env file"
        )
        
        if st.sidebar.button("🔄 Load from .env file"):
            if env_file_path:
                success, message = load_env_from_path(env_file_path)
                if success:
                    # Check for authentication keys
                    if "ANTHROPIC_API_KEY" in os.environ and os.environ["ANTHROPIC_API_KEY"]:
                        st.sidebar.success(message)
                        st.sidebar.success("✅ ANTHROPIC_API_KEY found and loaded!")
                        st.session_state.api_key_configured = True
                        st.session_state.auth_method = 'anthropic'
                        st.rerun()
                        return True
                    elif "AWS_BEARER_TOKEN_BEDROCK" in os.environ and os.environ["AWS_BEARER_TOKEN_BEDROCK"]:
                        # Store AWS token separately without mapping
                        st.session_state.aws_bearer_token = os.environ["AWS_BEARER_TOKEN_BEDROCK"]
                        st.sidebar.success(message)
                        st.sidebar.success("✅ AWS_BEARER_TOKEN_BEDROCK found and configured!")
                        
                        # Auto-configure AWS region if present
                        if "AWS_REGION" in os.environ and os.environ["AWS_REGION"]:
                            st.sidebar.info(f"✅ AWS_REGION automatically set to: {os.environ['AWS_REGION']}")
                        
                        # Auto-configure LLM source if present
                        if "LLM_SOURCE" in os.environ and os.environ["LLM_SOURCE"]:
                            st.sidebar.info(f"✅ LLM_SOURCE automatically set to: {os.environ['LLM_SOURCE']}")
                            # If LLM_SOURCE is Bedrock, make sure to set BIOMNI_LLM_SOURCE
                            if os.environ["LLM_SOURCE"].lower() == "bedrock":
                                os.environ["BIOMNI_LLM_SOURCE"] = "Bedrock"
                                st.sidebar.info("✅ BIOMNI_LLM_SOURCE automatically set to: Bedrock")
                        
                        st.session_state.api_key_configured = True
                        st.session_state.auth_method = 'aws'
                        st.rerun()
                        return True
                    else:
                        st.sidebar.warning("⚠️ .env file loaded but no ANTHROPIC_API_KEY or AWS_BEARER_TOKEN_BEDROCK found")
                        # Check if LLM_SOURCE is configured
                        if "LLM_SOURCE" in os.environ:
                            st.sidebar.info(f"💡 Found LLM_SOURCE: {os.environ['LLM_SOURCE']}")
                            st.sidebar.info("   Configure the corresponding API key for this LLM source")
                else:
                    st.sidebar.error(message)
            else:
                st.sidebar.error("❌ Please enter a path to your .env file")
    
    elif api_option == "Enter API key":
        # Show a hint about where to find the API key
        with st.sidebar.expander("💡 Need help finding your API key?"):
            st.write("You can find your API key in your .env file or get one from:")
            st.write("• Anthropic: https://console.anthropic.com/")
            st.write("• AWS Bedrock: Your AWS console")
            st.info("💡 AWS Bedrock tokens will be handled separately with proper AWS authentication")
            st.warning("⚠️ Note: AWS Bedrock may require installing the langchain_aws package manually due to dependency conflicts")
            st.code("pip install langchain_aws", language="bash")
        
        # Option to choose between Anthropic and AWS
        token_type = st.sidebar.selectbox(
            "Select token type:",
            ["Anthropic API Key", "AWS Bedrock Bearer Token"]
        )
        
        if token_type == "Anthropic API Key":
            api_key = st.sidebar.text_input(
                "Enter your Anthropic API key:",
                type="password",
                help="Get your API key from https://console.anthropic.com/"
            )
            
            if api_key:
                # Validate API key format
                if not api_key.startswith("sk-ant-"):
                    st.sidebar.warning("⚠️ API key should start with 'sk-ant-'. Please check your key.")
                
                os.environ["ANTHROPIC_API_KEY"] = api_key
                st.sidebar.success("✅ Anthropic API key configured")
                st.session_state.api_key_configured = True
                st.session_state.auth_method = 'anthropic'
                return True
        else:
            aws_token = st.sidebar.text_input(
                "Enter your AWS Bedrock Bearer Token:",
                type="password",
                help="Get your bearer token from AWS Bedrock console"
            )
            
            if aws_token:
                # Store AWS token separately for proper AWS client initialization
                st.session_state.aws_bearer_token = aws_token
                os.environ["AWS_BEARER_TOKEN_BEDROCK"] = aws_token
                st.sidebar.success("✅ AWS Bedrock bearer token configured")
                st.sidebar.info("💡 Token will be used with AWS Bedrock client")
                st.session_state.api_key_configured = True
                st.session_state.auth_method = 'aws'
                return True
    
    elif api_option == "Use python-dotenv":
        try:
            parent_dir = Path(__file__).parent.parent
            env_file_path = parent_dir / ".env"
            
            if env_file_path.exists():
                load_dotenv(env_file_path, override=True)
                
                if "ANTHROPIC_API_KEY" in os.environ and os.environ["ANTHROPIC_API_KEY"]:
                    st.sidebar.success(f"✅ ANTHROPIC_API_KEY loaded from {env_file_path}")
                    st.session_state.api_key_configured = True
                    st.session_state.auth_method = 'anthropic'
                    return True
                elif "AWS_BEARER_TOKEN_BEDROCK" in os.environ and os.environ["AWS_BEARER_TOKEN_BEDROCK"]:
                    # Store AWS token separately without mapping
                    st.session_state.aws_bearer_token = os.environ["AWS_BEARER_TOKEN_BEDROCK"]
                    st.sidebar.success(f"✅ AWS_BEARER_TOKEN_BEDROCK loaded from {env_file_path}")
                    
                    # Auto-configure AWS region if present
                    if "AWS_REGION" in os.environ and os.environ["AWS_REGION"]:
                        st.sidebar.info(f"✅ AWS_REGION automatically set to: {os.environ['AWS_REGION']}")
                    
                    # Auto-configure LLM source if present
                    if "LLM_SOURCE" in os.environ and os.environ["LLM_SOURCE"]:
                        st.sidebar.info(f"✅ LLM_SOURCE automatically set to: {os.environ['LLM_SOURCE']}")
                        # If LLM_SOURCE is Bedrock, make sure to set BIOMNI_LLM_SOURCE
                        if os.environ["LLM_SOURCE"].lower() == "bedrock":
                            os.environ["BIOMNI_LLM_SOURCE"] = "Bedrock"
                            st.sidebar.info("✅ BIOMNI_LLM_SOURCE automatically set to: Bedrock")
                    
                    st.session_state.api_key_configured = True
                    st.session_state.auth_method = 'aws'
                    return True
                else:
                    st.sidebar.error(f"❌ No ANTHROPIC_API_KEY or AWS_BEARER_TOKEN_BEDROCK found in {env_file_path}")
                    with open(env_file_path, 'r') as f:
                        content = f.read()
                        if "ANTHROPIC_API_KEY" in content or "AWS_BEARER_TOKEN_BEDROCK" in content:
                            st.sidebar.info("💡 Token found in file but not loaded properly. Try 'Load from custom .env file' option.")
                        if "LLM_SOURCE" in content:
                            st.sidebar.info("💡 Found LLM_SOURCE in .env file - configure the corresponding API key")
            else:
                load_dotenv(override=True)
                if "ANTHROPIC_API_KEY" in os.environ and os.environ["ANTHROPIC_API_KEY"]:
                    st.sidebar.success("✅ ANTHROPIC_API_KEY loaded from current directory .env file")
                    st.session_state.api_key_configured = True
                    st.session_state.auth_method = 'anthropic'
                    return True
                elif "AWS_BEARER_TOKEN_BEDROCK" in os.environ and os.environ["AWS_BEARER_TOKEN_BEDROCK"]:
                    # Store AWS token separately without mapping
                    st.session_state.aws_bearer_token = os.environ["AWS_BEARER_TOKEN_BEDROCK"]
                    st.sidebar.success("✅ AWS_BEARER_TOKEN_BEDROCK loaded from current directory .env file")
                    
                    # Auto-configure AWS region if present
                    if "AWS_REGION" in os.environ and os.environ["AWS_REGION"]:
                        st.sidebar.info(f"✅ AWS_REGION automatically set to: {os.environ['AWS_REGION']}")
                    
                    # Auto-configure LLM source if present
                    if "LLM_SOURCE" in os.environ and os.environ["LLM_SOURCE"]:
                        st.sidebar.info(f"✅ LLM_SOURCE automatically set to: {os.environ['LLM_SOURCE']}")
                        # If LLM_SOURCE is Bedrock, make sure to set BIOMNI_LLM_SOURCE
                        if os.environ["LLM_SOURCE"].lower() == "bedrock":
                            os.environ["BIOMNI_LLM_SOURCE"] = "Bedrock"
                            st.sidebar.info("✅ BIOMNI_LLM_SOURCE automatically set to: Bedrock")
                    
                    st.session_state.api_key_configured = True
                    st.session_state.auth_method = 'aws'
                    return True
                else:
                    st.sidebar.error("❌ No .env file found or no ANTHROPIC_API_KEY/AWS_BEARER_TOKEN_BEDROCK in .env file")
                    # Check if at least LLM_SOURCE is available
                    if "LLM_SOURCE" in os.environ:
                        st.sidebar.info(f"💡 Found LLM_SOURCE: {os.environ['LLM_SOURCE']}")
                        st.sidebar.info("   Please configure the corresponding API key")
        except Exception as e:
            st.sidebar.error(f"❌ Error loading .env file: {str(e)}")
    
    st.sidebar.warning("⚠️ API key or AWS bearer token required to use Biomni agent")
    
    # Show current environment configuration
    with st.sidebar.expander("🔍 Current Environment Status", expanded=False):
        st.write("**Authentication Status:**")
        st.write(f"• ANTHROPIC_API_KEY: {'✅ Set' if os.environ.get('ANTHROPIC_API_KEY') else '❌ Not set'}")
        st.write(f"• AWS_BEARER_TOKEN_BEDROCK: {'✅ Set' if os.environ.get('AWS_BEARER_TOKEN_BEDROCK') else '❌ Not set'}")
        st.write(f"• AWS_REGION: {os.environ.get('AWS_REGION', '❌ Not set (will default to us-east-1)')}")
        st.write(f"**LLM Configuration:**")
        st.write(f"• LLM_SOURCE: {os.environ.get('LLM_SOURCE', '❌ Not set (auto-detect)')}")
        st.write(f"• BIOMNI_LLM_SOURCE: {os.environ.get('BIOMNI_LLM_SOURCE', '❌ Not set')}")
        st.write(f"**Session State:**")
        st.write(f"• Auth Method: {getattr(st.session_state, 'auth_method', 'Not set')}")
        st.write(f"• API Configured: {getattr(st.session_state, 'api_key_configured', False)}")
        
        # Auto-detect configuration from environment
        if os.environ.get('AWS_BEARER_TOKEN_BEDROCK') and not st.session_state.api_key_configured:
            st.info("💡 AWS Bearer Token detected - click 'Load from custom .env file' to configure automatically")
        if os.environ.get('LLM_SOURCE'):
            st.info(f"💡 LLM_SOURCE detected: {os.environ.get('LLM_SOURCE')}")
        if os.environ.get('AWS_REGION'):
            st.info(f"💡 AWS_REGION detected: {os.environ.get('AWS_REGION')}")
        
        if st.button("🔄 Refresh Environment"):
            st.rerun()
    
    # Debug section
    if st.sidebar.button("🔍 Test API Connection"):
        if st.session_state.auth_method == 'anthropic' and "ANTHROPIC_API_KEY" in os.environ and os.environ["ANTHROPIC_API_KEY"]:
            try:
                # Test the Anthropic API key with a simple call
                import anthropic
                client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
                
                # Simple test call
                response = client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=10,
                    messages=[{"role": "user", "content": "Hello"}]
                )
                st.sidebar.success("✅ Anthropic API connection test successful!")
                st.sidebar.info(f"Response: {response.content[0].text[:50]}...")
                
            except Exception as e:
                st.sidebar.error(f"❌ Anthropic API connection test failed: {str(e)}")
                st.sidebar.error("💡 Check if your API key is valid")
        elif st.session_state.auth_method == 'aws' and st.session_state.aws_bearer_token:
            try:
                # Test AWS Bedrock connection
                import boto3
                from botocore.config import Config
                
                # Configure AWS Bedrock client with bearer token
                # Note: This is a simplified test - actual AWS setup may vary
                st.sidebar.info("🔍 Testing AWS Bedrock connection...")
                st.sidebar.warning("⚠️ AWS Bedrock requires proper AWS credentials and region configuration")
                st.sidebar.info(f"Bearer token configured: {st.session_state.aws_bearer_token[:20]}...")
                
                # For now, just confirm the token is stored
                st.sidebar.success("✅ AWS bearer token is configured")
                st.sidebar.info("💡 Full AWS Bedrock connection testing requires additional AWS setup")
                
            except ImportError:
                st.sidebar.error("❌ boto3 not available for AWS testing")
                st.sidebar.info("💡 Install boto3 to test AWS Bedrock connections")
            except Exception as e:
                st.sidebar.error(f"❌ AWS connection test failed: {str(e)}")
        else:
            st.sidebar.error("❌ No valid authentication method found for testing")
            st.sidebar.error("💡 Configure either ANTHROPIC_API_KEY or AWS_BEARER_TOKEN_BEDROCK first")
    
    return False

def configure_llm_for_retriever():
    """Configure LLM settings for the retriever based on current authentication method"""
    from biomni.llm import get_llm
    
    try:
        # Determine the appropriate LLM configuration based on auth method
        if st.session_state.auth_method == 'anthropic' and "ANTHROPIC_API_KEY" in os.environ:
            # Use Anthropic with the configured API key
            llm = get_llm(
                model="claude-3-haiku-20240307",  # Use a fast model for retrieval
                source="Anthropic"
            )
            return llm
        elif st.session_state.auth_method == 'aws' and st.session_state.aws_bearer_token:
            # Use native Bedrock support
            try:
                # Set up environment for Bedrock
                os.environ["AWS_BEARER_TOKEN_BEDROCK"] = st.session_state.aws_bearer_token
                if "AWS_REGION" not in os.environ:
                    os.environ["AWS_REGION"] = "us-east-1"
                
                llm = get_llm(
                    model="anthropic.claude-3-haiku-20240307-v1:0",  # Bedrock model ID
                    source="Bedrock"
                )
                return llm
            except Exception as bedrock_e:
                st.error(f"⚠️ Bedrock LLM failed for retriever, error: {str(bedrock_e)}")
                st.error("💡 Check AWS Bedrock configuration:")
                st.error("   - Verify AWS_BEARER_TOKEN_BEDROCK is valid")
                st.error("   - Check AWS region and model availability")
                st.error("   - Ensure Bedrock permissions are correctly set")
                return None
        else:
            # Default fallback
            st.error("❌ No valid authentication configured for LLM retriever")
            return None
            
    except Exception as e:
        st.error(f"❌ Error configuring LLM for retriever: {str(e)}")
        return None

def initialize_agent():
    """Initialize the Biomni agent"""
    if not st.session_state.api_key_configured:
        return False
    
    if st.session_state.agent is None:
        # Agent configuration
        data_path = st.sidebar.text_input(
            "Data path:",
            value="~/biomni_data",
            help="Path where Biomni will store data lake files"
        )
        
        llm_model = st.sidebar.selectbox(
            "LLM Model:",
            ["claude-sonnet-4-20250514", "claude-haiku-20240320", "claude-opus-20240229"],
            help="Choose the Claude model to use"
        )
        
        # Advanced configuration
        with st.sidebar.expander("🔧 Advanced Configuration", expanded=False):
            # Manual LLM source override
            llm_source_override = st.selectbox(
                "LLM Source Override (optional):",
                ["Auto", "Anthropic", "Bedrock", "OpenAI", "Custom"],
                help="Override automatic LLM source detection"
            )
            
            if llm_source_override == "Bedrock":
                st.info("🚀 **Bedrock Enabled**: Using native AWS Bedrock support with langchain_aws")
                st.success("💡 Bedrock support is now fully enabled with proper dependencies")
            elif llm_source_override != "Auto":
                st.info(f"LLM source will be set to: {llm_source_override}")
            
            # AWS-specific configuration
            if st.session_state.auth_method == 'aws' or llm_source_override == "Bedrock":
                aws_region_override = st.text_input(
                    "AWS Region:",
                    value=os.environ.get("AWS_REGION", "us-east-1"),
                    help="AWS region for Bedrock"
                )
                if aws_region_override:
                    os.environ["AWS_REGION"] = aws_region_override
                
                # Add option to disable tool retrieval for AWS
                disable_tool_retrieval = st.checkbox(
                    "Disable Tool Retrieval",
                    value=False,
                    help="Disable LLM-based tool retrieval to avoid authentication issues. Agent will use all tools."
                )
                
                if disable_tool_retrieval:
                    os.environ["BIOMNI_DISABLE_TOOL_RETRIEVAL"] = "true"
                    st.info("🔧 Tool retrieval disabled - agent will use all available tools")
                else:
                    os.environ.pop("BIOMNI_DISABLE_TOOL_RETRIEVAL", None)
        
        if st.sidebar.button("🚀 Initialize Agent"):
            try:
                with st.spinner("Initializing Biomni agent..."):
                    # Set up environment variables for agent initialization
                    if llm_source_override != "Auto":
                        os.environ["BIOMNI_LLM_SOURCE"] = llm_source_override
                        if llm_source_override == "Bedrock":
                            os.environ["LLM_SOURCE"] = "Bedrock"
                    elif st.session_state.auth_method == 'aws':
                        # Auto-detect Bedrock usage
                        os.environ["BIOMNI_LLM_SOURCE"] = "Bedrock"
                        os.environ["LLM_SOURCE"] = "Bedrock"
                    elif st.session_state.auth_method == 'anthropic':
                        # Auto-detect Anthropic usage
                        os.environ["BIOMNI_LLM_SOURCE"] = "Anthropic"
                        os.environ["LLM_SOURCE"] = "Anthropic"
                    
                    # Ensure AWS region is set for Bedrock
                    if (llm_source_override == "Bedrock" or st.session_state.auth_method == 'aws') and "AWS_REGION" not in os.environ:
                        os.environ["AWS_REGION"] = "us-east-1"
                    from biomni.agent import A1
                    
                    # Determine LLM source from environment or manual override
                    env_llm_source = os.environ.get("LLM_SOURCE")  # Read from .env file
                    
                    if llm_source_override != "Auto":
                        final_llm_source = llm_source_override
                        os.environ["BIOMNI_LLM_SOURCE"] = final_llm_source
                        st.sidebar.info(f"🔧 Using manual LLM source: {final_llm_source}")
                    elif env_llm_source and env_llm_source in ["OpenAI", "AzureOpenAI", "Anthropic", "Gemini", "Bedrock", "Ollama", "Groq", "Custom"]:
                        final_llm_source = env_llm_source
                        os.environ["BIOMNI_LLM_SOURCE"] = final_llm_source
                        st.sidebar.info(f"📄 Using LLM_SOURCE from .env file: {final_llm_source}")
                    elif st.session_state.auth_method == 'anthropic':
                        final_llm_source = "Anthropic"
                        os.environ["BIOMNI_LLM_SOURCE"] = final_llm_source
                        st.sidebar.info(f"🔧 Auto-detected LLM source: {final_llm_source}")
                    elif st.session_state.auth_method == 'aws':
                        final_llm_source = "Bedrock"  # Will be handled as workaround below
                        os.environ["BIOMNI_LLM_SOURCE"] = final_llm_source
                        st.sidebar.info(f"🔧 Auto-detected LLM source: {final_llm_source}")
                    else:
                        final_llm_source = "Auto"
                    
                    # Handle different authentication methods
                    if final_llm_source == "Anthropic" or st.session_state.auth_method == 'anthropic':
                        if "ANTHROPIC_API_KEY" in os.environ and os.environ["ANTHROPIC_API_KEY"]:
                            key_preview = os.environ["ANTHROPIC_API_KEY"][:10] + "..." if len(os.environ["ANTHROPIC_API_KEY"]) > 10 else "short key"
                            st.sidebar.info(f"🔑 Using Anthropic API key: {key_preview}")
                            
                            # Validate key format
                            if not os.environ["ANTHROPIC_API_KEY"].startswith("sk-ant-"):
                                st.sidebar.warning("⚠️ API key format may not be standard for Anthropic API")
                            
                            # Initialize agent with Anthropic API key
                            st.session_state.agent = A1(path=data_path, llm=llm_model)
                        else:
                            st.sidebar.error("❌ No ANTHROPIC_API_KEY found in environment")
                            return False
                    
                    elif final_llm_source == "Bedrock" or st.session_state.auth_method == 'aws':
                        if st.session_state.aws_bearer_token:
                            token_preview = st.session_state.aws_bearer_token[:20] + "..." if len(st.session_state.aws_bearer_token) > 20 else "short token"
                            st.sidebar.info(f"🔑 Using AWS Bearer Token: {token_preview}")
                            
                            # Configure environment variables for AWS Bedrock
                            os.environ["AWS_BEARER_TOKEN_BEDROCK"] = st.session_state.aws_bearer_token
                            
                            # Set AWS region if not already set
                            if "AWS_REGION" not in os.environ:
                                os.environ["AWS_REGION"] = "us-east-1"  # Default region
                            
                            st.sidebar.info(f"🌍 AWS Region: {os.environ.get('AWS_REGION', 'us-east-1')}")
                            
                            # Use Bedrock directly since langchain_aws is installed and code is uncommented
                            st.sidebar.success("✅ Using native AWS Bedrock support")
                            
                            # For AWS Bedrock, modify the model name to use Bedrock format
                            # Map Claude model names to AWS Bedrock model IDs
                            bedrock_model_mapping = {
                                "claude-sonnet-4-20250514": "anthropic.claude-3-5-sonnet-20241022-v2:0",
                                "claude-haiku-20240320": "anthropic.claude-3-haiku-20240307-v1:0", 
                                "claude-opus-20240229": "anthropic.claude-3-opus-20240229-v1:0"
                            }
                            
                            bedrock_model = bedrock_model_mapping.get(llm_model, llm_model)
                            st.sidebar.info(f"🤖 Using Bedrock model: {bedrock_model}")
                            
                            # Keep BIOMNI_LLM_SOURCE as Bedrock for native support
                            os.environ["BIOMNI_LLM_SOURCE"] = "Bedrock"
                            
                            # Initialize agent with Bedrock model
                            try:
                                st.session_state.agent = A1(path=data_path, llm=bedrock_model)
                            except Exception as bedrock_error:
                                st.sidebar.error(f"⚠️ Bedrock initialization failed: {str(bedrock_error)}")
                                st.sidebar.info("💡 Bedrock troubleshooting:")
                                st.sidebar.info("   1. Verify AWS_BEARER_TOKEN_BEDROCK is valid")
                                st.sidebar.info("   2. Check AWS region configuration")
                                st.sidebar.info("   3. Ensure AWS credentials have Bedrock permissions")
                                st.sidebar.info("   4. Verify langchain_aws is properly installed")
                                raise bedrock_error
                        else:
                            st.sidebar.error("❌ No AWS bearer token found")
                            return False
                    
                    else:
                        st.sidebar.error("❌ No valid authentication method configured")
                        st.sidebar.error("💡 Configure either Anthropic API key or AWS Bearer token first")
                        return False
                    
                    st.session_state.agent_status = "Initialized successfully"
                    st.sidebar.success("✅ Agent initialized successfully")
                    st.rerun()
                    return True
            except Exception as e:
                st.sidebar.error(f"❌ Error initializing agent: {str(e)}")
                
                # Enhanced error diagnostics
                if "retriever" in str(e).lower() and "prompt_based_retrieval" in str(e):
                    st.sidebar.error("💡 Retriever LLM Error:")
                    st.sidebar.error("   - The tool retriever failed to initialize its LLM")
                    st.sidebar.error("   - This often happens with AWS authentication issues")
                    st.sidebar.error("   - Try configuring the LLM source manually in Advanced Configuration")
                    if st.session_state.auth_method == 'aws':
                        st.sidebar.error("   - Verify AWS bearer token and region settings")
                        st.sidebar.error("   - Check AWS Bedrock permissions and model access")
                elif "Invalid source: Bedrock" in str(e):
                    st.sidebar.error("💡 Bedrock Configuration Error:")
                    st.sidebar.error("   - Check if langchain_aws is properly installed")
                    st.sidebar.error("   - Verify AWS_BEARER_TOKEN_BEDROCK is correctly configured")
                    st.sidebar.error("   - Ensure AWS region is set and accessible")
                    st.sidebar.error("   - Check if Bedrock models are available in your region")
                elif "authentication" in str(e).lower():
                    st.sidebar.error("💡 Authentication Error:")
                    st.sidebar.error(f"   - Current auth method: {st.session_state.auth_method}")
                    if st.session_state.auth_method == 'aws':
                        st.sidebar.error("   - Check AWS_BEARER_TOKEN_BEDROCK configuration")
                        st.sidebar.error("   - Verify AWS region settings")
                        st.sidebar.error("   - Ensure AWS credentials have Bedrock access")
                    else:
                        st.sidebar.error("   - Check ANTHROPIC_API_KEY format and validity")
                elif "llm" in str(e).lower() or "model" in str(e).lower():
                    st.sidebar.error("💡 LLM Configuration Error:")
                    st.sidebar.error(f"   - Model: {llm_model}")
                    st.sidebar.error(f"   - LLM Source: {os.environ.get('BIOMNI_LLM_SOURCE', 'Not set')}")
                    if st.session_state.auth_method == 'aws':
                        st.sidebar.error("   - Check if Bedrock model is available in your region")
                        st.sidebar.error("   - Verify model permissions in AWS console")
                else:
                    st.sidebar.error("💡 General initialization error - check all configuration")
                
                # Show environment diagnostics
                with st.sidebar.expander("🔍 Environment Diagnostics"):
                    st.write("**Environment Variables:**")
                    st.write(f"ANTHROPIC_API_KEY: {'Set' if os.environ.get('ANTHROPIC_API_KEY') else 'Not set'}")
                    st.write(f"AWS_BEARER_TOKEN_BEDROCK: {'Set' if os.environ.get('AWS_BEARER_TOKEN_BEDROCK') else 'Not set'}")
                    st.write(f"AWS_REGION: {os.environ.get('AWS_REGION', 'Not set')}")
                    st.write(f"BIOMNI_LLM_SOURCE: {os.environ.get('BIOMNI_LLM_SOURCE', 'Not set')}")
                    st.write(f"**Session State:**")
                    st.write(f"Auth method: {st.session_state.auth_method}")
                    st.write(f"API configured: {st.session_state.api_key_configured}")
                
                st.session_state.agent_status = f"Error: {str(e)}"
                return False
        else:
            # Show initialization status but don't return False
            st.sidebar.warning("⚠️ Click 'Initialize Agent' to start")
            return False
    else:
        st.sidebar.success("✅ Agent already initialized")
        return True

def setup_synapse_connection():
    """Setup Synapse connection"""
    st.sidebar.subheader("🔗 Synapse Connection")
    
    if st.session_state.synapse_connected:
        st.sidebar.success("✅ Synapse already connected")
        if hasattr(st.session_state, 'synapse_profile'):
            st.sidebar.info(f"Profile: {st.session_state.synapse_profile}")
        return True
    
    profile_name = st.sidebar.text_input(
        "Synapse profile name:",
        value="biomni-agent-test",
        help="Name of the Synapse profile to use for authentication"
    )
    
    if st.sidebar.button("Connect to Synapse"):
        try:
            with st.spinner("Connecting to Synapse..."):
                import synapseclient
                import synapseutils
                syn = synapseclient.login(profile=profile_name)
                st.session_state.synapse_connected = True
                st.session_state.syn = syn
                st.session_state.synapse_profile = profile_name  # Store profile name
                st.sidebar.success(f"✅ Connected to Synapse as {syn.username}")
                logging.info(f"Synapse connection established for profile: {profile_name}")
                return True
        except Exception as e:
            st.sidebar.error(f"❌ Synapse connection error: {str(e)}")
            logging.error(f"Synapse connection failed: {str(e)}")
            return False
    
    return False

def display_chat_message(message_type, content, timestamp):
    """Display a chat message with enhanced formatting showing visualizations and terminal output"""
    css_class = "user-message" if message_type == "user" else "ai-message"
    icon = "👤" if message_type == "user" else "🤖"
    
    # Create a container for the message
    with st.container():
        # Message header
        st.markdown(f"""
        <div style="padding: 0.5rem; margin: 0.5rem 0; border-radius: 8px; border-left: 4px solid {'#2196f3' if message_type == 'user' else '#9c27b0'}; background-color: {'#e3f2fd' if message_type == 'user' else '#f3e5f5'};">
            <strong>{icon} {message_type.title()} - {timestamp}</strong>
        </div>
        """, unsafe_allow_html=True)
        
        if message_type == "user":
            # Simple display for user messages
            st.markdown(f"**Query:** {content}")
        else:
            # Enhanced display for AI responses with visualization and terminal output
            display_ai_response_with_media(content)

def display_structured_agent_output(content_list):
    """Display the structured agent output format with proper parsing of Human/AI messages and structured blocks"""
    
    for item in content_list:
        item_str = str(item)
        
        # Skip empty items
        if not item_str.strip():
            continue
            
        # Check for message separators
        if "=" * 30 in item_str and "Human Message" in item_str:
            st.markdown("---")
            st.markdown("### 👤 **Human Query**")
            st.markdown('<div style="background-color: #e3f2fd; padding: 8px; border-radius: 4px; border-left: 4px solid #2196f3; margin: 8px 0;"><strong>User Input</strong></div>', unsafe_allow_html=True)
            continue
        elif "=" * 30 in item_str and "Ai Message" in item_str:
            st.markdown("---") 
            st.markdown("### 🤖 **AI Response**")
            st.markdown('<div style="background-color: #f3e5f5; padding: 8px; border-radius: 4px; border-left: 4px solid #9c27b0; margin: 8px 0;"><strong>Agent Analysis</strong></div>', unsafe_allow_html=True)
            continue
        
        # Process the content based on structure
        lines = item_str.split('\n')
        
        # Check if this is a plan update section
        if any("Plan Update:" in line for line in lines[:3]):
            with st.expander("📋 **Plan Progress**", expanded=True):
                st.markdown('<div style="background-color: #f8f9fa; padding: 12px; border-radius: 6px; border: 1px solid #dee2e6;">', unsafe_allow_html=True)
                for line in lines:
                    if line.strip():
                        # Convert checkboxes to proper markdown with better styling
                        if "[✓]" in line:
                            line = line.replace("[✓]", "✅").replace("✓]", "✅")
                            st.markdown(f'<div style="color: #28a745; font-weight: bold;">{line}</div>', unsafe_allow_html=True)
                        elif "[ ]" in line:
                            line = line.replace("[ ]", "⬜")
                            st.markdown(f'<div style="color: #6c757d;">{line}</div>', unsafe_allow_html=True)
                        else:
                            st.markdown(line)
                st.markdown('</div>', unsafe_allow_html=True)
            continue
        
        # Check for structured blocks (execute, observation, solution)
        if any(tag in item_str for tag in ['<execute>', '<observation>', '<solution>']):
            parse_structured_blocks(item_str)
        else:
            # Regular content - parse for code, visualizations, etc.
            parse_regular_content(item_str)

def parse_structured_blocks(content):
    """Parse content with <execute>, <observation>, <solution> blocks"""
    lines = content.split('\n')
    i = 0
    
    while i < len(lines):
        line = lines[i].strip()
        
        if line.startswith('<execute>'):
            # Code execution block
            code_content = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith('</execute>'):
                code_content.append(lines[i])
                i += 1
            
            if code_content:
                with st.expander("🐍 **Code Execution**", expanded=True):
                    st.markdown('<div style="background-color: #e3f2fd; padding: 8px; border-radius: 4px; border-left: 4px solid #2196f3; margin: 4px 0;"><strong>Python Code</strong></div>', unsafe_allow_html=True)
                    st.code('\n'.join(code_content), language='python')
                    
        elif line.startswith('<observation>'):
            # Observation block
            obs_content = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith('</observation>'):
                obs_content.append(lines[i])
                i += 1
                
            if obs_content:
                with st.expander("👁️ **Observation**", expanded=True):
                    st.markdown('<div style="background-color: #f3e5f5; padding: 8px; border-radius: 4px; border-left: 4px solid #9c27b0; margin: 4px 0;"><strong>Execution Results</strong></div>', unsafe_allow_html=True)
                    obs_text = '\n'.join(obs_content)
                    
                    # Check for visualizations in observation
                    if any(ext in obs_text.lower() for ext in ['.png', '.jpg', '.jpeg', '.svg']):
                        parse_content_with_visualizations(obs_text)
                    else:
                        st.markdown(obs_text)
                        
        elif line.startswith('<solution>'):
            # Solution block
            sol_content = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith('</solution>'):
                sol_content.append(lines[i])
                i += 1
                
            if sol_content:
                with st.expander("✅ **Solution**", expanded=True):
                    st.markdown('<div style="background-color: #d4edda; padding: 8px; border-radius: 4px; border-left: 4px solid #28a745; margin: 4px 0;"><strong>Final Analysis</strong></div>', unsafe_allow_html=True)
                    sol_text = '\n'.join(sol_content)
                    st.markdown(sol_text)
        else:
            # Regular line - display as markdown
            if line:
                st.markdown(line)
        
        i += 1

def parse_content_with_visualizations(content):
    """Parse content and extract/display visualizations"""
    lines = content.split('\n')
    text_content = []
    
    for line in lines:
        # Check for visualization mentions
        if any(ext in line.lower() for ext in ['.png', '.jpg', '.jpeg', '.svg']) and any(keyword in line.lower() for keyword in ['saved', 'generated', 'created', 'output', 'plot', 'boxplot']):
            # Display accumulated text first
            if text_content:
                st.markdown('\n'.join(text_content))
                text_content = []
            
            # Extract and display visualization
            file_patterns = [
                r"'([^']+\.(png|jpg|jpeg|svg))'",  # Single quoted: 'filename.png'
                r'"([^"]+\.(png|jpg|jpeg|svg))"',  # Double quoted: "filename.png"
                r'([a-zA-Z0-9_]+\.(?:png|jpg|jpeg|svg))',  # Unquoted filenames
                r'(?:saved|created|generated|output|boxplot).*?[\'"]?([^\s\'"]+\.(png|jpg|jpeg|svg))[\'"]?',  # After keywords
                r'([^\s\'"]+\.(png|jpg|jpeg|svg))',  # Basic file paths
            ]
            
            filepath = None
            for pattern in file_patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    filepath = match.group(1)
                    break
            
            if filepath:
                filepath = filepath.strip('\'"').strip()
                display_visualization(filepath, line)
            else:
                text_content.append(line)
        else:
            text_content.append(line)
    
    # Display any remaining text
    if text_content:
        st.markdown('\n'.join(text_content))

def parse_regular_content(content):
    """Parse regular content for various elements"""
    lines = content.split('\n')
    current_content = []
    
    for line in lines:
        line_stripped = line.strip()
        
        # Skip empty lines at the start
        if not line_stripped and not current_content:
            continue
            
        # Check for section headers (## )
        if line_stripped.startswith('## '):
            # Display accumulated content first
            if current_content:
                st.markdown('\n'.join(current_content))
                current_content = []
            # Display header
            st.markdown(f"### {line_stripped[3:]}")
            
        # Check for major sections with markdown headers
        elif line_stripped.startswith('### '):
            if current_content:
                st.markdown('\n'.join(current_content))
                current_content = []
            st.markdown(f"#### {line_stripped[4:]}")
            
        # Check for bullet points or analysis sections
        elif line_stripped.startswith('**') and line_stripped.endswith('**'):
            if current_content:
                st.markdown('\n'.join(current_content))
                current_content = []
            st.markdown(f"**{line_stripped}**")
            
        # Check for visualizations
        elif any(ext in line.lower() for ext in ['.png', '.jpg', '.jpeg', '.svg']) and any(keyword in line.lower() for keyword in ['saved', 'generated', 'created', 'output', 'plot', 'boxplot']):
            if current_content:
                st.markdown('\n'.join(current_content))
                current_content = []
            parse_content_with_visualizations(line)
            
        else:
            current_content.append(line)
    
    # Display any remaining content
    if current_content:
        st.markdown('\n'.join(current_content))

def display_ai_response_with_media(content):
    """Display AI response with inline visualizations and terminal-style output"""
    
    # Debug section to understand content structure
    if st.sidebar.checkbox("🔍 Debug Mode", key="debug_ai_response"):
        with st.expander("🔍 Raw Content Analysis", expanded=False):
            st.write("**Content Type:**", type(content).__name__)
            if hasattr(content, '__len__'):
                st.write("**Content Length:**", len(content))
            st.text_area("Raw Content", value=str(content)[:2000], height=200, disabled=True)
            
            # Test image detection patterns
            content_str = str(content)
            st.write("**Image Detection Test:**")
            
            # Test patterns
            image_patterns = [
                r"top_10_genes_boxplot\.png",
                r"top10_genes_boxplot\.png", 
                r"([^\s\'\"]+\.png)",
                r"saved.*\.png",
                r"generated.*\.png",
                r"Plot saved.*\.png"
            ]
            
            for i, pattern in enumerate(image_patterns):
                matches = re.findall(pattern, content_str, re.IGNORECASE)
                st.write(f"Pattern {i+1}: `{pattern}` → {matches}")
    
    # Handle the specific agent output format
    if isinstance(content, (list, tuple)) and len(content) > 0:
        # Check for structured agent output format
        content_has_structure = False
        for item in content:
            item_str = str(item)
            if ("Human Message" in item_str or "Ai Message" in item_str or 
                "=" * 20 in item_str or "<execute>" in item_str or 
                "<observation>" in item_str or "<solution>" in item_str):
                content_has_structure = True
                break
        
        if content_has_structure:
            # This is the structured agent output format
            display_structured_agent_output(content)
            return
        else:
            # Regular list content, join it
            content_str = '\n'.join(str(item) for item in content)
    else:
        content_str = str(content)
    
    # Always provide a raw content fallback for debugging
    with st.expander("🔍 Show Raw Agent Response", expanded=False):
        if isinstance(content, (list, tuple)):
            st.write("**Content Type:** List/Tuple")
            st.write(f"**Length:** {len(content)}")
            for i, item in enumerate(content):
                st.write(f"**Item {i+1}:**")
                st.text_area(f"Item {i+1}", value=str(item), height=150, disabled=True, key=f"raw_content_{i}")
        else:
            st.write("**Content Type:** String/Other")
            st.text_area("Complete Raw Response:", value=str(content), height=300, disabled=True)
    
    # Check if any PNG files exist and show them unconditionally if debug mode is on
    if st.sidebar.checkbox("🖼️ Show Available Images", key="show_available_images"):
        png_files = list(Path.cwd().glob("*.png"))
        if png_files:
            st.subheader("📊 Available Visualizations")
            for png_file in png_files:
                with st.expander(f"Image: {png_file.name}", expanded=True):
                    try:
                        st.image(str(png_file), caption=png_file.name, use_column_width=True)
                        st.caption(f"File: {png_file} | Size: {png_file.stat().st_size:,} bytes")
                    except Exception as e:
                        st.error(f"Error displaying {png_file.name}: {e}")
    
    # If processed content is significantly shorter than original, show a warning
    if hasattr(content, '__len__') and len(str(content)) > 100:
        st.info("💡 If the formatting above doesn't look correct, check the raw response in the expandable section.")
    
    # Parse regular content
    lines = content_str.split('\n')
    
    # Initialize containers for different content types
    current_text = []
    current_code = []
    current_terminal = []
    in_code_block = False
    
    # Process content to separate text, code, and terminal output
    i = 0
    while i < len(lines):
        line = lines[i]
        line_stripped = line.strip()
        
        # Skip message separators
        if "=" * 30 in line and ("Human Message" in line or "Ai Message" in line):
            i += 1
            continue
        
        # Check for visualizations and display them immediately
        # Enhanced pattern to catch various ways files might be mentioned
        visualization_detected = False
        if any(ext in line.lower() for ext in ['.png', '.jpg', '.jpeg', '.svg']):
            # Look for various patterns that indicate a file was created/saved
            keywords = ['saved', 'generated', 'created', 'output', 'plot', 'figure', 'boxplot', 'visualization']
            if any(keyword in line.lower() for keyword in keywords) or 'top_10_genes' in line.lower() or 'top10_genes' in line.lower():
                visualization_detected = True
            
        if visualization_detected:
            # Flush current content
            if current_text:
                st.markdown('\n'.join(current_text))
                current_text = []
            if current_terminal:
                display_terminal_output('\n'.join(current_terminal))
                current_terminal = []
            
            # Enhanced pattern matching for various file reference formats
            file_patterns = [
                r"'([^']+\.(png|jpg|jpeg|svg))'",  # Single quoted: 'filename.png'
                r'"([^"]+\.(png|jpg|jpeg|svg))"',  # Double quoted: "filename.png"
                r'(top_?1[0-9]_genes[^\s]*\.png)',  # Specific pattern for the known files
                r'([a-zA-Z0-9_]+\.(?:png|jpg|jpeg|svg))',  # Unquoted filenames
                r'(?:saved|created|generated|output|boxplot|plot).*?([a-zA-Z0-9_]+\.(png|jpg|jpeg|svg))',  # After keywords
                r'([^\s\'"]+\.(png|jpg|jpeg|svg))',  # Basic file paths
            ]
            
            filepath = None
            for pattern in file_patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    # Get the filename from the appropriate capture group
                    if len(match.groups()) >= 2 and match.group(2):  # For patterns with extension group
                        filepath = match.group(1)
                    else:
                        filepath = match.group(1)
                    break
            
            if filepath:
                # Clean up the filepath (remove quotes, extra characters)
                filepath = filepath.strip('\'"').strip()
                display_visualization(filepath, line)
            else:
                # If no file path found, show the line as informational and try to find any PNG files
                st.info(f"🖼️ Visualization mentioned: {line.strip()}")
                # Check if there are any PNG files in current directory
                png_files = list(Path.cwd().glob("*.png"))
                if png_files:
                    st.info(f"💡 Found PNG files in directory: {[f.name for f in png_files]}")
                    # Show the most recently modified PNG file
                    latest_png = max(png_files, key=lambda x: x.stat().st_mtime)
                    display_visualization(str(latest_png), f"Latest PNG file: {latest_png.name}")
            i += 1
            continue
        
        # Handle special blocks: <execute>, <observation>, <solution>
        elif line_stripped.startswith('<execute>') or line_stripped.startswith('<observation>') or line_stripped.startswith('<solution>'):
            # Flush current content
            if current_text:
                st.markdown('\n'.join(current_text))
                current_text = []
            if current_terminal:
                display_terminal_output('\n'.join(current_terminal))
                current_terminal = []
            
            # Determine block type and styling
            if line_stripped.startswith('<execute>'):
                block_type = "execute"
                block_title = "🐍 Code Execution"
                block_color = "#e3f2fd"
                border_color = "#2196f3"
            elif line_stripped.startswith('<observation>'):
                block_type = "observation"
                block_title = "📋 Observation"
                block_color = "#f3e5f5"
                border_color = "#9c27b0"
            else:  # solution
                block_type = "solution"
                block_title = "✅ Solution"
                block_color = "#e8f5e8"
                border_color = "#4caf50"
            
            # Collect block content
            block_content = []
            i += 1
            while i < len(lines):
                if lines[i].strip().startswith(f'</{block_type}>'):
                    break
                block_content.append(lines[i])
                i += 1
            
            # Display the block with proper formatting
            if block_content:
                content_text = '\n'.join(block_content)
                
                # Check for visualizations within the block
                has_visualization = any(ext in content_text.lower() for ext in ['.png', '.jpg', '.jpeg', '.svg'])
                
                # Create expandable section
                with st.expander(block_title, expanded=True):
                    if block_type == "execute":
                        st.code(content_text, language='python')
                    else:
                        # Check for visualizations in observation blocks
                        if has_visualization and block_type == "observation":
                            # Parse for file paths and display visualizations
                            viz_lines = content_text.split('\n')
                            text_content = []
                            for viz_line in viz_lines:
                                if any(ext in viz_line.lower() for ext in ['.png', '.jpg', '.jpeg', '.svg']) and any(keyword in viz_line.lower() for keyword in ['saved', 'generated', 'created', 'output', 'plot', 'boxplot']):
                                    # Display any accumulated text first
                                    if text_content:
                                        st.markdown('\n'.join(text_content))
                                        text_content = []
                                    
                                    # Extract and display visualization
                                    file_patterns = [
                                        r"'([^']+\.(png|jpg|jpeg|svg))'",  # Single quoted: 'filename.png'
                                        r'"([^"]+\.(png|jpg|jpeg|svg))"',  # Double quoted: "filename.png"
                                        r'([a-zA-Z0-9_]+\.(?:png|jpg|jpeg|svg))',  # Unquoted filenames
                                        r'(?:saved|created|generated|output|boxplot).*?[\'"]?([^\s\'"]+\.(png|jpg|jpeg|svg))[\'"]?',  # After keywords
                                        r'([^\s\'"]+\.(png|jpg|jpeg|svg))',  # Basic file paths
                                    ]
                                    
                                    filepath = None
                                    for pattern in file_patterns:
                                        match = re.search(pattern, viz_line, re.IGNORECASE)
                                        if match:
                                            filepath = match.group(1)
                                            break
                                    
                                    if filepath:
                                        filepath = filepath.strip('\'"').strip()
                                        display_visualization(filepath, viz_line)
                                    else:
                                        text_content.append(viz_line)
                                else:
                                    text_content.append(viz_line)
                            
                            # Display any remaining text
                            if text_content:
                                st.markdown('\n'.join(text_content))
                        else:
                            # Regular markdown display
                            st.markdown(content_text)
            
            i += 1
            continue
            # Flush current content
            if current_text:
                st.markdown('\n'.join(current_text))
                current_text = []
            if current_terminal:
                display_terminal_output('\n'.join(current_terminal))
                current_terminal = []
            
            if not in_code_block:
                in_code_block = True
                i += 1
                continue
            else:
                # End of code block
                if current_code:
                    display_code_block('\n'.join(current_code))
                    current_code = []
                in_code_block = False
                i += 1
                continue
        
        # Handle shell commands and their output
        elif (line_stripped.startswith('$') or 
              any(line_stripped.startswith(cmd) for cmd in ['ls ', 'echo ', 'cat ', 'head ', 'tail ', 'wc ', 'grep ', 'find ', 'python -c'])):
            
            # Flush current content
            if current_text:
                st.markdown('\n'.join(current_text))
                current_text = []
            
            # Add shell command to terminal output
            current_terminal.append(line)
            
            # Look for command output in following lines
            i += 1
            while i < len(lines) and lines[i].strip() and not lines[i].strip().startswith(('$', 'INFO:', 'WARNING:', 'ERROR:', '>>>', '```', 'import ', 'from ', 'def ', 'class ')):
                current_terminal.append(lines[i])
                i += 1
                if len(current_terminal) > 50:  # Limit to prevent overflow
                    break
            
            # Display terminal content immediately
            display_terminal_output('\n'.join(current_terminal))
            current_terminal = []
            continue
        
        # Handle log messages
        elif line_stripped.startswith(('INFO:', 'WARNING:', 'ERROR:', 'DEBUG:')):
            # Flush current content
            if current_text:
                st.markdown('\n'.join(current_text))
                current_text = []
            
            current_terminal.append(line)
            i += 1
            continue
        
        # Handle code content
        elif in_code_block:
            current_code.append(line)
        
        # Handle Python code lines
        elif (line_stripped.startswith(('import ', 'from ', 'def ', 'class ')) or 
              'print(' in line or line_stripped.startswith('>>>')):
            # Flush current text
            if current_text:
                st.markdown('\n'.join(current_text))
                current_text = []
            if current_terminal:
                display_terminal_output('\n'.join(current_terminal))
                current_terminal = []
            
            # Start collecting code
            current_code.append(line)
            
            # Look for more code lines
            i += 1
            while i < len(lines) and (lines[i].startswith('    ') or lines[i].strip().startswith(('print(', '>>>', 'import ', 'from '))):
                current_code.append(lines[i])
                i += 1
            
            # Display code block
            display_code_block('\n'.join(current_code))
            current_code = []
            continue
        
        # Regular text content
        else:
            if line.strip():  # Skip empty lines
                current_text.append(line)
        
        i += 1
    
    # Flush any remaining content
    if current_text:
        st.markdown('\n'.join(current_text))
    if current_code:
        display_code_block('\n'.join(current_code))
    if current_terminal:
        display_terminal_output('\n'.join(current_terminal))
    
    # Auto-detect and display any PNG files that might have been created recently
    try:
        png_files = list(Path.cwd().glob("*.png"))
        if png_files:
            # Sort by modification time, most recent first
            png_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            # Check if any PNG files contain keywords that suggest they're relevant
            relevant_files = []
            for png_file in png_files:
                if any(keyword in png_file.name.lower() for keyword in ['boxplot', 'plot', 'chart', 'graph', 'visualization', 'genes']):
                    relevant_files.append(png_file)
            
            # Display relevant files
            if relevant_files:
                st.markdown("---")
                st.subheader("📊 Generated Visualizations")
                for png_file in relevant_files[:3]:  # Show up to 3 most relevant files
                    try:
                        with st.expander(f"📈 {png_file.name}", expanded=True):
                            st.image(str(png_file), caption=f"Generated: {png_file.name}", use_column_width=True)
                            file_size = png_file.stat().st_size
                            mod_time = datetime.fromtimestamp(png_file.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                            st.caption(f"File: {png_file.name} | Size: {file_size:,} bytes | Modified: {mod_time}")
                            
                            # Download button
                            with open(png_file, 'rb') as f:
                                file_data = f.read()
                                st.download_button(
                                    "📥 Download Image",
                                    file_data,
                                    file_name=png_file.name,
                                    mime="image/png",
                                    key=f"auto_viz_download_{png_file.name}_{hash(str(png_file))}"
                                )
                    except Exception as e:
                        st.error(f"Error displaying {png_file.name}: {e}")
    except Exception as e:
        pass  # Silently ignore errors in auto-detection
    
    # Fallback: if content seems to be missing or very short, show raw content
    processed_content_length = len('\n'.join(current_text)) + len('\n'.join(current_code)) + len('\n'.join(current_terminal))
    original_content_length = len(content_str)
    
    # If processed content is significantly shorter than original, show a raw view as backup
    if original_content_length > 100 and processed_content_length < original_content_length * 0.5:
        with st.expander("🔍 Show Full Raw Response", expanded=False):
            st.text_area("Complete AI Response:", value=content_str, height=300, disabled=True)

def display_visualization(filepath, context_line):
    """Display a visualization file inline in the chat"""
    from pathlib import Path
    
    # Handle both absolute and relative paths
    file_path = Path(filepath)
    
    # If relative path, try current working directory and common subdirectories
    if not file_path.is_absolute():
        possible_paths = [
            Path.cwd() / filepath,  # Current directory
            Path.cwd() / "outputs" / filepath,  # Common outputs folder
            Path.cwd() / "plots" / filepath,  # Common plots folder
            Path.cwd() / "figures" / filepath,  # Common figures folder
            Path(filepath)  # Original path
        ]
        
        for path in possible_paths:
            if path.exists():
                file_path = path
                break
    
    if file_path.exists():
        try:
            # Create an expandable section for the visualization
            with st.expander(f"📊 Visualization: {file_path.name}", expanded=True):
                st.image(str(file_path), caption=f"Generated: {file_path.name}", use_column_width=True)
                
                # Show file info and context
                file_size = file_path.stat().st_size
                st.caption(f"File: {file_path.name} | Size: {file_size:,} bytes | Path: {file_path}")
                if context_line.strip():
                    st.caption(f"Context: {context_line.strip()}")
                
                # Download button
                with open(file_path, 'rb') as f:
                    file_data = f.read()
                    mime_type = "image/png" if file_path.suffix.lower() == '.png' else f"image/{file_path.suffix.lower().lstrip('.')}"
                    st.download_button(
                        "📥 Download Image",
                        file_data,
                        file_name=file_path.name,
                        mime=mime_type,
                        key=f"viz_download_{file_path.name}_{hash(str(file_path))}"
                    )
        except Exception as e:
            st.error(f"Could not display visualization {file_path.name}: {str(e)}")
    else:
        st.warning(f"⚠️ Visualization file not found: {filepath}")
        # Show the context line anyway
        if context_line.strip():
            st.info(f"Context: {context_line.strip()}")
        
        # Try to show what files are available in current directory
        try:
            current_files = list(Path.cwd().glob("*.png")) + list(Path.cwd().glob("*.jpg")) + list(Path.cwd().glob("*.jpeg"))
            if current_files:
                st.info(f"Available image files in current directory: {[f.name for f in current_files[:5]]}")
        except:
            pass

def display_code_block(code_content):
    """Display code in a syntax-highlighted block"""
    st.code(code_content, language='python')

def display_terminal_output(terminal_content):
    """Display terminal/shell output in a terminal-like format"""
    # Create terminal-style display
    st.markdown(f"""
    <div style="background-color: #0d1117; color: #c9d1d9; padding: 12px; border-radius: 6px; 
                font-family: 'SF Mono', Monaco, 'Cascadia Code', 'Roboto Mono', Consolas, 'Courier New', monospace; 
                font-size: 13px; line-height: 1.45; margin: 8px 0; border: 1px solid #30363d;">
        <div style="display: flex; align-items: center; margin-bottom: 8px; padding-bottom: 8px; border-bottom: 1px solid #30363d;">
            <div style="width: 12px; height: 12px; border-radius: 50%; background-color: #ff6058; margin-right: 8px;"></div>
            <div style="width: 12px; height: 12px; border-radius: 50%; background-color: #ffbd2e; margin-right: 8px;"></div>
            <div style="width: 12px; height: 12px; border-radius: 50%; background-color: #28ca42; margin-right: 12px;"></div>
            <span style="color: #8b949e; font-size: 12px;">Terminal Output</span>
        </div>
        <pre style="margin: 0; white-space: pre-wrap; word-wrap: break-word;">{terminal_content}</pre>
    </div>
    """, unsafe_allow_html=True)

def display_terminal_logs(log_entries):
    """Display log entries in a terminal-like format"""
    if not log_entries:
        return
    
    # Combine all log entries into terminal format
    terminal_content = []
    for log_entry in reversed(log_entries):  # Show newest first
        # Format log entry for terminal display
        level_color = {
            'INFO': '#58a6ff',
            'WARNING': '#f1e05a', 
            'ERROR': '#f85149',
            'CRITICAL': '#da3633',
            'DEBUG': '#8b949e'
        }.get(log_entry['level'], '#c9d1d9')
        
        # Format timestamp and level
        prefix = f"[{log_entry['timestamp']}] {log_entry['level']}"
        message = log_entry['message']
        
        # Color-code the level
        terminal_content.append(f'<span style="color: {level_color};">{prefix}</span>: {message}')
    
    # Display in terminal format
    terminal_text = '\n'.join(terminal_content)
    st.markdown(f"""
    <div style="background-color: #0d1117; color: #c9d1d9; padding: 12px; border-radius: 6px; 
                font-family: 'SF Mono', Monaco, 'Cascadia Code', 'Roboto Mono', Consolas, 'Courier New', monospace; 
                font-size: 13px; line-height: 1.45; margin: 8px 0; border: 1px solid #30363d; max-height: 500px; overflow-y: auto;">
        <div style="display: flex; align-items: center; margin-bottom: 8px; padding-bottom: 8px; border-bottom: 1px solid #30363d;">
            <div style="width: 12px; height: 12px; border-radius: 50%; background-color: #ff6058; margin-right: 8px;"></div>
            <div style="width: 12px; height: 12px; border-radius: 50%; background-color: #ffbd2e; margin-right: 8px;"></div>
            <div style="width: 12px; height: 12px; border-radius: 50%; background-color: #28ca42; margin-right: 12px;"></div>
            <span style="color: #8b949e; font-size: 12px;">Biomni Agent Terminal</span>
        </div>
        <div style="white-space: pre-wrap; word-wrap: break-word;">{terminal_text}</div>
    </div>
    """, unsafe_allow_html=True)

def display_generated_files_from_logs(log_entries):
    """Scan log entries for generated files and display them"""
    recent_files = []
    
    for log_entry in log_entries:
        message = log_entry['message'].lower()
        if any(ext in message for ext in ['.png', '.jpg', '.jpeg', '.pdf', '.csv', '.json']):
            import re
            file_pattern = r'([^\s]+\.(png|jpg|jpeg|pdf|csv|json))'
            matches = re.findall(file_pattern, log_entry['message'], re.IGNORECASE)
            for match in matches:
                file_path = Path(match[0])
                if file_path.exists():
                    recent_files.append(str(file_path))
    
    # Display recent generated files
    if recent_files:
        st.markdown("---")
        st.markdown("**📁 Recently Generated Files:**")
        
        for file_path in list(set(recent_files))[:5]:  # Show up to 5 unique files
            file_path_obj = Path(file_path)
            file_size = file_path_obj.stat().st_size if file_path_obj.exists() else 0
            
            col_file1, col_file2 = st.columns([3, 1])
            
            with col_file1:
                if file_path_obj.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                    st.markdown(f"🖼️ **{file_path_obj.name}** ({file_size:,} bytes)")
                    try:
                        st.image(str(file_path_obj), caption=file_path_obj.name, use_column_width=True)
                    except:
                        st.error(f"Could not display image: {file_path_obj.name}")
                elif file_path_obj.suffix.lower() == '.pdf':
                    st.markdown(f"📄 **{file_path_obj.name}** ({file_size:,} bytes)")
                    st.info("PDF file generated - download to view")
                elif file_path_obj.suffix.lower() in ['.csv', '.json']:
                    st.markdown(f"📊 **{file_path_obj.name}** ({file_size:,} bytes)")
                    if file_path_obj.suffix.lower() == '.csv':
                        try:
                            import pandas as pd
                            df = pd.read_csv(file_path_obj)
                            st.dataframe(df.head(), use_container_width=True)
                        except:
                            st.info("CSV file generated - download to view")
                else:
                    st.markdown(f"📁 **{file_path_obj.name}** ({file_size:,} bytes)")
            
            with col_file2:
                if file_path_obj.exists():
                    with open(file_path_obj, 'rb') as f:
                        st.download_button(
                            "📥 Download",
                            f.read(),
                            file_name=file_path_obj.name,
                            key=f"download_{file_path_obj.name}_{hash(str(file_path_obj))}"
                        )

def display_terminal_placeholder():
    """Display a placeholder when no logs are available"""
    st.markdown("""
    <div style="background-color: #0d1117; color: #c9d1d9; padding: 12px; border-radius: 6px; 
                font-family: 'SF Mono', Monaco, 'Cascadia Code', 'Roboto Mono', Consolas, 'Courier New', monospace; 
                font-size: 13px; line-height: 1.45; margin: 8px 0; border: 1px solid #30363d;">
        <div style="display: flex; align-items: center; margin-bottom: 8px; padding-bottom: 8px; border-bottom: 1px solid #30363d;">
            <div style="width: 12px; height: 12px; border-radius: 50%; background-color: #ff6058; margin-right: 8px;"></div>
            <div style="width: 12px; height: 12px; border-radius: 50%; background-color: #ffbd2e; margin-right: 8px;"></div>
            <div style="width: 12px; height: 12px; border-radius: 50%; background-color: #28ca42; margin-right: 12px;"></div>
            <span style="color: #8b949e; font-size: 12px;">Biomni Agent Terminal</span>
        </div>
        <div style="color: #8b949e; text-align: center; padding: 20px;">
            <p>🔍 No logs yet. Submit a task to see real-time execution output!</p>
            <p style="margin-top: 12px; font-size: 11px;">You'll see:</p>
            <p style="font-size: 11px;">• 🚀 Task initialization and setup</p>
            <p style="font-size: 11px;">• 📁 Data loading and file operations</p>
            <p style="font-size: 11px;">• 🔄 Synapse operations and downloads</p>
            <p style="font-size: 11px;">• 📊 Analysis progress and results</p>
            <p style="font-size: 11px;">• ✅ Success messages and completions</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

def format_message_content(content, message_type):
    """Enhanced format message content for better readability including shell output"""
    if message_type == "user":
        return content.replace('\n', '<br>')
    
    # For AI messages, format the log content with enhanced shell output support
    content_str = str(content)
    
    # Check for and display visualization files
    image_extensions = ['.png', '.jpg', '.jpeg', '.svg', '.pdf']
    lines = content_str.split('\n')
    formatted_lines = []
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Check if line mentions a file that might be a visualization
        file_found = False
        for ext in image_extensions:
            if ext in line.lower() and ('saved' in line.lower() or 'generated' in line.lower() or 'created' in line.lower()):
                # Try to extract filename and display it
                file_pattern = r'([^\s\'\"]+\.(png|jpg|jpeg|svg|pdf))'
                match = re.search(file_pattern, line, re.IGNORECASE)
                if match:
                    filename = match.group(1)
                    formatted_lines.append(f'<strong>📊 Visualization Generated:</strong> {filename}')
                    # Try to display the image if it exists
                    try:
                        file_path = Path(filename)
                        # Check multiple possible paths
                        possible_paths = [
                            file_path,
                            Path.cwd() / filename,
                            Path.cwd() / file_path.name
                        ]
                        
                        image_path = None
                        for path in possible_paths:
                            if path.exists() and path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                                image_path = path
                                break
                        
                        if image_path:
                            base64_img = get_image_base64(image_path)
                            if base64_img:
                                formatted_lines.append(f'<img src="data:image/png;base64,{base64_img}" style="max-width:100%; height:auto; margin:10px 0;"/>')
                            else:
                                formatted_lines.append(f'<p>⚠️ Could not encode image: {image_path.name}</p>')
                        else:
                            formatted_lines.append(f'<p>⚠️ Image file not found: {filename}</p>')
                    except Exception as e:
                        formatted_lines.append(f'<p>❌ Error displaying image: {str(e)}</p>')
                    file_found = True
                    break
        
        if file_found:
            i += 1
            continue
        
        # Enhanced shell command and output detection
        if line.strip().startswith('$') or any(line.strip().startswith(cmd) for cmd in ['ls ', 'echo ', 'cat ', 'head ', 'tail ', 'wc ', 'grep ', 'find ', 'python -c']):
            # Shell command
            cmd = line.strip().lstrip('$ ')
            formatted_lines.append(f'<div style="background:#fff3cd; padding:8px; border-left:4px solid #ffc107; margin:4px 0; border-radius:4px;"><strong>🔧 Shell Command:</strong><br><code style="background:#212529; color:#f8f9fa; padding:4px 6px; border-radius:3px; font-family:monospace;">$ {cmd}</code></div>')
            
            # Look for output in next lines
            i += 1
            output_lines = []
            while i < len(lines) and lines[i].strip() and not lines[i].strip().startswith(('$', 'INFO:', 'WARNING:', 'ERROR:', '>>>', '```', 'import ', 'from ', 'def ', 'class ')):
                output_lines.append(lines[i])
                i += 1
                if len(output_lines) > 20:  # Limit output lines
                    break
            
            if output_lines:
                output_text = '\n'.join(output_lines)
                formatted_lines.append(f'<div style="background:#f8f9fa; padding:8px; border-left:4px solid #007bff; margin:4px 0; border-radius:4px;"><strong>📤 Output:</strong><br><pre style="background:#212529; color:#f8f9fa; padding:8px; border-radius:3px; margin:4px 0; font-family:monospace; white-space:pre-wrap; overflow-x:auto;">{output_text}</pre></div>')
            
            continue
        
        # Handle specific output patterns
        elif 'SHELL_OUTPUT:' in line:
            output = line.split('SHELL_OUTPUT:', 1)[1].strip()
            formatted_lines.append(f'<div style="background:#f8f9fa; padding:8px; border-left:4px solid #007bff; margin:4px 0; border-radius:4px;"><strong>📤 Shell Output:</strong><br><pre style="background:#212529; color:#f8f9fa; padding:8px; border-radius:3px; margin:4px 0; font-family:monospace; white-space:pre-wrap;">{output}</pre></div>')
        elif 'PYTHON_OUTPUT:' in line:
            output = line.split('PYTHON_OUTPUT:', 1)[1].strip()
            formatted_lines.append(f'<div style="background:#f0fff0; padding:8px; border-left:4px solid #28a745; margin:4px 0; border-radius:4px;"><strong>🐍 Python Output:</strong><br><pre style="background:#212529; color:#f8f9fa; padding:8px; border-radius:3px; margin:4px 0; font-family:monospace; white-space:pre-wrap;">{output}</pre></div>')
        
        # Format different types of log messages
        elif line.strip().startswith('INFO:') or line.strip().startswith('WARNING:') or line.strip().startswith('ERROR:'):
            parts = line.split(':', 2)
            if len(parts) >= 3:
                level = parts[1].strip()
                message = parts[2].strip()
                color = {'INFO': '#17a2b8', 'WARNING': '#ffc107', 'ERROR': '#dc3545'}.get(level, '#000')
                icon = {'INFO': 'ℹ️', 'WARNING': '⚠️', 'ERROR': '❌'}.get(level, '📝')
                formatted_lines.append(f'<div style="background:{color}15; padding:6px; border-left:3px solid {color}; margin:2px 0; border-radius:3px;"><span style="color:{color}"><strong>{icon} [{level}]</strong> {message}</span></div>')
            else:
                formatted_lines.append(line)
        elif 'Task completed' in line or 'Success' in line or '✅' in line:
            formatted_lines.append(f'<div style="background:#d4edda; padding:6px; border-left:3px solid #28a745; margin:2px 0; border-radius:3px;"><span style="color:#28a745"><strong>✅ {line}</strong></span></div>')
        elif 'Error' in line or 'Failed' in line or '❌' in line:
            formatted_lines.append(f'<div style="background:#f8d7da; padding:6px; border-left:3px solid #dc3545; margin:2px 0; border-radius:3px;"><span style="color:#dc3545"><strong>❌ {line}</strong></span></div>')
        elif line.strip().startswith('>>>') or line.strip().startswith('```'):
            # Code blocks
            formatted_lines.append(f'<pre style="background:#f8f9fa; padding:8px; border-radius:4px; border:1px solid #dee2e6; margin:4px 0; font-family:monospace; overflow-x:auto;"><code>{line}</code></pre>')
        elif line.strip().startswith('import ') or line.strip().startswith('from ') or 'def ' in line or 'class ' in line:
            # Python code
            formatted_lines.append(f'<div style="background:#f0f8ff; padding:6px; border-left:3px solid #0066cc; margin:2px 0; border-radius:3px;"><code style="color:#0066cc; font-family:monospace;">{line}</code></div>')
        else:
            # Regular text
            if line.strip():  # Only add non-empty lines
                formatted_lines.append(line)
        
        i += 1
    
    return '<br>'.join(formatted_lines)

def get_image_base64(image_path):
    """Convert image to base64 for inline display"""
    import base64
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return ""

def parse_ai_response_for_notebook(content):
    """Parse AI response content into properly formatted notebook cells"""
    lines = content.split('\n')
    cells = []
    current_block = []
    block_type = 'markdown'
    in_code_block = False
    
    i = 0
    while i < len(lines):
        line = lines[i]
        line_stripped = line.strip()
        
        # Handle code block markers
        if line_stripped.startswith('```'):
            # Save current block if any
            if current_block:
                if block_type == 'code' and current_block:
                    cells.append({
                        "cell_type": "code",
                        "execution_count": None,
                        "metadata": {},
                        "outputs": [],
                        "source": current_block
                    })
                elif current_block:
                    cells.append({
                        "cell_type": "markdown",
                        "metadata": {},
                        "source": current_block
                    })
                current_block = []
            
            # Toggle block type
            if not in_code_block:
                block_type = 'code'
                in_code_block = True
            else:
                block_type = 'markdown'
                in_code_block = False
        
        # Handle shell commands (lines starting with $ or common shell commands)
        elif (line_stripped.startswith('$') or 
              any(line_stripped.startswith(cmd) for cmd in ['ls ', 'echo ', 'cat ', 'head ', 'tail ', 'wc ', 'grep ', 'find '])):
            
            # Save current block
            if current_block:
                cells.append({
                    "cell_type": "markdown" if block_type == 'markdown' else "code",
                    "metadata": {},
                    "outputs": [] if block_type == 'code' else None,
                    "execution_count": None if block_type == 'code' else None,
                    "source": current_block
                })
                current_block = []
            
            # Create shell command cell
            shell_cmd = line_stripped.lstrip('$ ')
            cells.append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [f"# Shell command\n!{shell_cmd}"]
            })
            
            # Check if next lines contain output
            i += 1
            output_lines = []
            while i < len(lines) and not lines[i].strip().startswith(('$', '```', 'import ', 'from ', 'def ', 'class ')):
                if lines[i].strip():  # Skip empty lines
                    output_lines.append(lines[i])
                i += 1
                if len(output_lines) > 10:  # Limit output lines
                    break
            
            # If we found output, create an output cell
            if output_lines:
                cells.append({
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [f"**Command Output:**\n```\n" + '\n'.join(output_lines) + "\n```"]
                })
            
            continue  # Skip the i += 1 at the end of the loop
        
        # Handle log output patterns
        elif any(pattern in line for pattern in ['INFO:', 'ERROR:', 'WARNING:', 'SHELL_OUTPUT:', 'PYTHON_OUTPUT:']):
            # Save current block
            if current_block:
                cells.append({
                    "cell_type": "markdown" if block_type == 'markdown' else "code",
                    "metadata": {},
                    "outputs": [] if block_type == 'code' else None,
                    "execution_count": None if block_type == 'code' else None,
                    "source": current_block
                })
                current_block = []
            
            # Extract and format log content
            if 'SHELL_OUTPUT:' in line:
                output = line.split('SHELL_OUTPUT:', 1)[1].strip()
                cells.append({
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [f"**Shell Output:**\n```\n{output}\n```"]
                })
            elif 'PYTHON_OUTPUT:' in line:
                output = line.split('PYTHON_OUTPUT:', 1)[1].strip()
                cells.append({
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [f"**Python Output:**\n```\n{output}\n```"]
                })
            else:
                # Regular log line
                current_block.append(line)
        
        # Regular content
        else:
            if line.strip() or current_block:  # Add line if not empty or if we have content
                current_block.append(line)
        
        i += 1
    
    # Save any remaining content
    if current_block:
        cells.append({
            "cell_type": "markdown" if block_type == 'markdown' else "code",
            "metadata": {},
            "outputs": [] if block_type == 'code' else None,
            "execution_count": None if block_type == 'code' else None,
            "source": current_block
        })
    
    return cells

def create_markdown_pdf_export():
    """Create a markdown document containing chat history and logs that can be converted to PDF"""
    from datetime import datetime
    import markdown
    import base64
    
    # Create markdown content
    md_content = []
    
    # Title and header
    md_content.append("# 🧬 Biomni AI Agent Session Report\n")
    md_content.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    md_content.append(f"**Total Conversations:** {len(st.session_state.chat_history)}\n")
    md_content.append(f"**Total Log Entries:** {len(st.session_state.log_messages)}\n")
    md_content.append(f"**Agent Status:** {'✅ Active' if st.session_state.agent else '❌ Not Initialized'}\n")
    md_content.append(f"**Synapse Connected:** {'✅ Yes' if st.session_state.synapse_connected else '❌ No'}\n")
    md_content.append("\n---\n\n")
    
    # Chat History
    if st.session_state.chat_history:
        md_content.append("## 💬 Chat History\n\n")
        
        for i, message in enumerate(st.session_state.chat_history, 1):
            # Message header
            if message["type"] == "user":
                icon = "👤"
                style_class = "User Query"
            else:
                icon = "🤖"
                style_class = "AI Response"
            
            md_content.append(f"### {icon} {style_class} {i} - {message['timestamp']}\n\n")
            
            # Message content
            content = str(message["content"])
            
            # Clean up content for markdown
            if "```" in content:
                # Content already has code blocks, use as-is
                md_content.append(content)
            else:
                # Regular text content
                md_content.append(content.replace("\n", "\n\n"))
            
            md_content.append("\n\n---\n\n")
    
    # Logs Section
    if st.session_state.log_messages:
        md_content.append("## 📊 Real-time Execution Logs\n\n")
        
        # Log summary
        log_levels = {}
        log_sources = {}
        for log in st.session_state.log_messages:
            level = log.get('level', 'INFO')
            name = log.get('name', 'Unknown')
            log_levels[level] = log_levels.get(level, 0) + 1
            log_sources[name] = log_sources.get(name, 0) + 1
        
        md_content.append(f"**Total Log Entries:** {len(st.session_state.log_messages)}\n")
        md_content.append(f"**Log Levels:** {dict(list(log_levels.items())[:5])}\n")
        md_content.append(f"**Top Sources:** {dict(list(log_sources.items())[:3])}\n\n")
        
        # Recent logs (last 20)
        md_content.append("### Recent Log Entries:\n\n")
        
        for log in st.session_state.log_messages[-20:]:
            timestamp = log.get('timestamp', 'Unknown')
            level = log.get('level', 'INFO')
            message = str(log.get('message', ''))[:200] + ('...' if len(str(log.get('message', ''))) > 200 else '')
            
            # Escape markdown special characters
            message = message.replace("|", "\\|").replace("*", "\\*").replace("_", "\\_")
            
            md_content.append(f"- **[{timestamp}] {level}:** {message}\n")
        
        md_content.append("\n")
    
    # Footer
    md_content.append("\n---\n\n")
    md_content.append(f"*Generated by Biomni Streamlit Interface - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
    
    # Join all content
    markdown_text = "".join(md_content)
    
    # Simple HTML conversion for PDF-like display
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Biomni Session Report</title>
        <style>
            body {{ 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
                line-height: 1.6;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                color: #333;
            }}
            h1 {{ color: #2c5aa0; text-align: center; }}
            h2 {{ color: #0d8043; border-bottom: 2px solid #0d8043; padding-bottom: 5px; }}
            h3 {{ color: #1f77b4; }}
            code {{ 
                background-color: #f4f4f4; 
                padding: 2px 4px; 
                border-radius: 3px;
                font-family: 'Monaco', 'Consolas', monospace;
            }}
            pre {{ 
                background-color: #f8f8f8; 
                border: 1px solid #ddd; 
                border-radius: 5px;
                padding: 10px;
                overflow-x: auto;
            }}
            blockquote {{ 
                border-left: 4px solid #ddd; 
                margin: 0; 
                padding-left: 20px; 
                color: #666;
            }}
            .user-message {{ background-color: #e3f2fd; padding: 10px; border-radius: 5px; }}
            .ai-message {{ background-color: #f1f8e9; padding: 10px; border-radius: 5px; }}
        </style>
    </head>
    <body>
    {markdown_text.replace('👤 User Query', '<div class="user-message">👤 User Query').replace('🤖 AI Response', '</div><div class="ai-message">🤖 AI Response') + '</div>' if '🤖 AI Response' in markdown_text else ''}
    </body>
    </html>
    """
    
    # Convert to bytes for download
    return html_content.encode('utf-8')

def main():
    """Main application function"""
    initialize_session_state()
    #setup_logging()
    
    # Header
    st.markdown('<h1 class="main-header">🧬 Biomni-Synapse Integration</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar setup
    st.sidebar.title("⚙️ Configuration")
    
    # Setup components
    api_ready = setup_api_key()
    agent_ready = initialize_agent() if api_ready else False
    setup_synapse_connection()
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("💬 Chat Interface")
        
        # Status indicators
        st.markdown("**System Status:**")
        status_col1, status_col2, status_col3 = st.columns(3)
        
        with status_col1:
            if st.session_state.api_key_configured:
                st.markdown('<span class="status-success">✅ API Key</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="status-error">❌ API Key</span>', unsafe_allow_html=True)
        
        with status_col2:
            if st.session_state.agent:
                st.markdown('<span class="status-success">✅ Agent</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="status-error">❌ Agent</span>', unsafe_allow_html=True)
        
        with status_col3:
            if st.session_state.synapse_connected:
                st.markdown('<span class="status-success">✅ Synapse</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="status-warning">⚠️ Synapse</span>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Prompt input
        st.subheader("Enter your biomedical research prompt:")
        
        # Predefined example prompts
        example_prompts = {
            "Select an example...": "",
            "CRISPR Screen Planning": """Plan a CRISPR screen to identify genes that regulate T cell exhaustion, measured by the change in T cell receptor (TCR) signaling between acute (interleukin-2 [IL-2] only) and chronic (anti-CD3 and IL-2) stimulation conditions. Generate 32 genes that maximize the perturbation effect.""",
            "Drug Discovery": """Identify potential drug targets for treating Alzheimer's disease by analyzing protein-protein interactions, gene expression data, and known disease pathways. Focus on targets that are druggable and have minimal off-target effects.""",
            "Synapse Data Exploration": """Download entity ID syn52663091 to the current directory. Do not print out any of the login credentials into the logs. Then review the file and generate a boxplot of the top 10 highly expressed genes in the file. Then tell me if I can use this dataset to find drug response for MPNST tumors""",
            "Biomarker Discovery": """Analyze single-cell RNA sequencing data to identify biomarkers that distinguish between cancer stem cells and differentiated cancer cells in glioblastoma. Provide a ranked list of the top 20 biomarkers with biological justification."""
        }
        
        selected_example = st.selectbox("Choose an example prompt:", list(example_prompts.keys()))
        
        prompt = st.text_area(
            "Your prompt:",
            value=example_prompts[selected_example],
            height=150,
            help="Enter your biomedical research question or task"
        )
        
        # Submit button
        submit_col1, submit_col2 = st.columns([1, 4])
        
        with submit_col1:
            # Show different button state based on processing status
            if st.session_state.processing_task:
                st.button("⏳ Processing...", disabled=True)
            else:
                submit_button = st.button("🚀 Submit", disabled=not agent_ready)
        
        with submit_col2:
            clear_button = st.button("🗑️ Clear History")
        
        # Show processing status
        if st.session_state.processing_task:
            st.warning("🤖 **Agent is working on your request...** Please wait and do not submit another query.")
        elif not agent_ready:
            st.warning("⚠️ Please configure API key and initialize the agent first.")
        
        if clear_button:
            st.session_state.chat_history = []
            st.rerun()
        
        # Process submission - only if submit_button exists (when not processing)
        if 'submit_button' in locals() and submit_button and prompt.strip() and st.session_state.agent and not st.session_state.processing_task:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Set processing flag to prevent duplicate execution
            st.session_state.processing_task = True
            
            # Add a unique execution ID to prevent duplicate processing
            st.session_state.execution_counter += 1
            execution_id = f"{timestamp}_{st.session_state.execution_counter}_{hash(prompt)}"
            if hasattr(st.session_state, 'last_execution_id') and st.session_state.last_execution_id == execution_id:
                st.warning("⚠️ This task is already being processed. Please wait...")
                st.session_state.processing_task = False
                return
            st.session_state.last_execution_id = execution_id
            
            # Log execution start with counter
            logging.info(f"Starting task execution #{st.session_state.execution_counter}: {prompt[:100]}...")
            
            # Clear previous logs
            st.session_state.log_messages = []
            
            # Add user message to history
            st.session_state.chat_history.append({
                "type": "user",
                "content": prompt,
                "timestamp": timestamp
            })
            
            try:
                with st.spinner("🤖 Biomni agent is working..."):
                    # Log the start of execution
                    logging.info(f"Executing agent.go() for task #{st.session_state.execution_counter}")
                    
                    # Set up Synapse authentication for the agent if connected
                    synapse_setup_code = ""
                    if st.session_state.synapse_connected and hasattr(st.session_state, 'synapse_profile'):
                        profile_name = st.session_state.synapse_profile
                        logging.info(f"Configuring Synapse authentication with profile: {profile_name}")
                        # Prepare Synapse login code that will be available in the agent's execution context
                        synapse_setup_code = f"""
# Synapse authentication setup
try:
    import synapseclient
    import synapseutils
    syn = synapseclient.login(profile='{profile_name}')
    print(f"✅ Synapse authenticated as: {{syn.username}}")
except Exception as e:
    print(f"⚠️ Synapse authentication warning: {{e}}")
    syn = None
"""
                        # Execute the Synapse setup code in the agent's context
                        try:
                            exec(synapse_setup_code)
                            logging.info("Synapse authentication configured for agent execution")
                        except Exception as syn_error:
                            logging.warning(f"Synapse setup in agent context failed: {syn_error}")
                    elif st.session_state.synapse_connected:
                        logging.info("Synapse connection exists but profile information missing")
                    
                    # Create an enhanced prompt that includes Synapse setup if needed
                    enhanced_prompt = prompt
                    if synapse_setup_code.strip():
                        enhanced_prompt = f"""
{synapse_setup_code.strip()}

# Now execute the main task:
{prompt}
"""
                    
                    # Execute the agent
                    log = st.session_state.agent.go(enhanced_prompt)
                    
                    logging.info(f"Task execution #{st.session_state.execution_counter} completed successfully")
                    
                    # Add AI response to history - preserve original structure
                    st.session_state.chat_history.append({
                        "type": "ai",
                        "content": log,  # Keep original log object, don't convert to string
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                    
                st.success("✅ Task completed successfully!")
                
            except Exception as e:
                error_msg = f"Error executing task: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
                logging.error(f"Task execution failed: {str(e)}")
                st.session_state.chat_history.append({
                    "type": "ai",
                    "content": error_msg,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                st.error("❌ An error occurred while processing your request.")
            
            finally:
                # Reset processing flag
                st.session_state.processing_task = False
                # Clear the execution ID to allow new submissions
                if hasattr(st.session_state, 'last_execution_id'):
                    delattr(st.session_state, 'last_execution_id')
                # Removed automatic rerun to prevent duplicate executions
                # st.rerun() - User can manually refresh if needed
    
    with col2:
        st.subheader("📋 Chat History & Logs")
        
        # Create tabs for chat history and real-time logs
        tab1, tab2 = st.tabs(["💬 Chat History", "📊 Real-time Logs"])
        
        with tab1:
            # Chat history container with enhanced visualization display
            st.markdown("### 💬 Conversation History")
            
            if st.session_state.chat_history:
                # Display messages with enhanced formatting
                for message in reversed(st.session_state.chat_history):
                    display_chat_message(
                        message["type"],
                        message["content"],
                        message["timestamp"]
                    )
                    
                    # Add separator between messages
                    st.markdown("---")
            else:
                st.info("No conversations yet. Submit a prompt to start chatting with the Biomni agent!")
        
        with tab2:
            # Real-time logs container with terminal-style display
            st.markdown("### 📊 Real-time Execution Terminal")
            
            # Log controls row
            control_col1, control_col2, control_col3 = st.columns([2, 1, 1])
            
            with control_col1:
                # Log level filter
                log_levels = st.multiselect(
                    "Filter log levels:",
                    ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                    default=["INFO", "WARNING", "ERROR", "CRITICAL"],
                    key="log_filter"
                )
            
            with control_col2:
                # Auto-refresh toggle
                auto_refresh = st.checkbox(
                    "🔄 Auto-refresh", 
                    value=False,
                    help="Automatically refresh logs when new entries arrive"
                )
            
            with control_col3:
                # Clear logs button
                if st.button("🗑️ Clear Terminal"):
                    st.session_state.log_messages = []
                    st.rerun()
            
            # Process any new log messages from the queue
            new_logs_count = 0
            while not st.session_state.log_queue.empty():
                try:
                    log_entry = st.session_state.log_queue.get_nowait()
                    st.session_state.log_messages.append(log_entry)
                    new_logs_count += 1
                except:
                    break
            
            # Auto-refresh if there are new logs (but only when not processing a task)
            if new_logs_count > 0 and auto_refresh and not st.session_state.processing_task:
                # Only rerun if we're not already in a rerun cycle
                if 'last_rerun_time' not in st.session_state:
                    st.session_state.last_rerun_time = time.time()
                
                # Throttle reruns to prevent infinite loops and conflicts with task execution
                current_time = time.time()
                if current_time - st.session_state.last_rerun_time > 2:  # Increased to 2 seconds
                    st.session_state.last_rerun_time = current_time
                    st.rerun()
            
            # Note: Removed auto-refresh loop to prevent infinite reruns
        
        # Export functionality
        if st.session_state.chat_history:
            st.markdown("---")
            st.subheader("💾 Export Options")
            
            export_col1, export_col2, export_col3 = st.columns(3)
            
            with export_col1:
                if st.button("📄 Export as Text"):
                    export_text = ""
                    for message in st.session_state.chat_history:
                        export_text += f"[{message['timestamp']}] {message['type'].upper()}:\n{message['content']}\n\n"
                    
                    st.download_button(
                        label="Download Chat History",
                        data=export_text,
                        file_name=f"biomni_chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
            
            with export_col2:
                if st.button("📊 Export as JSON"):
                    import json
                    export_json = json.dumps(st.session_state.chat_history, indent=2)
                    
                    st.download_button(
                        label="Download as JSON",
                        data=export_json,
                        file_name=f"biomni_chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
            
            with export_col3:
                if st.button("� Export as PDF"):
                    html_content = create_markdown_pdf_export()
                    
                    st.download_button(
                        label="Download as HTML",
                        data=html_content,
                        file_name=f"biomni_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                        mime="text/html"
                    )

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>🧬 Biomni AI Agent Interface | Built with Streamlit | 
        <a href="https://github.com/snap-stanford/Biomni" target="_blank">GitHub</a></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
