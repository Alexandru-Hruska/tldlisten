import gradio as gr
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from pathlib import Path
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI embedding model with API key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
embed_model = OpenAIEmbedding()
Settings.embed_model = embed_model

# Path to the podcast transcript file
TRANSCRIPT_PATH = Path("test.txt")

def load_podcast_index():
    """Load and process podcast transcript data from text file"""
    try:
        # Read the transcript file
        if not TRANSCRIPT_PATH.exists():
            raise FileNotFoundError(f"Transcript file not found at {TRANSCRIPT_PATH}")
            
        with open(TRANSCRIPT_PATH, 'r', encoding='utf-8') as f:
            transcript_text = f.read()
        
        # Create a Document object from the transcript
        doc = Document(
            text=transcript_text,
            metadata={
                "source": str(TRANSCRIPT_PATH),
                "title": "Podcast Transcript"
            }
        )
        
        # Create and return index from document with sentence splitter
        # Using smaller chunk size for more granular context
        parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
        return VectorStoreIndex.from_documents(
            [doc],
            embed_model=embed_model,
            transformations=[parser]
        )
    
    except Exception as e:
        print(f"Error loading podcast transcript: {str(e)}")
        return None

def query_podcast(query, chat_history=None):
    """Query the podcast transcript based on user input"""
    try:
        # Get the pre-loaded index
        index = load_podcast_index()
        if not index:
            return "Error: Could not load podcast transcript data"
        
        # Configure retriever for better context
        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=5,  # Retrieve enough context
        )
        
        # Create query engine with custom retriever
        query_engine = RetrieverQueryEngine.from_args(
            retriever=retriever,
        )
        
        # Query and get response
        response = query_engine.query(query)
        
        # Return formatted response
        return str(response)
    
    except Exception as e:
        return f"Error processing query: {str(e)}"

# Gradio interface setup
with gr.Blocks() as app:
    gr.Markdown("# TL;DListen: Podcast Transcript Chat")
    
    with gr.Row():
        # Chat interface
        chatbot = gr.Chatbot(
            label="Conversation",
            height=400
        )
    
    with gr.Row():
        # Query input
        with gr.Column(scale=4):
            msg = gr.Textbox(
                label="Ask about the podcast",
                placeholder="Ask a question about the podcast content...",
                lines=2
            )
        
        with gr.Column(scale=1):
            # Submit button
            submit_btn = gr.Button("Send")
    
    # Chat history state
    chat_state = gr.State([])
    
    def respond(message, chat_history):
        """Process user message and update chat history"""
        if not message.strip():
            return "", chat_history
            
        # Get response from query engine
        bot_response = query_podcast(message, chat_history)
        
        # Update chat history
        chat_history.append((message, bot_response))
        
        return "", chat_history
    
    # Set up interaction
    submit_btn.click(
        respond,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot]
    )
    
    msg.submit(
        respond,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot]
    )
    
    gr.Markdown("""
    ### Tips:
    - Ask specific questions about the podcast content
    - Try "What were the main topics discussed?"
    - Ask for summaries of specific segments
    - You can request key takeaways or insights
    """)

if __name__ == "__main__":
    # Check for environment variables
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not found in environment variables")
    
    # Verify transcript file exists
    if not TRANSCRIPT_PATH.exists():
        print(f"Warning: Transcript file not found at {TRANSCRIPT_PATH}")
        print(f"Please ensure the file exists at {TRANSCRIPT_PATH.absolute()}")
    
    # Load index on startup for faster initial query
    print("Loading podcast transcript index...")
    index = load_podcast_index()
    if index:
        print("Index loaded successfully!")
    else:
        print("Failed to load transcript index")
    
    # Launch the app
    app.launch()