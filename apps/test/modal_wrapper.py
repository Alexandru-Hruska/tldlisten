from fastapi import FastAPI
from gradio.routes import mount_gradio_app
import modal
import os

# Import the Gradio app
from app import app as blocks

# Create a Modal image with required dependencies
image = modal.Image.debian_slim().pip_install(
    "gradio<6",  # Gradio version below 6
    "llama-index-core",
    "llama-index-embeddings-openai",
    "llama-index-llms-openai",
    "python-dotenv",  # Environment variables
).add_local_file("test.txt", "/root/test.txt") 

# Define the Modal app
app = modal.App("tldlisten", image=image)

@app.function(
    concurrency_limit=1,  # Only one instance for consistent file access
    allow_concurrent_inputs=1000,  # Async handling for up to 1000 concurrent requests
    secrets=[
        modal.Secret.from_name("openai-secret-2"), # change as needed
    ]
)
@modal.asgi_app() # Register this as an ASGI app (compatible with FastAPI)
def serve() -> FastAPI:
    """
    Main server function that:
    - Wraps Gradio inside FastAPI
    - Deploys the API through Modal with a single instance for session consistency
    """
    # Set the environment variable for the transcript path in the container
    os.environ["TRANSCRIPT_PATH"] = "/root/test.txt"
    
    api = FastAPI(docs=True) # Enable Swagger documentation at /docs
    
    # Mount Gradio app at root path
    return mount_gradio_app(
        app=api,
        blocks=blocks,
        path="/"
    )

@app.local_entrypoint()
def main():
    """
    Local development entry point: 
    - Allows running the app locally for testing
    - Prints the type of Gradio app to confirm readiness
    """
    # For local development, use the file in the current directory
    os.environ["TRANSCRIPT_PATH"] = "test.txt"
    print(f"{type(blocks)} is ready to go!")
    print(f"Using transcript from: {os.path.abspath('test.txt')}")