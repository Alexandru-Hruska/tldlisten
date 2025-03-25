from fastapi import FastAPI
from gradio.routes import mount_gradio_app
import modal
import os

# Import the Gradio app
from app_frontend import app as blocks

# Create a Modal image with required dependencies
image = modal.Image.debian_slim().pip_install(
    "gradio<6",  # Gradio version below 6
    "llama-index-core",
    "llama-index-embeddings-openai",
    "llama-index-llms-openai",
    "python-dotenv",  # Environment variables
)

# Handle the podcast transcript file from a different directory
# This will copy the file from ../data/test.txt to /root/data/test.txt in the container
data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
image = image.add_local_dir(
    local_dir=data_dir, 
    remote_dir="/root/data"
)

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
    # Need to update the transcript path in the environment for the app
    os.environ["TRANSCRIPT_PATH"] = "/root/data/raw/test.txt"
    
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
    print(f"{type(blocks)} is ready to go!")
    print(f"Using transcript from: {os.path.join(data_dir, 'test.txt')}")