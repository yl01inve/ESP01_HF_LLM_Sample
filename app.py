import os
import sys
import subprocess
import uvicorn
from typing import List, Dict, Any, Optional

# --- Configuration ---
MODEL_NAME = "Qwen3-0.6B-IQ4_XS.gguf"
MODEL_REPO = "unsloth/Qwen3-0.6B-GGUF"
AMD_SPACE_ID = "amd/gpt-oss-120b-chatbot" # Gradio Space ID for remote inference

# --- 0. Dynamic Module Installation ---
# WARNING: This may fail in many hosted environments due to permission issues.
# A `requirements.txt` is generally recommended for production.

def install_required_modules():
    """
    Installs necessary Python modules at runtime using pip,
    forcing compilation with AVX-512 flags for llama-cpp-python.
    """
    required_packages = [
        "fastapi", "uvicorn", "pydantic", "huggingface-hub",
        "llama-cpp-python", "gradio_client"
    ]
    
    # ----------------------------------------------------
    # **Core Modification: Llama.cpp Compile Options**
    # ----------------------------------------------------
    compile_env = os.environ.copy()
    compile_env["FORCE_CMAKE"] = "1"
    # Note: If your CPU does not support AVX512, this will cause a runtime error (Illegal instruction).
    compile_env["CMAKE_ARGS"] = "-DLLAMA_AVX512=ON -DLLAMA_AVX512_VNNI=ON"
    # ----------------------------------------------------

    print("--- Attempting Dynamic Installation/Upgrade (AVX-512 Compilation) ---")
    
    try:
        subprocess.check_call(
            [
                sys.executable, "-m", "pip", "install", 
                *required_packages, 
                "--upgrade", "--no-cache-dir", "--force-reinstall" # Ensure recompile
            ], 
            env=compile_env
        )
        print("All modules successfully installed/updated. llama-cpp-python compiled with AVX-512.")
    except subprocess.CalledProcessError as e:
        print(f"**FATAL ERROR**: Module installation failed. Error: {e}")
        print("Check if your CPU supports AVX-512 or try removing the CMAKE_ARGS environment variable.")
        sys.exit(1)
    except Exception as e:
        print(f"**FATAL ERROR**: An unknown error occurred. Error: {e}")
        sys.exit(1)

install_required_modules()


# --- 1. Module Imports (Must be after installation) ---
try:
    from pydantic import BaseModel, Field
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse, HTMLResponse
    from fastapi.middleware.cors import CORSMiddleware
    from huggingface_hub import hf_hub_download
    from llama_cpp import Llama, llama_print_system_info
    from gradio_client import Client
except ImportError as e:
    print(f"**FATAL ERROR**: Failed to import modules. Error: {e}")
    sys.exit(1)

# --- 2. Global State ---
LLAMA_INSTANCE: Optional[Llama] = None

def initialize_llm():
    """Downloads the model and initializes the global Llama instance."""
    global LLAMA_INSTANCE
    
    if LLAMA_INSTANCE is not None:
        return

    # Check AVX-512 status
    print("--- Llama.cpp System Info ---")
    print(llama_print_system_info())
    print("-----------------------------")

    print(f"--- 1. Starting model download: {MODEL_NAME} ---")
    try:
        model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_NAME)
    except Exception as e:
        raise RuntimeError(f"Failed to download model: {e}")

    print("--- 2. Initializing Llama.cpp instance ---")
    try:
        # Use half of physical CPU cores for threads, minimum 1
        n_threads = os.cpu_count() // 2 or 1
        LLAMA_INSTANCE = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_batch=512,
            n_threads=n_threads,
            n_gpu_layers=0,
            verbose=False
        )
        print("Llama.cpp model successfully loaded.")
    except Exception as e:
        raise RuntimeError(f"Llama instance initialization failed: {e}")


# --- 3. FastAPI Setup and Middleware ---
app = FastAPI(
    title="LLM Inference API (Llama.cpp)",
    description="API service for direct inference using Llama.cpp."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 4. Pydantic Request Model ---
class InferenceRequestMinimal(BaseModel):
    """Data structure for a minimal inference request, accepting only a question."""
    question: str = Field(..., description="The user's input question or prompt.")


# --- 5. Core Inference Function (Non-Streaming) ---
def get_inference_response(
    messages: List[Dict[str, str]],
    system_message: str,
    max_tokens: int,
    temperature: float = 0.7,
    top_p: float = 0.95,
) -> str:
    """Calls the Llama.cpp instance and returns a single text response."""
    if LLAMA_INSTANCE is None:
        raise HTTPException(status_code=503, detail="LLM Service not initialized.")
    
    # Prepend the system message to the conversation history
    full_messages = [{"role": "system", "content": system_message}]
    full_messages.extend(messages)
    
    try:
        response = LLAMA_INSTANCE.create_chat_completion(
            messages=full_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        
        # Safely extract the content
        content = response.get('choices', [{}])[0].get('message', {}).get('content')
        
        if content:
            return content
        
        return "⚠️ LLM service returned empty content."

    except Exception as e:
        print(f"[Error] LLM Inference failed: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"LLM Server Response Error: {e}"
        )


# --- 6. FastAPI Routes ---

@app.on_event("startup")
async def startup_event():
    """Execute model initialization when FastAPI starts up."""
    try:
        initialize_llm()
    except Exception as e:
        print(f"Application startup failed: {e}")
        # If initialization fails, LLM_INSTANCE is None, and inference will return 503.

@app.get("/", summary="Home/Health Check")
async def root():
    status = "running" if LLAMA_INSTANCE else "starting/failed (LLM unavailable)"
    return HTMLResponse(content=f"<html><body><h1>LLM API Status: {status}</h1></body></html>", status_code=200)


@app.post("/local/qwen-0-6b", summary="Execute Local LLM Inference (Minimal Input)")
async def infer_local_endpoint(request: InferenceRequestMinimal):
    """
    Executes inference using the local Llama.cpp instance.
    Returns a JSON with the 'response' field.
    """
    FIXED_SYSTEM_MESSAGE = "You are a friendly and concise assistant."
    FIXED_MAX_TOKENS = 4096

    try:
        messages = [{"role": "user", "content": request.question}]

        content = get_inference_response(
            messages=messages,
            system_message=FIXED_SYSTEM_MESSAGE,
            max_tokens=FIXED_MAX_TOKENS,
        )
        
        return JSONResponse(content={"response": content})

    except HTTPException:
        raise
    except Exception as e:
        print(f"[Fatal Error] During local API call: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error.")
        

@app.post("/remote/amd", summary="Call External AMD LLM Space via Gradio Client")
async def infer_amd_endpoint(request: InferenceRequestMinimal):
    """
    Uses gradio_client to call the /chat API of the AMD_SPACE_ID.
    Input/output format is consistent with the local endpoint.
    """
    try:
        # Initialize Gradio Client using the global AMD_SPACE_ID
        client = Client(AMD_SPACE_ID)

        # Call the Space API
        result = client.predict(
            message=request.question,
            system_prompt="You are a helpful assistant.",
            temperature=0.7,
            api_name="/chat"
        )
        
        # Process and return result in the required format
        if isinstance(result, str):
            return JSONResponse(content={"response": result})
        else:
            raise ValueError("External API returned unexpected non-string format.")

    except Exception as e:
        print(f"[Fatal Error] Gradio Client API call failed: {e}")
        # Return 503 Service Unavailable for external API errors
        raise HTTPException(
            status_code=503,
            detail=f"External AMD LLM Service Error: {e}"
        )

        
# --- 9. Application Startup ---
if __name__ == "__main__":
    print("FastAPI service is starting...")
    # The 'app:app' structure tells uvicorn to look for the 'app' object 
    # inside the current module (which is also named 'app' when run directly).
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)
