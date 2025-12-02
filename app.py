import os
import sys
import subprocess
from typing import List, Dict, Any, Optional

# --- 0. 內嵌模組安裝 ---
# 警告: 這在許多託管環境中可能因權限不足而失敗。建議使用 requirements.txt。

def install_required_modules():
    """使用 pip 在運行時安裝所有必要的 Python 模組，並強制啟用 AVX-512 編譯。"""
    required_packages = [
        "fastapi",
        "uvicorn",
        "pydantic",
        "huggingface-hub",
        "llama-cpp-python" 
    ]
    
    # ----------------------------------------------------
    # **核心修改處：設定 Llama.cpp 編譯選項**
    # ----------------------------------------------------
    compile_env = os.environ.copy()
    
    # 1. 強制使用 CMake
    compile_env["FORCE_CMAKE"] = "1"
    
    # 2. 設定 CMake 參數，啟用 AVX512 和 AVX512_VNNI
    # 注意: 如果您的 CPU 不支援 AVX512，這將導致程式運行時錯誤 (Illegal instruction)。
    # 推薦將其設為環境變數，例如 os.environ.get("LLAMA_COMPILER_FLAGS", "-DLLAMA_AVX512=ON -DLLAMA_AVX512_VNNI=ON")
    compile_env["CMAKE_ARGS"] = "-DLLAMA_AVX512=ON -DLLAMA_AVX512_VNNI=ON"
    # ----------------------------------------------------

    print("--- 嘗試動態安裝/升級必要的 Python 模組 (啟用 AVX-512 編譯) ---")
    
    try:
        subprocess.check_call([
            sys.executable, 
            "-m", 
            "pip", 
            "install", 
            *required_packages, 
            "--upgrade",
            "--no-cache-dir", # 確保重新編譯
            "--force-reinstall" # 確保重新編譯
        ], 
        # 將設定好的環境變數傳遞給 subprocess
        env=compile_env)
        
        print("所有模組安裝/更新成功，llama-cpp-python 已使用 AVX-512 編譯。")
    except subprocess.CalledProcessError as e:
        print(f"**致命錯誤**：模組安裝失敗。錯誤訊息: {e}")
        print("請檢查您的 CPU 是否支援 AVX-512，或嘗試移除 CMAKE_ARGS 環境變數。")
        sys.exit(1)
    except Exception as e:
        print(f"**致命錯誤**：發生未知錯誤。錯誤訊息: {e}")
        sys.exit(1)

install_required_modules()


# --- 1. 模組引入 (必須在安裝之後) ---

try:
    # 引入 FastAPI 相關模組
    from pydantic import BaseModel, Field
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse, HTMLResponse
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
    
    # 引入模型下載工具
    from huggingface_hub import hf_hub_download
    
    # 引入 Llama.cpp 模組
    from llama_cpp import Llama, llama_print_system_info # 增加 system info 檢查
except ImportError as e:
    print(f"**致命錯誤**：模組引入失敗。錯誤: {e}")
    sys.exit(1)


# --- 2. 模型設定與初始化 ---

#MODEL_NAME = "Qwen3-0.6B-Q8_0.gguf"
#MODEL_REPO = "Qwen/Qwen3-0.6B-GGUF"
MODEL_NAME = "Qwen3-0.6B-IQ4_XS.gguf"
MODEL_REPO = "unsloth/Qwen3-0.6B-GGUF"
LLAMA_INSTANCE: Optional[Llama] = None # 全域 Llama 實例

def initialize_llm():
    """下載模型並初始化 Llama 實例"""
    global LLAMA_INSTANCE
    
    if LLAMA_INSTANCE is not None:
        return

    # 檢查 AVX-512 是否啟用
    print("--- Llama.cpp System Info ---")
    print(llama_print_system_info())
    print("-----------------------------")


    print(f"--- 1. 開始下載模型 {MODEL_NAME} ---")
    try:
        model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_NAME)
    except Exception as e:
        raise RuntimeError(f"無法下載模型: {e}")

    print("--- 2. 初始化 Llama.cpp 實例 ---")
    try:
        LLAMA_INSTANCE = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_batch=512,
            n_threads=os.cpu_count() // 2 or 1,
            n_gpu_layers=0,
            verbose=False
        )
        print("Llama.cpp 模型加載成功。")
    except Exception as e:
        raise RuntimeError(f"Llama 實例初始化失敗: {e}")


# --- 3. FastAPI 設定與中介層 (Middleware) ---

app = FastAPI(
    title="LLM 推論 API (Llama.cpp)",
    description="直接使用 Llama.cpp 進行推論的 API 服務。"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- 4. Pydantic 請求模型 (僅保留極簡版) ---

class InferenceRequestMinimal(BaseModel):
    """極簡推論請求的資料結構，僅接收問題。"""
    question: str = Field(..., description="使用者輸入的問題或提示。")


# --- 5. 推論核心函式 (非流式) ---

def get_inference_response(
    messages: List[Dict[str, str]],
    system_message: str,
    max_tokens: int,
    temperature: float = 0.7,
    top_p: float = 0.95,
) -> str:
    """呼叫 Llama.cpp 實例並返回單一文字回應。"""

    if LLAMA_INSTANCE is None:
        raise HTTPException(status_code=503, detail="LLM 服務尚未初始化。")
    
    full_messages = [{"role": "system", "content": system_message}]
    full_messages.extend(messages)
    
    try:
        response = LLAMA_INSTANCE.create_chat_completion(
            messages=full_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        
        if response.get('choices') and response['choices'][0].get('message') and response['choices'][0]['message'].get('content'):
            content = response['choices'][0]['message']['content']
            return content
        
        return "⚠️ LLM 服務回傳空內容。"

    except Exception as e:
        print(f"[Error] LLM Inference failed: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"LLM Server Response Error: {e}"
        )


# --- 6. FastAPI 路由: / (健康檢查/首頁) ---

@app.on_event("startup")
async def startup_event():
    """FastAPI 啟動時執行模型初始化"""
    try:
        initialize_llm()
    except Exception as e:
        print(f"應用程式啟動失敗: {e}")
        # 如果初始化失敗，LLM 實例為 None，推論會拋出 503 錯誤

@app.get("/", summary="首頁/健康檢查")
async def root():
    status = "running" if LLAMA_INSTANCE else "starting/failed (LLM unavailable)"
    return HTMLResponse(content=f"<html><body><h1>LLM API Status: {status}</h1></body></html>", status_code=200)


# --- 7. FastAPI 路由: /infer4 (極簡版) ---

@app.post("/infer4", summary="執行 LLM 推論 (v4: 極簡輸入/僅回傳 response 欄位)")
async def infer4_endpoint(request: InferenceRequestMinimal):
    FIXED_SYSTEM_MESSAGE = "You are a friendly and concise assistant."
    FIXED_MAX_TOKENS = 4096

    try:
        messages = [{"role": "user", "content": request.question}]

        content = get_inference_response(
            messages=messages,
            system_message=FIXED_SYSTEM_MESSAGE,
            max_tokens=FIXED_MAX_TOKENS,
        )
        
        return JSONResponse(content={
            "response": content
        })

    except HTTPException as http_ex:
        raise http_ex
    except Exception as e:
        print(f"[Fatal Error] During API call: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal Server Error."
        )
        
        
# --- 8. 啟動應用程式 ---

if __name__ == "__main__":
    print("FastAPI 服務正在啟動...")
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)
