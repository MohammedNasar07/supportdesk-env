import uvicorn
import os

if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    # We run the FastAPI app which has Gradio mounted inside it
    uvicorn.run("server.app:app", host="0.0.0.0", port=port)
