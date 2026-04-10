import uvicorn
import os

if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    # Import from server.app ensures all routes and Gradio mount are registered
    uvicorn.run("server.app:app", host="0.0.0.0", port=port)
