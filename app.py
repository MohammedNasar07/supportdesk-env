import uvicorn
import os
from server.app import app
import gradio as gr
from app_ui import demo

# Combined app
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
