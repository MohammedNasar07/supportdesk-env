from fastapi import FastAPI
import uvicorn

app = FastAPI()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def root():
    return {"message": "Support Desk Running"}


# ✅ REQUIRED BY HUGGING FACE OPENENV
def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)


# ✅ REQUIRED ENTRY GUARD
if __name__ == "__main__":
    main()
