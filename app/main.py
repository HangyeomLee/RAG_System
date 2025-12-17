from fastapi import FastAPI
from app.api.question import router as question_router

app = FastAPI(title="RAG System API")

# 라우터 등록
app.include_router(question_router, prefix="/api")

@app.get("/")
def root():
    return {"status": "RAG API is running"}
