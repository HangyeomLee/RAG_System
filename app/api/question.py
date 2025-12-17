from fastapi import APIRouter
from app.schemas.question import QuestionRequest, QuestionResponse

router = APIRouter()

@router.post("/question", response_model=QuestionResponse)
def ask_question(request: QuestionRequest):
    # 지금은 GPT/FAISS 안 씀
    return QuestionResponse(
        question=request.question,
        message="Question received successfully"
    )
