"""知识库和文档 HTTP 接口；路由只做协议转换，业务逻辑放在 service。"""

from typing import List

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.schemas.knowledge import DocumentDetailResponse, DocumentResponse, KnowledgeBaseCreate, KnowledgeBaseResponse
from app.services import knowledge
from app.core.database import User
from app.api.dependencies import get_current_user

router = APIRouter(prefix="/knowledge-bases")


@router.get("", response_model=List[KnowledgeBaseResponse], summary="List knowledge bases")
def get_knowledge_bases(current_user: User = Depends(get_current_user)) -> List[dict]:
    return knowledge.list_knowledge_bases(current_user.id)


@router.post("", response_model=KnowledgeBaseResponse, status_code=status.HTTP_201_CREATED, summary="Create a knowledge base")
def post_knowledge_base(payload: KnowledgeBaseCreate, current_user: User = Depends(get_current_user)) -> dict:
    return knowledge.create_knowledge_base(payload.name.strip(), payload.description.strip(), current_user.id)


@router.get("/{knowledge_base_id}/documents", response_model=List[DocumentResponse], summary="List documents")
def get_documents(knowledge_base_id: str, current_user: User = Depends(get_current_user)) -> List[dict]:
    try:
        knowledge.ensure_knowledge_base_owner(knowledge_base_id, current_user.id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    return knowledge.list_documents(knowledge_base_id)


@router.get("/{knowledge_base_id}/documents/{document_id}", response_model=DocumentDetailResponse, summary="Get parsed document pages")
def get_document(knowledge_base_id: str, document_id: str, current_user: User = Depends(get_current_user)) -> dict:
    try:
        knowledge.ensure_knowledge_base_owner(knowledge_base_id, current_user.id)
        document = knowledge.get_document(document_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    if document["knowledge_base_id"] != knowledge_base_id:
        raise HTTPException(status_code=404, detail="文档不存在")
    return document


@router.post("/{knowledge_base_id}/documents", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED, summary="Upload and parse a document")
async def post_document(knowledge_base_id: str, file: UploadFile = File(...), current_user: User = Depends(get_current_user)) -> dict:
    try:
        knowledge.ensure_knowledge_base_owner(knowledge_base_id, current_user.id)
        return await knowledge.ingest_document(knowledge_base_id, file)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@router.delete("/{knowledge_base_id}/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Delete a document")
def delete_document(knowledge_base_id: str, document_id: str, current_user: User = Depends(get_current_user)) -> None:
    try:
        knowledge.ensure_knowledge_base_owner(knowledge_base_id, current_user.id)
        knowledge.delete_document(knowledge_base_id, document_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
