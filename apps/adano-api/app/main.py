from contextlib import asynccontextmanager

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from . import classifier
from .db import Base, engine, get_db
from .models import Category, Note


@asynccontextmanager
async def lifespan(_: FastAPI):
    Base.metadata.create_all(engine)
    classifier.warmup()
    yield


app = FastAPI(title="adano", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class CategoryIn(BaseModel):
    name: str = Field(min_length=1, max_length=60)
    description: str = Field(default="", max_length=300)


class CategoryPatch(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=60)
    description: str | None = Field(default=None, max_length=300)


class NoteIn(BaseModel):
    text: str = Field(min_length=1, max_length=500)


class FeedbackIn(BaseModel):
    category_id: int | None  # None = move to Uncategorized


def cat_out(c: Category, count: int | None = None) -> dict:
    out = {"id": c.id, "name": c.name, "description": c.description}
    if count is not None:
        out["count"] = count
    return out


def note_out(n: Note) -> dict:
    return {
        "id": n.id,
        "text": n.text,
        "category_id": n.category_id,
        "predicted_category_id": n.predicted_category_id,
        "confidence": n.confidence,
        "is_confirmed": n.is_confirmed,
        "created_at": n.created_at.isoformat(),
    }


def predict(db: Session, vec_text: str) -> tuple[int | None, float]:
    cats = [cat_out(c) for c in db.scalars(select(Category).order_by(Category.id))]
    examples: dict[int, list[np.ndarray]] = {}
    rows = db.execute(
        select(Note.category_id, Note.embedding).where(Note.is_confirmed, Note.category_id.is_not(None))
    )
    for cid, emb in rows:
        examples.setdefault(cid, []).append(np.frombuffer(emb, dtype=np.float32))
    return classifier.classify(vec_text, cats, examples)


@app.get("/health")
def health(db: Session = Depends(get_db)):
    db.execute(select(1))
    return {"ok": True}


@app.get("/categories")
def list_categories(db: Session = Depends(get_db)):
    counts = dict(db.execute(select(Note.category_id, func.count()).group_by(Note.category_id)).all())
    return [cat_out(c, counts.get(c.id, 0)) for c in db.scalars(select(Category).order_by(Category.id))]


@app.post("/categories", status_code=201)
def create_category(body: CategoryIn, db: Session = Depends(get_db)):
    cat = Category(name=body.name.strip(), description=body.description.strip())
    db.add(cat)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(409, "category exists")
    return cat_out(cat)


@app.patch("/categories/{cat_id}")
def edit_category(cat_id: int, body: CategoryPatch, db: Session = Depends(get_db)):
    cat = db.get(Category, cat_id)
    if not cat:
        raise HTTPException(404, "not found")
    if body.name is not None:
        cat.name = body.name.strip()
    if body.description is not None:
        cat.description = body.description.strip()
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(409, "category exists")
    return cat_out(cat)


@app.delete("/categories/{cat_id}", status_code=204)
def delete_category(cat_id: int, db: Session = Depends(get_db)):
    cat = db.get(Category, cat_id)
    if not cat:
        raise HTTPException(404, "not found")
    for n in db.scalars(select(Note).where(Note.category_id == cat_id)):
        n.is_confirmed = False  # FK ON DELETE SET NULL clears the category itself
    db.flush()
    db.delete(cat)
    db.commit()


@app.post("/notes", status_code=201)
def create_note(body: NoteIn, db: Session = Depends(get_db)):
    text = body.text.strip()
    cat_id, conf = predict(db, text)
    note = Note(
        text=text,
        embedding=classifier.embed(text).tobytes(),
        category_id=cat_id,
        predicted_category_id=cat_id,
        confidence=round(conf, 4),
    )
    db.add(note)
    db.commit()
    db.refresh(note)
    return note_out(note)


@app.get("/notes")
def list_notes(
    category_id: int | None = None,
    uncategorized: bool = False,
    q: str | None = Query(default=None),
    db: Session = Depends(get_db),
):
    stmt = select(Note).order_by(Note.id.desc())
    if uncategorized:
        stmt = stmt.where(Note.category_id.is_(None))
    elif category_id is not None:
        stmt = stmt.where(Note.category_id == category_id)
    if q:
        stmt = stmt.where(Note.text.ilike(f"%{q}%"))
    return [note_out(n) for n in db.scalars(stmt)]


@app.patch("/notes/{note_id}/category")
def confirm_note(note_id: int, body: FeedbackIn, db: Session = Depends(get_db)):
    note = db.get(Note, note_id)
    if not note:
        raise HTTPException(404, "note not found")
    if body.category_id is not None and not db.get(Category, body.category_id):
        raise HTTPException(404, "category not found")
    note.category_id = body.category_id
    note.is_confirmed = body.category_id is not None
    db.commit()
    return note_out(note)


@app.delete("/notes/{note_id}", status_code=204)
def delete_note(note_id: int, db: Session = Depends(get_db)):
    note = db.get(Note, note_id)
    if not note:
        raise HTTPException(404, "note not found")
    db.delete(note)
    db.commit()


@app.post("/notes/reclassify")
def reclassify(db: Session = Depends(get_db)):
    changed = 0
    for n in db.scalars(select(Note).where(Note.is_confirmed.is_(False))):
        cat_id, conf = predict(db, n.text)
        if cat_id != n.category_id:
            changed += 1
        n.category_id = n.predicted_category_id = cat_id
        n.confidence = round(conf, 4)
    db.commit()
    return {"changed": changed}
