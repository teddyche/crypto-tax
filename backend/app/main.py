from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, Depends, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import func

from db import Base, engine, SessionLocal
from models import TransactionDB   # modèle SQLAlchemy

Base.metadata.create_all(bind=engine)

app = FastAPI(title="CryptoTax API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TransactionOut(BaseModel):
    id: int | None = None
    datetime: datetime
    exchange: str
    pair: str
    side: str
    quantity: float
    price_eur: float
    fees_eur: float
    note: Optional[str] = None

    class Config:
        orm_mode = True  # important pour SQLAlchemy → Pydantic


class SummaryOut(BaseModel):
    total_transactions: int
    total_buy: int
    total_sell: int
    total_deposit: int
    total_withdrawal: int


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/transactions", response_model=List[TransactionOut])
def list_transactions(
        limit: int = Query(500, ge=1, le=5000),
        offset: int = Query(0, ge=0),
        db: Session = Depends(get_db),
):
    rows = (
        db.query(TransactionDB)
        .order_by(TransactionDB.datetime.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    return rows


@app.get("/summary", response_model=SummaryOut)
def get_summary(
        year: int | None = Query(None),
        asset: str | None = Query(None),
        db: Session = Depends(get_db),
):
    query = db.query(TransactionDB)

    if year is not None:
        query = query.filter(func.extract("year", TransactionDB.datetime) == year)

    if asset:
        query = query.filter(TransactionDB.pair.ilike(f"%{asset}%"))

    total = query.count()
    total_buy = query.filter(TransactionDB.side == "BUY").count()
    total_sell = query.filter(TransactionDB.side == "SELL").count()
    total_deposit = query.filter(TransactionDB.side == "DEPOSIT").count()
    total_withdrawal = query.filter(TransactionDB.side == "WITHDRAWAL").count()

    return SummaryOut(
        total_transactions=total,
        total_buy=total_buy,
        total_sell=total_sell,
        total_deposit=total_deposit,
        total_withdrawal=total_withdrawal,
    )