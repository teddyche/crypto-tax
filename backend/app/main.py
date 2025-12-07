from datetime import datetime
from typing import List, Literal, Optional

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session

from db import Base, engine, SessionLocal
from models import TransactionDB

from fastapi import UploadFile, File
import csv
from io import StringIO
from fastapi import Query
from sqlalchemy import func, select
from db import SessionLocal
from models import Transaction  # adapte au nom réel

# Crée les tables si elles n'existent pas
Base.metadata.create_all(bind=engine)

app = FastAPI(title="CryptoTax API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # à restreindre plus tard
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from pydantic import BaseModel
from typing import Literal

class TransactionOut(BaseModel):
    id: int
    datetime: datetime
    exchange: str
    pair: str
    side: str
    quantity: float
    price_eur: float | None = None
    fees_eur: float | None = None
    note: str | None = None

    class Config:
        from_attributes = True  # si SQLAlchemy 2.x

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

class Transaction(BaseModel):
    id: int | None = None
    datetime: datetime
    exchange: str
    pair: str
    side: str  # on accepte ce que Binance envoie, on normalisera plus tard
    quantity: float
    price_eur: float
    fees_eur: float
    note: Optional[str] = None

    class Config:
        orm_mode = True

@app.get("/health")
async def health():
    return {"status": "ok"}

from fastapi import Query

@app.get("/transactions", response_model=List[Transaction])
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
):
    db = SessionLocal()
    try:
        query = db.query(Transaction)

        if year is not None:
            query = query.filter(func.extract("year", Transaction.datetime) == year)

        if asset:
            # supposition: pair contient l'asset, ex "BCH", "BCH/USDT", etc
            query = query.filter(Transaction.pair.ilike(f"%{asset}%"))

        total = query.count()

        total_buy = query.filter(Transaction.side == "BUY").count()
        total_sell = query.filter(Transaction.side == "SELL").count()
        total_deposit = query.filter(Transaction.side == "DEPOSIT").count()
        total_withdrawal = query.filter(Transaction.side == "WITHDRAWAL").count()

        return SummaryOut(
            total_transactions=total,
            total_buy=total_buy,
            total_sell=total_sell,
            total_deposit=total_deposit,
            total_withdrawal=total_withdrawal,
        )
    finally:
        db.close()

@app.post("/transactions", response_model=Transaction)
def create_transaction(tx: Transaction, db: Session = Depends(get_db)):
    tx_db = TransactionDB(
        datetime=tx.datetime,
        exchange=tx.exchange,
        pair=tx.pair,
        side=tx.side,
        quantity=tx.quantity,
        price_eur=tx.price_eur,
        fees_eur=tx.fees_eur,
        note=tx.note,
    )
    db.add(tx_db)
    db.commit()
    db.refresh(tx_db)
    return tx_db

@app.post("/import/binance")
async def import_binance(file: UploadFile = File(...), db: Session = Depends(get_db)):
    content = await file.read()
    s = content.decode("utf-8", errors="ignore")

    # Détection séparateur , ou ;
    sample = s[:1024]
    dialect = csv.Sniffer().sniff(sample, delimiters=",;")
    reader = csv.DictReader(StringIO(s), dialect=dialect)

    inserted = 0

    for row in reader:
        # Ton CSV : UTC_Time, Operation, Coin, Change, Remark
        raw_date = row.get("UTC_Time")
        if not raw_date:
            continue

        # Format: 2020-11-02 07:39:45
        try:
            parsed_date = datetime.strptime(raw_date, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue

        operation = (row.get("Operation") or "UNKNOWN").upper()
        coin = row.get("Coin") or "UNKNOWN"
        change_str = row.get("Change") or "0"

        # Quantité : on garde la valeur signée
        try:
            quantity = float(str(change_str).replace(",", "."))
        except ValueError:
            quantity = 0.0

        # Pas de prix dans ce CSV → on laisse 0 pour l'instant
        price_eur = 0.0
        fees_eur = 0.0

        # On peut interpréter FEE comme un retrait de frais plus tard
        note = row.get("Remark") or None

        tx = TransactionDB(
            datetime=parsed_date,
            exchange="Binance",
            pair=coin,          # pour ce type de fichier : on met le coin
            side=operation,     # ex: BUY / FEE / SELL / etc.
            quantity=quantity,  # signé
            price_eur=price_eur,
            fees_eur=fees_eur,
            note=note,
        )

        db.add(tx)
        inserted += 1

    db.commit()
    return {"inserted": inserted}
