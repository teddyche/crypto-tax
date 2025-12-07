from datetime import datetime
from typing import List, Optional

import csv
from io import StringIO

from fastapi import (
    FastAPI,
    Depends,
    UploadFile,
    File,
    Query,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import func

from .db import Base, engine, SessionLocal
from .models import TransactionDB


# -------------------------------------------------------------------
# DB init
# -------------------------------------------------------------------

# Crée les tables si elles n'existent pas
Base.metadata.create_all(bind=engine)


# -------------------------------------------------------------------
# FastAPI app
# -------------------------------------------------------------------

app = FastAPI(title="CryptoTax API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # à restreindre plus tard
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------------------------------------------------
# Dépendance DB
# -------------------------------------------------------------------

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# -------------------------------------------------------------------
# Schémas Pydantic
# -------------------------------------------------------------------

class TransactionOut(BaseModel):
    id: int
    datetime: datetime
    exchange: str
    pair: str
    side: str
    quantity: float
    price_eur: float | None = None
    fees_eur: float | None = None
    note: Optional[str] = None

    class Config:
        from_attributes = True  # SQLAlchemy -> Pydantic


class TransactionCreate(BaseModel):
    datetime: datetime
    exchange: str
    pair: str
    side: str
    quantity: float
    price_eur: float = 0.0
    fees_eur: float = 0.0
    note: Optional[str] = None


class SummaryOut(BaseModel):
    total_transactions: int
    total_buy: int
    total_sell: int
    total_deposit: int
    total_withdrawal: int


# -------------------------------------------------------------------
# Utilitaires
# -------------------------------------------------------------------

def normalize_binance_operation(operation: str) -> str:
    """
    Normalise les opérations Binance en un petit set :
    BUY / SELL / DEPOSIT / WITHDRAWAL / FEE / CONVERT / OTHER
    """
    op = (operation or "").strip().upper()

    # Trades classiques
    if op == "BUY":
        return "BUY"
    if op == "SELL":
        return "SELL"

    # Dépôts / revenus (commissions, rewards, etc.)
    if op in {
        "DEPOSIT",
        "FIAT DEPOSIT",
        "CRYPTO DEPOSIT",
        "REFERRER COMMISSION",
        "COMMISSION HISTORY",
        "DISTRIBUTION",
        "REALIZED PROFIT",
        "REALIZED_PNL",
        "INCOME",
        "INCOME_HISTORY",
        "USD-M FUTURES REFERRER COMMISSION",
    }:
        return "DEPOSIT"

    # Retraits
    if op in {
        "WITHDRAW",
        "WITHDRAWAL",
        "FIAT WITHDRAW",
        "CRYPTO WITHDRAW",
        "TRANSFER_OUT",
    }:
        return "WITHDRAWAL"

    # Frais
    if op in {
        "FEE",
        "TRADING FEE",
        "COMMISSION",
        "COMMISSION FEE",
        "FUNDING FEE",
    }:
        return "FEE"

    # Conversions internes Binance
    if "CONVERT" in op or "SMALL ASSETS EXCHANGE BNB" in op:
        return "CONVERT"

    return "OTHER"


# -------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------

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


@app.post("/transactions", response_model=TransactionOut)
def create_transaction(
        tx: TransactionCreate,
        db: Session = Depends(get_db),
):
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


@app.get("/summary", response_model=SummaryOut)
def get_summary(
        year: int | None = Query(None),
        asset: str | None = Query(None),
        db: Session = Depends(get_db),
):
    """
    Petit résumé agrégé : nombre de BUY / SELL / DEPOSIT / WITHDRAWAL,
    éventuellement filtré par année et/ou asset (pair).
    """
    query = db.query(TransactionDB)

    if year is not None:
        query = query.filter(func.extract("year", TransactionDB.datetime) == year)

    if asset:
        # filtre large : BCH ou BCH/USDT, etc.
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


@app.post("/import/binance")
async def import_binance(
        file: UploadFile = File(...),
        db: Session = Depends(get_db),
):
    """
    Import CSV 'Transaction History' Binance au format :

    "User_ID","UTC_Time","Account","Operation","Coin","Change","Remark"
    """

    content = await file.read()
    s = content.decode("utf-8", errors="ignore")

    # Détection séparateur ("," ou ";")
    sample = s[:1024]
    dialect = csv.Sniffer().sniff(sample, delimiters=",;")
    reader = csv.DictReader(StringIO(s), dialect=dialect)

    inserted = 0

    for row in reader:
        raw_date = row.get("UTC_Time")
        if not raw_date:
            continue

        # Format: 2020-11-02 07:39:45
        try:
            parsed_date = datetime.strptime(raw_date.strip(), "%Y-%m-%d %H:%M:%S")
        except ValueError:
            # Si autre format : on skip pour l'instant
            continue

        account = (row.get("Account") or "").strip()                # Spot / Futures...
        raw_operation = (row.get("Operation") or "UNKNOWN").strip() # ex: Referrer Commission
        coin = (row.get("Coin") or "UNKNOWN").strip()
        change_str = (row.get("Change") or "0").strip()
        remark = (row.get("Remark") or "").strip() or None

        # Quantité signée (dans la devise "Coin")
        try:
            quantity = float(change_str.replace(",", "."))
        except ValueError:
            quantity = 0.0

        # Normalisation de l'opération
        normalized_side = normalize_binance_operation(raw_operation)

        # Pour debug/compliance : on garde tout dans note
        note_parts = [raw_operation]
        if account:
            note_parts.append(f"Account={account}")
        if remark:
            note_parts.append(f"Remark={remark}")
        full_note = " | ".join(note_parts)

        # Pas de prix / fees en EUR dans cet export → 0 pour l’instant
        price_eur = 0.0
        fees_eur = 0.0

        tx = TransactionDB(
            datetime=parsed_date,
            exchange="Binance",
            pair=coin,              # ici Coin = USDT, BCH, etc.
            side=normalized_side,   # 🔥 valeur normalisée
            quantity=quantity,      # quantité signée
            price_eur=price_eur,
            fees_eur=fees_eur,
            note=full_note,
        )

        db.add(tx)
        inserted += 1

    db.commit()
    return {"inserted": inserted}