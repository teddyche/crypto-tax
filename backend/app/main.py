from __future__ import annotations

from collections import Counter
from datetime import datetime
from io import StringIO
from typing import List, Optional

import csv
from fastapi import Depends, FastAPI, File, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.app.db import Base, SessionLocal, engine
from backend.app.models import TransactionDB

# --------------------------------------------------------------------
# DB init
# --------------------------------------------------------------------

# Crée les tables si elles n'existent pas
Base.metadata.create_all(bind=engine)

# --------------------------------------------------------------------
# FastAPI app
# --------------------------------------------------------------------

app = FastAPI(title="CryptoTax API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # à restreindre plus tard (React dev, etc.)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------
# Pydantic models
# --------------------------------------------------------------------


class Transaction(BaseModel):
    """
    Modèle exposé par l’API (DTO).
    On sépare bien ça du modèle SQLAlchemy TransactionDB.
    """

    id: int | None = None
    datetime: datetime
    exchange: str
    pair: str
    side: str  # Operation Binance brute (BUY, SELL, FEE, Referrer Commission, etc.)
    quantity: float
    price_eur: float
    fees_eur: float
    note: Optional[str] = None

    class Config:
        from_attributes = True  # pour SQLAlchemy 2.x (remplace orm_mode)


class SummaryOut(BaseModel):
    total_transactions: int
    total_buy: int
    total_sell: int
    total_deposit: int
    total_withdrawal: int
    total_fee: int
    total_commission: int
    total_other: int


# --------------------------------------------------------------------
# DB dependency
# --------------------------------------------------------------------


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------


def classify_operation(operation: str | None) -> str:
    """
    Prend le champ TransactionDB.side (Operation Binance brute)
    et renvoie un bucket normalisé.

    Exemples :
    - "BUY"                     -> BUY
    - "SELL"                    -> SELL
    - "WITHDRAW"               -> WITHDRAWAL
    - "Deposit" / "DEPOSIT"    -> DEPOSIT
    - "Referrer Commission"    -> COMMISSION
    - "Commission History"     -> COMMISSION
    - "Fee" / "Trading fee"    -> FEE
    - tout le reste            -> OTHER
    """
    op = (operation or "").upper()

    if op == "BUY":
        return "BUY"
    if op == "SELL":
        return "SELL"
    if "WITHDRAW" in op:
        return "WITHDRAWAL"
    if "DEPOSIT" in op:
        return "DEPOSIT"
    if "COMMISSION" in op:
        # Referrer Commission, Commission History, etc.
        return "COMMISSION"
    if "FEE" in op:
        return "FEE"
    return "OTHER"


# --------------------------------------------------------------------
# Routes simples
# --------------------------------------------------------------------


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/transactions", response_model=List[Transaction])
def list_transactions(
        limit: int = Query(500, ge=1, le=5000),
        offset: int = Query(0, ge=0),
        db: Session = Depends(get_db),
):
    """
    Liste paginée des transactions, ordonnées par date DESC.
    """
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
    """
    Résumé des transactions, avec classification des opérations Binance.
    Les compteurs sont faits à la volée à partir de TransactionDB.side.
    """
    query = db.query(TransactionDB)

    if year is not None:
        query = query.filter(func.extract("year", TransactionDB.datetime) == year)

    if asset:
        # pair contient le coin : "BCH", "USDT", etc.
        query = query.filter(TransactionDB.pair.ilike(f"%{asset}%"))

    counter: Counter[str] = Counter()
    total = 0

    for tx in query:
        bucket = classify_operation(tx.side)
        counter[bucket] += 1
        total += 1

    # Dépôts = vrais DEPOSIT + toutes les commissions (revenus)
    deposit_like = counter["DEPOSIT"] + counter["COMMISSION"]

    return SummaryOut(
        total_transactions=total,
        total_buy=counter["BUY"],
        total_sell=counter["SELL"],
        total_deposit=deposit_like,
        total_withdrawal=counter["WITHDRAWAL"],
        total_fee=counter["FEE"],
        total_commission=counter["COMMISSION"],
        total_other=counter["OTHER"],
    )


@app.post("/transactions", response_model=Transaction)
def create_transaction(tx: Transaction, db: Session = Depends(get_db)):
    """
    Création manuelle d’une transaction (pour tests / futurs imports).
    """
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


# --------------------------------------------------------------------
# Import Binance CSV (Spot & Futures)
# --------------------------------------------------------------------


@app.post("/import/binance")
async def import_binance(
        file: UploadFile = File(...),
        db: Session = Depends(get_db),
):
    """
    Import CSV Binance (export "Account Statement" type).
    Format typique :
    - User_ID
    - UTC_Time
    - Account
    - Operation
    - Coin
    - Change
    - Remark
    """
    content = await file.read()
    s = content.decode("utf-8", errors="ignore")

    # Détection séparateur , ou ;
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
            parsed_date = datetime.strptime(raw_date, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue

        operation = (row.get("Operation") or "UNKNOWN").strip()
        coin = (row.get("Coin") or "UNKNOWN").strip()
        change_str = row.get("Change") or "0"

        # Quantité : on garde la valeur signée
        try:
            quantity = float(str(change_str).replace(",", "."))
        except ValueError:
            quantity = 0.0

        # Pour l’instant on ne connaît pas le prix en EUR à partir de ce CSV
        price_eur = 0.0
        fees_eur = 0.0

        note = row.get("Remark") or None

        tx = TransactionDB(
            datetime=parsed_date,
            exchange="Binance",
            account=row.get("Account") or None,  # si tu as ce champ dans ton modèle
            pair=coin,          # ici on stocke juste le coin (BCH, USDT, etc.)
            side=operation,     # Operation Binance brute
            quantity=quantity,  # signé
            price_eur=price_eur,
            fees_eur=fees_eur,
            note=note,
        )

        db.add(tx)
        inserted += 1

    db.commit()
    return {"inserted": inserted}