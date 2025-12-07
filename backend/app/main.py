from datetime import datetime
from typing import List, Optional, Literal

import csv
from io import StringIO

from fastapi import FastAPI, Depends, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.orm import Session

from .db import Base, engine, SessionLocal
from .models import TransactionDB

# Création des tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="CryptoTax API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # à restreindre plus tard
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Pydantic models ----------

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
        from_attributes = True


class TransactionIn(BaseModel):
    datetime: datetime
    exchange: str
    pair: str
    side: str
    quantity: float
    price_eur: float | None = None
    fees_eur: float | None = None
    note: Optional[str] = None


class SummaryOut(BaseModel):
    total_transactions: int
    total_buy: int
    total_sell: int
    total_deposit: int
    total_withdrawal: int


# ---------- DB utils ----------

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ---------- Helpers de normalisation Binance ----------

def normalize_side(tx: TransactionDB) -> str:
    """
    Normalise le field `side` pour l'affichage et les stats.

    On garde uniquement :
      - BUY
      - SELL
      - DEPOSIT
      - WITHDRAWAL
      - CONVERT
      - OTHER
    """
    raw = (tx.side or "").upper().strip()
    note = (tx.note or "").lower()

    # Cas déjà propres
    if raw in {"BUY", "SELL", "DEPOSIT", "WITHDRAWAL"}:
        return raw

    # Buy Crypto With Fiat -> BUY (tu voulais ça)
    if "buy crypto with fiat" in note:
        return "BUY"

    # Withdraw / Deposit détectés dans la note
    if "withdraw" in note:
        return "WITHDRAWAL"
    if "deposit" in note:
        return "DEPOSIT"

    # Convert : plein de variantes Binance
    if "convert" in note or raw in {
        "CONVERT",
        "TRANSACTION SPEND",
        "TRANSACTION BUY",
        "TRANSACTION FEE",
    }:
        return "CONVERT"

    return "OTHER"


# ---------- Routes simples ----------

@app.get("/health")
async def health():
    return {"status": "ok"}


# ---------- Filtres disponibles (années / actifs) ----------

@app.get("/years")
def list_years(db: Session = Depends(get_db)):
    """
    Retourne la liste des années présentes dans les transactions.
    Format attendu par le front : {"years": [2021, 2022, ...]}
    """
    years = (
        db.query(func.extract("year", TransactionDB.datetime).label("y"))
        .distinct()
        .order_by("y")
        .all()
    )
    # years = [(2021.0,), (2022.0,)...] → on cast en int
    years_int = [int(row.y) for row in years if row.y is not None]
    return {"years": years_int}


@app.get("/assets")
def list_assets(db: Session = Depends(get_db)):
    """
    Retourne la liste des actifs (pair/coin) distincts.
    Format attendu : {"assets": ["BCH", "USDT", ...]}
    """
    rows = (
        db.query(TransactionDB.pair)
        .distinct()
        .order_by(TransactionDB.pair.asc())
        .all()
    )
    assets = [row.pair for row in rows if row.pair]
    return {"assets": assets}


# ---------- Transactions & summary ----------

@app.get("/transactions", response_model=List[TransactionOut])
def list_transactions(
        limit: int = Query(100, ge=1, le=5000),
        offset: int = Query(0, ge=0),
        year: int | None = Query(None),
        asset: str | None = Query(None),
        db: Session = Depends(get_db),
):
    """
    Retourne les transactions filtrées, avec `side` NORMALISÉE.
    """
    query = db.query(TransactionDB)

    if year is not None:
        query = query.filter(func.extract("year", TransactionDB.datetime) == year)

    if asset:
        query = query.filter(TransactionDB.pair.ilike(f"%{asset}%"))

    rows = (
        query.order_by(TransactionDB.datetime.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )

    out: list[TransactionOut] = []
    for tx in rows:
        normalized = normalize_side(tx)
        out.append(
            TransactionOut(
                id=tx.id,
                datetime=tx.datetime,
                exchange=tx.exchange,
                pair=tx.pair,
                side=normalized,
                quantity=tx.quantity,
                price_eur=tx.price_eur,
                fees_eur=tx.fees_eur,
                note=tx.note,
            )
        )
    return out


@app.get("/summary", response_model=SummaryOut)
def get_summary(
        year: int | None = Query(None),
        asset: str | None = Query(None),
        db: Session = Depends(get_db),
):
    """
    Summary basé sur les `side` normalisés (BUY/SELL/DEPOSIT/WITHDRAWAL).
    Les CONVERT restent en dehors du comptage buy/sell pour l'instant.
    """
    query = db.query(TransactionDB)

    if year is not None:
        query = query.filter(func.extract("year", TransactionDB.datetime) == year)

    if asset:
        query = query.filter(TransactionDB.pair.ilike(f"%{asset}%"))

    rows = query.all()

    total = len(rows)
    total_buy = total_sell = total_deposit = total_withdrawal = 0

    for tx in rows:
        s = normalize_side(tx)
        if s == "BUY":
            total_buy += 1
        elif s == "SELL":
            total_sell += 1
        elif s == "DEPOSIT":
            total_deposit += 1
        elif s == "WITHDRAWAL":
            total_withdrawal += 1

    return SummaryOut(
        total_transactions=total,
        total_buy=total_buy,
        total_sell=total_sell,
        total_deposit=total_deposit,
        total_withdrawal=total_withdrawal,
    )


# ---------- Création manuelle ----------

@app.post("/transactions", response_model=TransactionOut)
def create_transaction(tx: TransactionIn, db: Session = Depends(get_db)):
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

    normalized = normalize_side(tx_db)
    return TransactionOut(
        id=tx_db.id,
        datetime=tx_db.datetime,
        exchange=tx_db.exchange,
        pair=tx_db.pair,
        side=normalized,
        quantity=tx_db.quantity,
        price_eur=tx_db.price_eur,
        fees_eur=tx_db.fees_eur,
        note=tx_db.note,
    )


# ---------- Import Binance (vite fait) ----------

@app.post("/import/binance")
async def import_binance(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Import à partir du CSV Binance (export historique).
    On stocke les lignes "brutes", mais la normalisation se fait
    via normalize_side() au moment de l'affichage.
    """
    content = await file.read()
    s = content.decode("utf-8", errors="ignore")

    sample = s[:2048]
    dialect = csv.Sniffer().sniff(sample, delimiters=",;")
    reader = csv.reader(StringIO(s), dialect=dialect)

    header = next(reader, None)
    if not header:
        return {"inserted": 0}

    # On suppose un export du type :
    # user_id, time, account, operation, coin, change, remark
    #    0      1      2        3        4     5       6
    inserted = 0

    for row in reader:
        if len(row) < 6:
            continue

        raw_date = row[1]
        try:
            dt = datetime.strptime(raw_date, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue

        account = row[2]
        operation = (row[3] or "").strip()
        coin = (row[4] or "").strip()
        change_str = row[5] or "0"
        remark = row[6] if len(row) > 6 else ""

        try:
            quantity = float(str(change_str).replace(",", "."))
        except ValueError:
            quantity = 0.0

        tx_db = TransactionDB(
            datetime=dt,
            exchange=account or "Binance",
            pair=coin,
            side=operation,   # brut, on normalise plus tard
            quantity=quantity,
            price_eur=0.0,
            fees_eur=0.0,
            note=remark,
        )
        db.add(tx_db)
        inserted += 1

    db.commit()
    return {"inserted": inserted}