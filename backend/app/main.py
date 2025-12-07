from datetime import datetime
from typing import List, Optional, Literal

import csv
from io import StringIO
from collections import defaultdict

from fastapi import FastAPI, Depends, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import func

from .db import Base, engine, SessionLocal
from .models import TransactionDB

# ------------------------------------------------------------------------
# DB init
# ------------------------------------------------------------------------

Base.metadata.create_all(bind=engine)

app = FastAPI(title="CryptoTax API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # à restreindre plus tard
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------------
# Pydantic models
# ------------------------------------------------------------------------

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
        from_attributes = True  # FastAPI + SQLAlchemy 2.x


class SummaryOut(BaseModel):
    total_transactions: int
    total_buy: int
    total_sell: int
    total_deposit: int
    total_withdrawal: int


class AssetsOut(BaseModel):
    assets: List[str]


class YearsOut(BaseModel):
    years: List[int]


class TransactionIn(BaseModel):
    """
    Utilisé si un jour tu veux créer une transaction à la main via POST /transactions.
    """
    datetime: datetime
    exchange: str
    pair: str
    side: str
    quantity: float
    price_eur: float = 0.0
    fees_eur: float = 0.0
    note: Optional[str] = None

    class Config:
        orm_mode = True


# ------------------------------------------------------------------------
# Dépendance DB
# ------------------------------------------------------------------------

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ------------------------------------------------------------------------
# Healthcheck
# ------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok"}


# ------------------------------------------------------------------------
# LISTE TRANSACTIONS + FILTRES
# ------------------------------------------------------------------------

@app.get("/transactions", response_model=List[TransactionOut])
def list_transactions(
        limit: int = Query(100, ge=1, le=1000),
        offset: int = Query(0, ge=0),
        year: int | None = Query(None),
        asset: str | None = Query(None),
        db: Session = Depends(get_db),
):
    """
    Retourne les transactions, filtrées par année + actif si fourni.
    """
    q = db.query(TransactionDB)

    if year is not None:
        q = q.filter(func.extract("year", TransactionDB.datetime) == year)

    if asset:
        # On matche dans pair (BCH, BCH/EUR, USDT, etc.)
        like = f"%{asset}%"
        q = q.filter(TransactionDB.pair.ilike(like))

    rows = (
        q.order_by(TransactionDB.datetime.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    return rows


# ------------------------------------------------------------------------
# SUMMARY + FILTRES
# ------------------------------------------------------------------------

@app.get("/summary", response_model=SummaryOut)
def get_summary(
        year: int | None = Query(None),
        asset: str | None = Query(None),
        db: Session = Depends(get_db),
):
    """
    Renvoie les compteurs globaux, filtrables par année + actif.
    """
    base_q = db.query(TransactionDB)

    if year is not None:
        base_q = base_q.filter(func.extract("year", TransactionDB.datetime) == year)

    if asset:
        like = f"%{asset}%"
        base_q = base_q.filter(TransactionDB.pair.ilike(like))

    total = base_q.count()

    def count_side(side: str) -> int:
        return base_q.filter(TransactionDB.side == side).count()

    total_buy = count_side("BUY")
    total_sell = count_side("SELL")
    total_deposit = count_side("DEPOSIT")
    total_withdrawal = count_side("WITHDRAWAL")

    return SummaryOut(
        total_transactions=total,
        total_buy=total_buy,
        total_sell=total_sell,
        total_deposit=total_deposit,
        total_withdrawal=total_withdrawal,
    )


# ------------------------------------------------------------------------
# LISTE DES ACTIFS (pour alimenter le select côté front)
# ------------------------------------------------------------------------

@app.get("/assets", response_model=AssetsOut)
def list_assets(db: Session = Depends(get_db)):
    """
    Retourne la liste des assets trouvés dans `pair`.

    Si pair = "BCH/EUR" => on ajoute BCH et EUR.
    Si pair = "BCH" => on ajoute BCH.
    """
    pairs = db.query(TransactionDB.pair).distinct().all()
    assets_set: set[str] = set()

    for (pair,) in pairs:
        if not pair:
            continue
        if "/" in pair:
            base, quote = pair.split("/", 1)
            assets_set.add(base.strip())
            assets_set.add(quote.strip())
        else:
            assets_set.add(pair.strip())

    assets = sorted(a for a in assets_set if a)
    return AssetsOut(assets=assets)


# ------------------------------------------------------------------------
# LISTE DES ANNÉES
# ------------------------------------------------------------------------

@app.get("/years", response_model=YearsOut)
def list_years(db: Session = Depends(get_db)):
    years_rows = (
        db.query(func.extract("year", TransactionDB.datetime).label("y"))
        .distinct()
        .order_by("y")
        .all()
    )
    years = [int(row[0]) for row in years_rows if row[0] is not None]
    return YearsOut(years=years)


# ------------------------------------------------------------------------
# CRÉATION D’UNE TRANSACTION À LA MAIN (optionnel)
# ------------------------------------------------------------------------

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
    return tx_db


# ------------------------------------------------------------------------
# IMPORT BINANCE – VERSION AGRÉGÉE
# ------------------------------------------------------------------------

@app.post("/import/binance")
async def import_binance(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Import pour l'export Binance "Transactions" avec colonnes :
    User_ID, UTC_Time, Account, Operation, Coin, Change, Remark

    - Les simples Deposit / Withdraw deviennent 1 ligne DEPOSIT/WITHDRAWAL
    - Les blocs Transaction Spend / Buy / Fee (carte, convert, etc.)
      sont agrégés en 1 seule ligne BUY avec :
        * pair = BASE/QUOTE (ex: BCH/EUR)
        * quantity = somme des BUY (BASE)
        * price_eur = somme des Spend (QUOTE)
        * fees_eur = fees BASE converties au même prix
    """
    content = await file.read()
    s = content.decode("utf-8", errors="ignore")
    reader = csv.DictReader(StringIO(s))

    parsed_rows: list[dict] = []

    for raw in reader:
        utc = raw.get("UTC_Time") or raw.get("UTC Time") or raw.get("Time")
        if not utc:
            continue

        op = (raw.get("Operation") or "").strip()
        coin = (raw.get("Coin") or "").strip()
        change_str = (raw.get("Change") or "0").replace(",", ".")
        remark = raw.get("Remark") or ""

        try:
            dt = datetime.strptime(utc, "%Y-%m-%d %H:%M:%S")
        except Exception:
            # format chelou -> on skip
            continue

        try:
            change = float(change_str)
        except ValueError:
            change = 0.0

        parsed_rows.append(
            {
                "dt": dt,
                "operation": op,
                "coin": coin,
                "change": change,
                "remark": remark,
            }
        )

    # Tri chrono
    parsed_rows.sort(key=lambda r: r["dt"])

    inserted = 0
    bundle: list[dict] = []
    current_key: tuple[datetime, bool] | None = None  # (dt, is_transaction_group)

    def flush_bundle(rows: list[dict]):
        nonlocal inserted
        if not rows:
            return

        dt = rows[0]["dt"]
        ops = {r["operation"] for r in rows}

        # ------------------------------------------------------------------
        # Cas 1 : DEPOSIT / WITHDRAW simple
        # ------------------------------------------------------------------
        if (
                len(rows) == 1
                and (("Deposit" in ops) or ("Withdraw" in ops))
        ):
            r = rows[0]
            op = r["operation"]
            coin = r["coin"]
            qty = r["change"]

            if op == "Deposit":
                tx = TransactionDB(
                    datetime=dt,
                    exchange="Binance",
                    pair=coin,
                    side="DEPOSIT",
                    quantity=qty,  # normalement positif
                    price_eur=0.0,
                    fees_eur=0.0,
                    note=r["remark"],
                )
                db.add(tx)
                inserted += 1
                return

            if op == "Withdraw":
                tx = TransactionDB(
                    datetime=dt,
                    exchange="Binance",
                    pair=coin,
                    side="WITHDRAWAL",
                    quantity=qty,  # normalement négatif, Binance inclut le fee
                    price_eur=0.0,
                    fees_eur=0.0,
                    note=r["remark"] or "Withdraw fee is included",
                )
                db.add(tx)
                inserted += 1
                return

        # ------------------------------------------------------------------
        # Cas 2 : bloc Transaction Spend / Buy / Fee (carte, convert…)
        # ------------------------------------------------------------------
        if not any(op.startswith("Transaction") for op in ops):
            # rien de spécial -> on ignore pour l’instant
            return

        buys = [r for r in rows if r["operation"] == "Transaction Buy"]
        spends = [r for r in rows if r["operation"] == "Transaction Spend"]
        fees = [r for r in rows if r["operation"] == "Transaction Fee"]

        if not buys or not spends:
            # bloc incomplet
            return

        base_coin = buys[0]["coin"]   # ex: BCH
        quote_coin = spends[0]["coin"]  # ex: EUR / USDT

        amount_base = sum(r["change"] for r in buys if r["change"] > 0)
        spent_quote = -sum(r["change"] for r in spends if r["change"] < 0)
        fee_base = -sum(r["change"] for r in fees if r["change"] < 0)

        if amount_base <= 0 or spent_quote <= 0:
            return

        price_per_base = spent_quote / amount_base
        fees_eur = fee_base * price_per_base

        tx = TransactionDB(
            datetime=dt,
            exchange="Binance",
            pair=f"{base_coin}/{quote_coin}",
            side="BUY",
            quantity=amount_base,
            price_eur=spent_quote,
            fees_eur=fees_eur,
            note="Aggregated Binance Transaction block",
        )
        db.add(tx)
        inserted += 1

    # Boucle de regroupement
    for r in parsed_rows:
        key = (
            r["dt"],
            r["operation"].startswith("Transaction"),
        )

        if current_key is None:
            current_key = key

        if key != current_key:
            flush_bundle(bundle)
            bundle = []
            current_key = key

        bundle.append(r)

    flush_bundle(bundle)  # dernier paquet

    db.commit()
    return {"inserted": inserted}