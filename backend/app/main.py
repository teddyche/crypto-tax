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

from collections import defaultdict

@app.post("/import/binance")
async def import_binance(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Import avancé du CSV 'Transactions' de Binance.
    On reconstruit des opérations logiques à partir des lignes brutes :
      - Deposit / Withdraw
      - Transaction Spend / Buy / Fee (Convert, achat, etc.)
      - Earn / Staking / Rewards -> INCOME
    """

    content = await file.read()
    s = content.decode("utf-8", errors="ignore")

    # Détection séparateur , ou ;
    sample = s[:2048]
    dialect = csv.Sniffer().sniff(sample, delimiters=",;")
    reader = csv.DictReader(StringIO(s), dialect=dialect)

    # On va stocker les opérations composées par group_key
    composed_ops: dict[str, dict] = defaultdict(lambda: {
        "datetime": None,
        "account": None,
        "remark": None,
        "spends": [],   # [ (coin, amount) ]
        "buys": [],     # [ (coin, amount) ]
        "fees": [],     # [ (coin, amount) ]
        "raw_ops": [],  # debug / traçabilité
    })

    simple_rows: list[dict] = []  # deposits, withdrawals, income simples

    for row in reader:
        # Essaye d'être tolérant sur les noms de colonnes
        utc_time = (row.get("UTC_Time")
                    or row.get("Date(UTC)")
                    or row.get("Time")
                    or "").strip()

        operation = (row.get("Operation") or row.get("Type") or "").strip()
        account = (row.get("Account") or "").strip()
        coin = (row.get("Coin") or row.get("Asset") or "").strip()
        change_str = (row.get("Change") or row.get("Amount") or "0").strip()
        remark = (row.get("Remark") or row.get("Notes") or "").strip()

        if not utc_time or not operation:
            continue

        # Parse date
        try:
            dt = datetime.strptime(utc_time, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            # tente autre format au cas où
            try:
                dt = datetime.fromisoformat(utc_time.replace("Z", "+00:00"))
            except Exception:
                continue

        # Quantité float
        try:
            qty = float(str(change_str).replace(",", "."))
        except ValueError:
            qty = 0.0

        op_upper = operation.upper()
        remark_upper = remark.upper()

        # --- Cas simples d'abord : DEPOSIT / WITHDRAW / EARN (INCOME) ---
        if op_upper == "DEPOSIT":
            simple_rows.append({
                "datetime": dt,
                "side": "DEPOSIT",
                "pair": coin,
                "quantity": qty,
                "note": f"{account} | {remark}".strip(" |"),
            })
            continue

        if op_upper == "WITHDRAW":
            # Binance indique souvent "Withdraw fee is included" dans Remark
            simple_rows.append({
                "datetime": dt,
                "side": "WITHDRAWAL",
                "pair": coin,
                "quantity": qty,  # déjà négatif, frais inclus
                "note": f"{account} | {remark}".strip(" |"),
            })
            continue

        # EARN / INCOME : Binance Earn, Simple Earn, Staking, Launchpool, etc.
        if ("EARN" in remark_upper
            or "SIMPLE EARN" in remark_upper
            or "STAKING" in remark_upper
            or "REWARD" in remark_upper
            or "INTEREST" in remark_upper) and qty > 0:
            simple_rows.append({
                "datetime": dt,
                "side": "INCOME",
                "pair": coin,
                "quantity": qty,
                "note": f"{account} | {remark}".strip(" |"),
            })
            continue

        # --- Nouveau bloc : Binance Convert (2 lignes : +asset et -asset) ---
        if operation == "Binance Convert":
            group_key = f"{account}|{dt.strftime('%Y-%m-%d %H:%M:%S')}|BINANCE_CONVERT"

            comp = composed_ops[group_key]
            comp["datetime"] = dt
            comp["account"] = account
            comp["remark"] = "Binance Convert"

            if qty < 0:
                comp["spends"].append((coin, qty))
            elif qty > 0:
                comp["buys"].append((coin, qty))

            continue

        # --- Cas composés : Transaction Spend / Buy / Fee / Convert / Trade ---
        # On groupe par account + timestamp + remark (clé empirique mais efficace)
        group_key = f"{account}|{dt.strftime('%Y-%m-%d %H:%M:%S')}|{remark}"

        comp = composed_ops[group_key]
        comp["datetime"] = dt
        comp["account"] = account
        comp["remark"] = remark
        comp["raw_ops"].append(operation)

        op_u = operation.upper()

        if "SPEND" in op_u or op_u == "SELL":
            comp["spends"].append((coin, qty))
        elif "BUY" in op_u:
            comp["buys"].append((coin, qty))
        elif "FEE" in op_u:
            comp["fees"].append((coin, qty))
        else:
            # Inclassable -> on le gardera en OTHER plus bas
            comp["buys"].append((coin, qty))  # fallback
            comp["raw_ops"].append(f"FALLBACK_OTHER:{operation}")

    inserted = 0

    # On insère les simples d'abord
    for r in simple_rows:
        tx = TransactionDB(
            datetime=r["datetime"],
            exchange="Binance",
            pair=r["pair"],
            side=r["side"],
            quantity=r["quantity"],
            price_eur=0.0,
            fees_eur=0.0,
            note=r["note"],
        )
        db.add(tx)
        inserted += 1

    # Puis les opérations composées
    for key, comp in composed_ops.items():
        dt = comp["datetime"]
        account = comp["account"]
        remark = comp["remark"]
        spends = comp["spends"]
        buys = comp["buys"]
        fees = comp["fees"]

        if not dt:
            continue

        # Détermine actif "from" et "to"
        # On prend le premier spend négatif comme from,
        # et le plus gros buy positif comme to.
        from_asset, from_amount = None, 0.0
        to_asset, to_amount = None, 0.0

        for coin, qty in spends:
            if qty < 0 and from_asset is None:
                from_asset, from_amount = coin, qty

        max_buy = 0.0
        for coin, qty in buys:
            if qty > 0 and abs(qty) > max_buy:
                max_buy = abs(qty)
                to_asset, to_amount = coin, qty

        # Fees agrégés en libellé
        fees_summary = ", ".join(
            f"{c} {qty}" for c, qty in fees
        )

        note_parts = []
        if account:
            note_parts.append(f"Account={account}")
        if remark:
            note_parts.append(f"Remark={remark}")
        if from_asset and to_asset:
            note_parts.append(f"From {from_amount} {from_asset} -> {to_amount} {to_asset}")
        if fees_summary:
            note_parts.append(f"Fees: {fees_summary}")

        note = " | ".join(note_parts) if note_parts else None

        # Classification finale
        side = "OTHER"
        pair = to_asset or from_asset or "UNKNOWN"
        quantity = to_amount if to_amount != 0 else from_amount

        if from_asset and to_asset:
            # Conversion d'un asset en un autre
            side = "CONVERT"
        elif from_asset and not to_asset:
            # Vente sans contrepartie crypto détectée (ou export tronqué)
            side = "SELL"
        elif to_asset and not from_asset:
            # Achat direct depuis fiat (ou export partiel)
            side = "BUY"

        tx = TransactionDB(
            datetime=dt,
            exchange="Binance",
            pair=pair,
            side=side,
            quantity=quantity,
            price_eur=0.0,
            fees_eur=0.0,
            note=note,
        )
        db.add(tx)
        inserted += 1

    db.commit()
    return {"inserted": inserted}