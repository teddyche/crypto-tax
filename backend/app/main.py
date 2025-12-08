from datetime import datetime, date, timedelta
from typing import List, Optional, Literal
import requests
import csv
from io import StringIO

from fastapi import FastAPI, Depends, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.orm import Session

from .db import Base, engine, SessionLocal
from .models import TransactionDB, FXRate, CoinMeta, CoinPrice

# ---------- CryptoCurrencyChart API (historique de prix) ----------

CCC_API_KEY = "912f4971567ad6da574774b52bdd0a5f"
CCC_API_SECRET = "7cb5e29e15306be715942b8675383e74"
CCC_BASE_URL = "https://www.cryptocurrencychart.com/api"

FIAT_SYMS = {"EUR", "USD"}
STABLE_SYMS = {"USDT", "USDC", "BUSD", "TUSD", "FDUSD"}

def is_fiat_or_stable(symbol: str) -> bool:
    s = (symbol or "").upper()
    return s in FIAT_SYMS or s in STABLE_SYMS

def get_usd_eur_rate(db: Session, dt: datetime) -> float:
    """
    Retourne le taux USD->EUR pour la date de la transaction.
    S'il n'y a pas de taux exact ce jour-là, on prend le plus récent avant.
    """
    d = dt.date()

    rate_obj = (
        db.query(FXRate)
        .filter(
            FXRate.base == "USD",
            FXRate.quote == "EUR",
            FXRate.date <= d,
            )
        .order_by(FXRate.date.desc())
        .first()
    )

    if rate_obj is None:
        # fallback au cas où la table est vide
        return 1.0

    return rate_obj.rate

from sqlalchemy import func

def ccc_get(path: str):
    """
    Appel générique à l'API CCC avec auth basic (clé + secret).
    """
    url = f"{CCC_BASE_URL}{path}"
    r = requests.get(url, auth=(CCC_API_KEY, CCC_API_SECRET), timeout=15)
    r.raise_for_status()
    return r.json()


def get_or_create_coin_meta(db: Session, symbol: str) -> CoinMeta | None:
    """
    Récupère (ou crée) le mapping symbol -> id CCC.
    Retourne None si le symbole n'existe pas chez CCC
    (ou si c'est un fiat/stable type USDT, EUR).
    """
    symbol = (symbol or "").upper()

    # Fiat / stables : on ne va pas chez CCC, on gère à part
    if is_fiat_or_stable(symbol):
        return None

    meta = db.query(CoinMeta).filter_by(symbol=symbol).first()
    if meta:
        return meta

    # Appel /coin/list une fois, on cherche par symbol
    data = ccc_get("/coin/list")
    coins = data.get("coins", [])

    match = None
    for c in coins:
        if c.get("symbol", "").upper() == symbol:
            match = c
            break

    if not match:
        # Pas connu chez CCC
        return None

    meta = CoinMeta(
        api_id=int(match["id"]),
        symbol=symbol,
        name=match.get("name"),
        base_currency=(match.get("baseCurrency") or "USD").upper(),
    )
    db.add(meta)
    db.commit()
    db.refresh(meta)
    return meta


def fetch_history_chunk(coin_id: int, start: date, end: date, base_currency: str = "USD"):
    """
    Récupère l'historique journaliers pour [start, end] (max 2 ans).
    """
    path = f"/coin/history/{coin_id}/{start}/{end}/price/{base_currency}"
    return ccc_get(path)


def ensure_coin_history(
        db: Session,
        symbol: str,
        start_date: date,
        end_date: date,
        base_currency: str = "USD",
):
    """
    S'assure qu'on a les prix journaliers pour `symbol` sur [start_date, end_date].
    Ne télécharge que ce qui manque.
    """
    symbol = (symbol or "").upper()

    # Fiat / stables : rien à faire
    if is_fiat_or_stable(symbol):
        return

    meta = get_or_create_coin_meta(db, symbol)
    if not meta:
        # Coin pas dispo chez CCC
        return

    base_currency = "USD"

    # Ce qu'on a déjà
    existing_min, existing_max = db.query(
        func.min(CoinPrice.date),
        func.max(CoinPrice.date),
    ).filter(
        CoinPrice.symbol == symbol,
        CoinPrice.base == base_currency,
        ).one()

    cur = start_date
    if existing_max is not None:
        # On ne redemande que ce qui est après ce qu'on a déjà
        cur = max(cur, existing_max + timedelta(days=1))

    if cur > end_date:
        return

    while cur <= end_date:
        chunk_end = min(
            cur.replace(year=cur.year + 2) - timedelta(days=1),
            end_date,
            )

        payload = fetch_history_chunk(meta.api_id, cur, chunk_end, base_currency)
        for d in payload.get("data", []):
            day = date.fromisoformat(d["date"])
            price = float(d["price"])
            cp = CoinPrice(
                date=day,
                symbol=symbol,
                base=base_currency,
                price=price,
            )
            # merge = insert or update
            db.merge(cp)

        db.commit()
        cur = chunk_end + timedelta(days=1)


def get_coin_price_usd(db: Session, symbol: str, dt: datetime) -> float | None:
    """
    Retourne le prix 1 coin -> USD pour ce jour (ou le plus récent avant).
    """
    symbol = (symbol or "").upper()

    if symbol in {"USD", "USDT", "USDC", "BUSD", "TUSD", "FDUSD"}:
        return 1.0
    if symbol == "EUR":
        # prix “en USD” pour l'EUR : 1 / fx (si on voulait)
        return None

    d = dt.date()
    row = (
        db.query(CoinPrice)
        .filter(
            CoinPrice.symbol == symbol,
            CoinPrice.base == "USD",
            CoinPrice.date <= d,
            )
        .order_by(CoinPrice.date.desc())
        .first()
    )
    return row.price if row else None

def map_operation_to_side(operation: str, quantity: float) -> str:
    op = (operation or "").upper()

    # 1. Commissions / rewards
    if "REFERRER COMMISSION" in op or "COMMISSION HISTORY" in op:
        # On les traite comme des dépôts / rewards entrants
        return "DEPOSIT"

    # 2. Dépôts / retraits classiques
    if op.startswith("DEPOSIT"):
        return "DEPOSIT"
    if op.startswith("WITHDRAW"):
        return "WITHDRAWAL"

    if "SIMPLE EARN FLEXIBLE INTEREST" in op:
        return "INCOME"  # réel revenu

    if "SIMPLE EARN FLEXIBLE SUBSCRIPTION" in op:
        return "SUBSCRIPTION"  # immobilisation

    # 3. Conversions, spend/buy, etc. (selon ce qu’on avait déjà)
    if "BINANCE CONVERT" in op or op == "CONVERT":
        return "CONVERT"

    # 4. Fallback générique
    if quantity > 0:
        return "BUY"
    if quantity < 0:
        return "SELL"
    return "OTHER"

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
    total_convert: int


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

    Quelques règles :
      - INCOME = affiché comme DEPOT (earn, intérêts…)
      - Toute opé qui ressemble à un CONVERT passe en CONVERT,
        même si le side brut est "BUY" ou "SELL".
    """
    raw = (tx.side or "").upper().strip()
    note = (tx.note or "").lower()

    # 1. CONVERT en priorité (même si side brut = BUY/SELL)
    if "convert" in note or raw in {
        "CONVERT",
        "TRANSACTION SPEND",
        "TRANSACTION BUY",
        "TRANSACTION FEE",
    }:
        return "CONVERT"

    # 2. INCOME (earn / intérêts) → affiché comme DEPOT
    if raw == "INCOME":
        return "DEPOSIT"

    # 3. Cas déjà propres
    if raw in {"BUY", "SELL", "DEPOSIT", "WITHDRAWAL"}:
        return raw

    # 4. Buy Crypto With Fiat -> BUY
    if "buy crypto with fiat" in note:
        return "BUY"

    # 5. Withdraw / Deposit détectés dans la note
    if "withdraw" in note:
        return "WITHDRAWAL"
    if "deposit" in note:
        return "DEPOSIT"

    return "OTHER"


# ---------- Routes simples ----------

@app.get("/health")
async def health():
    return {"status": "ok"}


# ---------- Filtres disponibles (années / actifs) ----------

@app.get("/years")
def list_years(
        asset: str | None = Query(None),
        db: Session = Depends(get_db),
):
    query = db.query(func.extract("year", TransactionDB.datetime).label("y"))
    if asset:
        query = query.filter(TransactionDB.pair.ilike(f"%{asset}%"))

    years = (
        query.distinct()
        .order_by("y")
        .all()
    )
    years_int = [int(row.y) for row in years if row.y is not None]
    return {"years": years_int}


@app.get("/assets")
def list_assets(
        year: int | None = Query(None),
        db: Session = Depends(get_db),
):
    query = db.query(TransactionDB.pair)
    if year is not None:
        query = query.filter(func.extract("year", TransactionDB.datetime) == year)

    rows = (
        query.distinct()
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
        types: List[str] | None = Query(None),
        db: Session = Depends(get_db),
):
    """
    Retourne les transactions filtrées, avec `side` NORMALISÉE.
    La pagination est appliquée APRÈS filtrage par type pour ne pas
    perdre les opérations rares (Convert, Earn, etc.).
    """
    query = db.query(TransactionDB)

    if year is not None:
        query = query.filter(func.extract("year", TransactionDB.datetime) == year)

    if asset:
        query = query.filter(TransactionDB.pair.ilike(f"%{asset}%"))

    # 1. On récupère toutes les lignes correspondantes à année/asset
    rows = query.order_by(TransactionDB.datetime.desc()).all()

    # 2. Filtre par type (sur la version normalisée)
    if types:
        allowed = {t.upper() for t in types}
        rows = [tx for tx in rows if normalize_side(tx) in allowed]

    # 3. Pagination manuelle
    start = offset
    end = offset + limit
    page_rows = rows[start:end]

    # 4. Sérialisation
    out: list[TransactionOut] = []
    for tx in page_rows:
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
        types: List[str] | None = Query(None),   # 👈
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

    if types:
        allowed = {t.upper() for t in types}
        rows = [tx for tx in rows if normalize_side(tx) in allowed]

    total = len(rows)
    total_buy = total_sell = total_deposit = total_withdrawal = total_convert = 0

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
        elif s == "CONVERT":
            total_convert += 1

    return SummaryOut(
        total_transactions=total,
        total_buy=total_buy,
        total_sell=total_sell,
        total_deposit=total_deposit,
        total_withdrawal=total_withdrawal,
        total_convert=total_convert,
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

    Avant de parser, on scanne le fichier pour :
      - récupérer la liste des coins utilisés
      - la plage de dates min/max
      - pré-remplir l'historique de prix CCC pour chaque coin
    """

    content = await file.read()
    s = content.decode("utf-8", errors="ignore")

    # Détection séparateur , ou ;
    sample = s[:2048]
    dialect = csv.Sniffer().sniff(sample, delimiters=",;")

    # On bufferise toutes les rows pour pouvoir faire 2 passes
    rows = list(csv.DictReader(StringIO(s), dialect=dialect))

    # --- Scan des assets + min/max dates ---
    assets: set[str] = set()
    min_date: date | None = None
    max_date: date | None = None

    for row in rows:
        utc_time = (row.get("UTC_Time")
                    or row.get("Date(UTC)")
                    or row.get("Time")
                    or "").strip()
        if not utc_time:
            continue

        try:
            dt = datetime.strptime(utc_time, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            try:
                dt = datetime.fromisoformat(utc_time.replace("Z", "+00:00"))
            except Exception:
                continue

        d = dt.date()
        if min_date is None or d < min_date:
            min_date = d
        if max_date is None or d > max_date:
            max_date = d

        coin = (row.get("Coin") or row.get("Asset") or "").strip()
        if coin:
            assets.add(coin.upper())

    # Si on a des dates valides, on précharge les historiques de prix
    if min_date is not None and max_date is not None:
        for sym in assets:
            ensure_coin_history(db, sym, min_date, max_date, base_currency="USD")

    # --- Parsing “réel” des transactions maintenant que les prix sont en base ---

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

    for row in rows:
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

        # --- Cas simples : DEPOSIT / WITHDRAW / EARN (INCOME) ---
        if op_upper == "FIAT DEPOSIT" or op_upper == "DEPOSIT":
            simple_rows.append({
                "datetime": dt,
                "side": "DEPOSIT",
                "pair": coin,
                "quantity": qty,
                "note": f"{account} | {remark}".strip(" |"),
                "price_eur": abs(qty) if coin == "EUR" else 0.0,
                "fees_eur": 0.0,
            })
            continue

        if op_upper == "WITHDRAW":
            simple_rows.append({
                "datetime": dt,
                "side": "WITHDRAWAL",
                "pair": coin,
                "quantity": qty,
                "note": f"{account} | {remark}".strip(" |"),
                "price_eur": abs(qty) if coin == "EUR" else 0.0,
                "fees_eur": 0.0,
            })
            continue

        # Simple Earn : subscription (sortie vers Earn, neutre fiscalement)
        if "SIMPLE EARN FLEXIBLE SUBSCRIPTION" in op_upper:
            simple_rows.append({
                "datetime": dt,
                "side": "SUBSCRIPTION",
                "pair": coin,
                "quantity": qty,
                "note": f"{account} | {remark}".strip(" |"),
            })
            continue

        # EARN / INCOME
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

        # --- Binance Convert (2 lignes : +asset et -asset) ---
        if operation == "Binance Convert":
            bucket = dt.strftime("%Y-%m-%d %H:%M")
            group_key = f"{account}|{bucket}|BINANCE_CONVERT"

            comp = composed_ops[group_key]

            if comp["datetime"] is None or dt > comp["datetime"]:
                comp["datetime"] = dt

            comp["account"] = account
            comp["remark"] = "Binance Convert"
            comp["raw_ops"].append(operation)

            if qty < 0:
                comp["spends"].append((coin, qty))
            elif qty > 0:
                comp["buys"].append((coin, qty))

            continue

        # --- Cas composés : Transaction Spend / Buy / Fee / etc. ---
        group_key = f"{account}|{dt.strftime('%Y-%m-%d %H:%M:%S')}|{remark}"

        comp = composed_ops[group_key]
        comp["datetime"] = dt
        comp["account"] = account
        comp["remark"] = remark
        comp["raw_ops"].append(operation)

        op_u = operation.upper()

        if "TRANSACTION REVENUE" in op_u:
            comp["spends"].append((coin, -abs(qty)))
        elif "SPEND" in op_u or "SOLD" in op_u or op_u == "SELL":
            comp["spends"].append((coin, qty))
        elif "BUY" in op_u:
            comp["buys"].append((coin, qty))
        elif "FEE" in op_u:
            comp["fees"].append((coin, qty))
        else:
            comp["buys"].append((coin, qty))
            comp["raw_ops"].append(f"FALLBACK_OTHER:{operation}")

    inserted = 0

    # --- Insertion des simples ---
    for r in simple_rows:
        qty = r["quantity"]
        pair = r["pair"]
        side = r["side"]

        price_eur = r.get("price_eur", 0.0)
        fees_eur = r.get("fees_eur", 0.0)

        if pair == "EUR":
            price_eur = abs(qty)

        tx = TransactionDB(
            datetime=r["datetime"],
            exchange="Binance",
            pair=pair,
            side=side,
            quantity=qty,
            price_eur=price_eur,
            fees_eur=fees_eur,
            note=r["note"],
        )
        db.add(tx)
        inserted += 1

    # --- Insertion des opérations composées ---
    usd_stables = {"USDT", "USDC", "BUSD", "USD"}

    for key, comp in composed_ops.items():
        dt = comp["datetime"]
        if not dt:
            continue

        account = comp["account"]
        remark = comp["remark"]
        spends = comp["spends"]
        buys = comp["buys"]
        fees = comp["fees"]
        raw_ops = comp["raw_ops"]

        # ---- EUR direct ----
        total_spent_eur = sum(abs(qty) for coin, qty in spends if coin == "EUR")
        total_fees_eur = sum(abs(qty) for coin, qty in fees if coin == "EUR")

        # ---- USD / stablecoins -> EUR via fx table ----
        usd_spent = sum(abs(qty) for coin, qty in spends if coin in usd_stables)
        usd_fees  = sum(abs(qty) for coin, qty in fees   if coin in usd_stables)

        fx = get_usd_eur_rate(db, dt)

        total_spent_eur += usd_spent * fx
        total_fees_eur  += usd_fees * fx

        # -------- from / to assets (agrégés) --------
        from_asset, from_amount = None, 0.0
        to_asset, to_amount = None, 0.0

        for coin, qty in spends:
            if qty < 0:
                if from_asset is None:
                    from_asset = coin
                if coin == from_asset:
                    from_amount += qty

        for coin, qty in buys:
            if qty > 0:
                if to_asset is None:
                    to_asset = coin
                if coin == to_asset:
                    to_amount += qty

        # Cas particulier : "Buy Crypto With Fiat" (aucune ligne EUR dans le CSV)
        if (
                total_spent_eur == 0
                and from_asset is None
                and to_asset is not None
                and any("BUY CRYPTO WITH FIAT" in op.upper() for op in raw_ops)
        ):
            # On reconstruit le prix total en EUR: quantité * prix_jour(BCH_USD) * fx
            price_usd = get_coin_price_usd(db, to_asset, dt)
            if price_usd is not None:
                total_spent_eur = abs(to_amount) * price_usd * fx

        fees_summary = ", ".join(f"{c} {qty}" for c, qty in fees)

        note_parts = []
        if account:
            note_parts.append(f"Account={account}")
        if remark:
            note_parts.append(f"Remark={remark}")
        if from_asset and to_asset:
            note_parts.append(
                f"From {from_amount} {from_asset} -> {to_amount} {to_asset}"
            )
        if fees_summary:
            note_parts.append(f"Fees: {fees_summary}")

        note = " | ".join(note_parts) if note_parts else None

        # -------- Classification + affectation du price_eur --------
        side = "OTHER"
        pair = to_asset or from_asset or "UNKNOWN"
        quantity = to_amount if to_amount != 0 else from_amount
        price_eur = total_spent_eur

        if from_asset in {"EUR"} | usd_stables and to_asset and to_asset not in {"EUR"} | usd_stables:
            side = "BUY"
            pair = to_asset
            quantity = to_amount

        elif to_asset in {"EUR"} | usd_stables and from_asset and from_asset not in {"EUR"} | usd_stables:
            side = "SELL"
            pair = from_asset
            quantity = from_amount

        elif from_asset and to_asset:
            side = "CONVERT"
            pair = to_asset
            quantity = to_amount

        elif from_asset and not to_asset:
            side = "SELL"
            pair = from_asset
            quantity = from_amount

        elif to_asset and not from_asset:
            side = "BUY"
            pair = to_asset
            quantity = to_amount

        tx = TransactionDB(
            datetime=dt,
            exchange="Binance",
            pair=pair,
            side=side,
            quantity=quantity,
            price_eur=price_eur,
            fees_eur=total_fees_eur,
            note=note,
        )
        db.add(tx)
        inserted += 1

    db.commit()
    return {"inserted": inserted}

from xml.etree import ElementTree as ET
from datetime import datetime, date

@app.post("/import/fx-usdeur")
async def import_fx_usdeur(
        file: UploadFile = File(...),
        db: Session = Depends(get_db),
):
    """
    Importe un fichier XML ECB contenant les taux USD/EUR journaliers.
    On ne garde que la série CURRENCY=USD / CURRENCY_DENOM=EUR.
    """

    content = await file.read()
    # Parse XML
    tree = ET.fromstring(content)

    # Namespace ECB (présent dans ton fichier)
    NS = {"exr": "http://www.ecb.europa.eu/vocabulary/stats/exr/1"}

    # On va chercher les Series
    series_list = tree.findall(".//exr:Series", NS)

    if not series_list:
        return {"inserted": 0, "detail": "Aucune série trouvée"}

    inserted = 0

    for series in series_list:
        attrs = series.attrib
        curr = attrs.get("CURRENCY")
        denom = attrs.get("CURRENCY_DENOM")

        # On ne prend que USD/EUR
        if curr != "USD" or denom != "EUR":
            continue

        # Pour éviter les doublons brutaux, on peut supprimer l'ancien jeu
        db.query(FXRate).filter(
            FXRate.base == "USD",
            FXRate.quote == "EUR",
            ).delete()

        # Chaque Obs = 1 jour de taux
        for obs in series.findall("exr:Obs", NS):
            d_str = obs.attrib.get("TIME_PERIOD")
            v_str = obs.attrib.get("OBS_VALUE")

            if not d_str or not v_str:
                continue

            try:
                d = datetime.strptime(d_str, "%Y-%m-%d").date()
                rate = float(v_str)
            except Exception:
                continue

            fx = FXRate(
                date=d,
                base="USD",
                quote="EUR",
                rate=rate,
            )
            db.add(fx)
            inserted += 1

    db.commit()
    return {"inserted": inserted}