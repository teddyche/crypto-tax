from datetime import datetime, date, timedelta
from typing import List, Optional
import csv
from io import StringIO
import re
from collections import defaultdict

import requests
from fastapi import FastAPI, Depends, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.orm import Session

from .db import Base, engine, SessionLocal
from .models import TransactionDB, FXRate, CoinMeta, CoinPrice
from xml.etree import ElementTree as ET

# ===================== Constantes / regex =====================

DIRECTION_RE = re.compile(
    r"From\s+([-\d\.]+)\s+([A-Z0-9]+)\s+->\s+([-\d\.]+)\s+([A-Z0-9]+)"
)

CCC_API_KEY = "912f4971567ad6da574774b52bdd0a5f"
CCC_API_SECRET = "7cb5e29e15306be715942b8675383e74"
CCC_BASE_URL = "https://www.cryptocurrencychart.com/api"

# ⚠️ Pour la LOI : seuls les FIAT sont "cash" => fait générateur
FIAT_SYMS = {"EUR", "USD"}          # monnaies ayant cours légal
STABLE_SYMS = {"USDT", "USDC", "BUSD", "TUSD", "FDUSD"}

TYPE_MAP = {
    "ACHAT": "BUY",
    "VENTE": "SELL",
    "DEPOT": "DEPOSIT",
    "DÉPÔT": "DEPOSIT",
    "RETRAIT": "WITHDRAWAL",
    "CONVERT": "CONVERT",

    # Au cas où le front envoie déjà les clés internes :
    "BUY": "BUY",
    "SELL": "SELL",
    "DEPOSIT": "DEPOSIT",
    "WITHDRAWAL": "WITHDRAWAL",
}

# ============ Cache simple des événements fiscaux ============

_tax_cache: dict[str, object] = {
    "events": None,     # dict[int, dict]
    "tx_count": 0,      # nombre de lignes TransactionDB
}

def is_fiat_or_stable(symbol: str) -> bool:
    s = (symbol or "").upper()
    return s in FIAT_SYMS or s in STABLE_SYMS

def is_fiat(symbol: str) -> bool:
    return (symbol or "").upper() in FIAT_SYMS

def is_stable(symbol: str) -> bool:
    return (symbol or "").upper() in STABLE_SYMS

# ===================== Helpers FX / prix =====================

def is_fiat_or_stable(symbol: str) -> bool:
    s = (symbol or "").upper()
    return s in FIAT_SYMS or s in STABLE_SYMS


def get_usd_eur_rate(db: Session, dt: datetime) -> float:
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
    return rate_obj.rate if rate_obj else 1.0


def ccc_get(path: str):
    url = f"{CCC_BASE_URL}{path}"
    r = requests.get(url, auth=(CCC_API_KEY, CCC_API_SECRET), timeout=15)
    r.raise_for_status()
    return r.json()


def get_or_create_coin_meta(db: Session, symbol: str) -> CoinMeta | None:
    symbol = (symbol or "").upper()
    if is_fiat_or_stable(symbol):
        return None

    meta = db.query(CoinMeta).filter_by(symbol=symbol).first()
    if meta:
        return meta

    data = ccc_get("/coin/list")
    coins = data.get("coins", [])

    match = None
    for c in coins:
        if c.get("symbol", "").upper() == symbol:
            match = c
            break

    if not match:
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
    path = f"/coin/history/{coin_id}/{start}/{end}/price/{base_currency}"
    return ccc_get(path)


def ensure_coin_history(
        db: Session,
        symbol: str,
        start_date: date,
        end_date: date,
        base_currency: str = "USD",
):
    symbol = (symbol or "").upper()
    if is_fiat_or_stable(symbol):
        return

    meta = get_or_create_coin_meta(db, symbol)
    if not meta:
        return

    base_currency = "USD"

    existing_min, existing_max = db.query(
        func.min(CoinPrice.date),
        func.max(CoinPrice.date),
    ).filter(
        CoinPrice.symbol == symbol,
        CoinPrice.base == base_currency,
        ).one()

    cur = start_date
    if existing_max is not None:
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
            db.merge(cp)
        db.commit()
        cur = chunk_end + timedelta(days=1)


def get_coin_price_usd(db: Session, symbol: str, dt: datetime) -> float | None:
    symbol = (symbol or "").upper()

    if symbol in STABLE_SYMS or symbol == "USD":
        return 1.0
    if symbol == "EUR":
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

# ===================== DB / FastAPI setup =====================

Base.metadata.create_all(bind=engine)

app = FastAPI(title="CryptoTax API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ===================== Pydantic models =====================

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

    taxable: bool
    direction: str | None = None

    pv_eur: float | None = None
    cum_pv_year_eur: float | None = None
    estimated_tax_eur: float | None = None

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


class TaxEventOut(BaseModel):
    id: int
    datetime: datetime
    pair: str
    side: str
    proceeds_eur: float
    pv_eur: float

    class Config:
        from_attributes = True


class TaxYearOut(BaseModel):
    year: int
    total_pv_eur: float
    flat_tax_30: float
    events: List[TaxEventOut]


class PositionOut(BaseModel):
    symbol: str
    quantity: float
    value_eur: float


class DailyPortfolioOut(BaseModel):
    date: date
    total_value_eur: float
    daily_pnl_eur: float
    positions: List[PositionOut]

# ===================== Portfolio / normalisation =====================

def normalize_side(tx: TransactionDB) -> str:
    raw = (tx.side or "").upper().strip()
    note = (tx.note or "").lower()

    if "convert" in note or raw in {
        "CONVERT",
        "TRANSACTION SPEND",
        "TRANSACTION BUY",
        "TRANSACTION FEE",
    }:
        return "CONVERT"

    if raw == "INCOME":
        return "DEPOSIT"

    if raw in {"BUY", "SELL", "DEPOSIT", "WITHDRAWAL"}:
        return raw

    if "buy crypto with fiat" in note:
        return "BUY"

    if "withdraw" in note:
        return "WITHDRAWAL"
    if "deposit" in note:
        return "DEPOSIT"

    if raw in {"SUBSCRIPTION", "EARN_RETURN"}:
        return "HIDDEN"

    return "OTHER"


def compute_daily_portfolio(db: Session, year: int | None = None) -> list[DailyPortfolioOut]:
    query = db.query(TransactionDB)
    if year is not None:
        query = query.filter(func.extract("year", TransactionDB.datetime) == year)

    txs: list[TransactionDB] = query.order_by(TransactionDB.datetime.asc()).all()
    if not txs:
        return []

    holdings: dict[str, float] = defaultdict(float)
    by_day: dict[date, list[TransactionDB]] = defaultdict(list)

    for tx in txs:
        d = tx.datetime.date()
        by_day[d].append(tx)

    all_days = sorted(by_day.keys())
    results: list[DailyPortfolioOut] = []
    prev_total_value = 0.0

    for d in all_days:
        day_txs = by_day[d]
        net_flow_eur = 0.0

        for tx in day_txs:
            side = normalize_side(tx)
            pair = (tx.pair or "").upper()
            qty = tx.quantity or 0.0

            if pair == "EUR" and side in {"DEPOSIT", "WITHDRAWAL"}:
                if side == "DEPOSIT":
                    net_flow_eur += abs(tx.price_eur or 0.0)
                else:
                    net_flow_eur -= abs(tx.price_eur or 0.0)

            if pair == "EUR" or is_fiat_or_stable(pair):
                continue

            if side in {"BUY", "DEPOSIT", "INCOME"}:
                holdings[pair] += abs(qty)
            elif side in {"SELL", "WITHDRAWAL"}:
                holdings[pair] -= abs(qty)
            elif side == "CONVERT":
                note = tx.note or ""
                m = re.search(
                    r"From\s+([\-0-9\.]+)\s+([A-Z0-9]+)\s*->\s*([\-0-9\.]+)\s+([A-Z0-9]+)",
                    note,
                )
                if m:
                    from_amount = float(m.group(1))
                    from_sym = m.group(2).upper()
                    to_amount = float(m.group(3))
                    to_sym = m.group(4).upper()

                    if not is_fiat_or_stable(from_sym) and from_sym != "EUR":
                        holdings[from_sym] -= abs(from_amount)
                    if not is_fiat_or_stable(to_sym) and to_sym != "EUR":
                        holdings[to_sym] += abs(to_amount)
                else:
                    holdings[pair] += abs(qty)

        day_total_value = 0.0
        positions_out: list[PositionOut] = []
        day_dt = datetime.combine(d, datetime.min.time())

        for sym, q in list(holdings.items()):
            if abs(q) < 1e-12:
                holdings.pop(sym, None)
                continue

            if is_fiat_or_stable(sym) or sym == "EUR":
                continue

            price_usd = get_coin_price_usd(db, sym, day_dt)
            if price_usd is None:
                continue

            fx = get_usd_eur_rate(db, day_dt)
            value_eur = q * price_usd * fx
            day_total_value += value_eur

            positions_out.append(
                PositionOut(
                    symbol=sym,
                    quantity=q,
                    value_eur=value_eur,
                )
            )

        positions_out.sort(key=lambda p: -abs(p.value_eur))
        daily_pnl = day_total_value - (prev_total_value + net_flow_eur)
        prev_total_value = day_total_value

        results.append(
            DailyPortfolioOut(
                date=d,
                total_value_eur=day_total_value,
                daily_pnl_eur=daily_pnl,
                positions=positions_out,
            )
        )

    return results

# ===================== Imposition =====================

def extract_from_to(tx: TransactionDB) -> tuple[str | None, str | None]:
    m = DIRECTION_RE.search(tx.note or "")
    if not m:
        return None, None
    from_asset = m.group(2)
    to_asset = m.group(4)
    return from_asset, to_asset

def is_taxable(tx: TransactionDB) -> bool:
    """
    FR - art. 150 VH bis :
    Fait générateur = cession d'actifs numériques contre une monnaie FIAT
    (ou achat de biens/services, qu'on ne sait pas détecter facilement ici).

    Donc :
      - on ne considère que les SELL
      - on est imposable si on termine dans du FIAT (EUR, USD, ...)
      - crypto -> stable (USDT, USDC...) = NON imposable.
    """
    s = normalize_side(tx)
    if s != "SELL":
        return False

    from_asset, to_asset = extract_from_to(tx)

    # Cas normal : note "From X COIN -> Y EUR"
    if from_asset or to_asset:
        return is_fiat(to_asset)

    # Fallback : pas de note, mais pair = fiat (rare)
    pair = (tx.pair or "").upper()
    if is_fiat(pair):
        return True

    return False

def compute_portfolio_value_eur(
        db: Session,
        holdings_qty: dict[str, float],
        dt: datetime,
        price_cache: dict[tuple[str, date], float],
        fx_cache: dict[date, float],
) -> float:
    """
    Valeur globale du portefeuille (en EUR) à la date dt.
    Utilise des caches `price_cache` et `fx_cache` pour éviter les requêtes SQL
    répétées.
    """
    total = 0.0
    d = dt.date()

    # cache FX
    if d in fx_cache:
        fx = fx_cache[d]
    else:
        fx = get_usd_eur_rate(db, dt)
        fx_cache[d] = fx

    for sym, qty in holdings_qty.items():
        if abs(qty) < 1e-12:
            continue

        s = sym.upper()

        # FIAT hors portefeuille d'actifs numériques
        if is_fiat(s):
            continue

        # prix en USD
        if is_stable(s):
            price_usd = 1.0
        else:
            key = (s, d)
            if key in price_cache:
                price_usd = price_cache[key]
            else:
                price_usd = get_coin_price_usd(db, s, dt)
                price_cache[key] = price_usd

        if price_usd is None:
            continue

        total += qty * price_usd * fx

    return total

from sqlalchemy import func

def compute_tax_events_all_years(db: Session) -> dict[int, dict]:
    """
    Wrapper avec cache en mémoire.
    On ne recalcule la fiscalité que si le nombre de transactions a changé
    (i.e. nouvel import / purge).
    """
    global _tax_cache

    tx_count = db.query(func.count(TransactionDB.id)).scalar() or 0

    if _tax_cache["events"] is None or _tax_cache["tx_count"] != tx_count:
        events = _compute_tax_events_all_years_internal(db)
        _tax_cache["events"] = events
        _tax_cache["tx_count"] = tx_count

    return _tax_cache["events"]  # dict {tx_id: {...}}

def _compute_tax_events_all_years_internal(db: Session) -> dict[int, dict]:
    """
    Implémentation (approx) de l'article 150 VH bis :

    Pour chaque cession imposable (crypto -> FIAT) :

        PV = C - ( A * C / V )

      - C : prix de cession en EUR (montant FIAT reçu)
      - V : valeur globale du portefeuille d'actifs numériques
            juste AVANT la cession (en EUR)
      - A : prix total d'acquisition net du portefeuille
            juste AVANT la cession (en EUR)

    Après chaque cession :
      - on retire du pool une fraction du coût : A <- A - (A * C / V)
      - le portefeuille en coins est mis à jour (on enlève la quantité vendue).
    """
    price_cache: dict[tuple[str, date], float] = {}
    fx_cache: dict[date, float] = {}
    txs = (
        db.query(TransactionDB)
        .order_by(TransactionDB.datetime.asc())
        .all()
    )

    # Quantités (crypto + stables) détenues
    holdings_qty: dict[str, float] = defaultdict(float)

    # Prix total d'acquisition global (A) en EUR
    acquisition_cost_eur: float = 0.0

    cum_pv_by_year: dict[int, float] = defaultdict(float)
    events_by_id: dict[int, dict] = {}

    for tx in txs:
        side_norm = normalize_side(tx)
        asset = (tx.pair or "").upper()
        if not asset:
            continue

        qty = tx.quantity or 0.0
        trade_value_eur = tx.price_eur or 0.0
        year = tx.datetime.year

        from_asset, to_asset = extract_from_to(tx)
        from_asset = (from_asset or "").upper()
        to_asset = (to_asset or "").upper()

        # ---------- 1) Cas NON imposables : mise à jour du pool ----------
        # 1.a) Achats de crypto avec FIAT => augmentent A
        if side_norm == "BUY":
            # Achat via "From XXX FIAT -> YYY COIN"
            if is_fiat(from_asset) and not is_fiat(asset):
                # pool de coins
                holdings_qty[asset] += abs(qty)
                # coût d'acquisition : montant FIAT dépensé
                acquisition_cost_eur += abs(trade_value_eur)
                continue

            # Cas "Buy Crypto With Fiat" sans from_asset explicite
            note_lower = (tx.note or "").lower()
            if "buy crypto with fiat" in note_lower and not is_fiat(asset):
                holdings_qty[asset] += abs(qty)
                acquisition_cost_eur += abs(trade_value_eur)
                continue

            # Achat crypto-crypto (ex: USDT -> ALT) => CONVERT au sens de la loi
            # => ne modifie PAS A, seulement la composition du portefeuille
            if from_asset and not is_fiat(from_asset) and to_asset:
                # on considère que le parser de note a trouvé from/to
                if from_asset:
                    holdings_qty[from_asset] -= abs(qty)   # approx
                if to_asset:
                    holdings_qty[to_asset] += abs(qty)
                continue

            # fallback : on considère que c'est un simple ajout de crypto
            holdings_qty[asset] += abs(qty)
            continue

        # 1.b) Dépôts / INCOME / airdrops -> acquisition à titre gratuit
        if side_norm == "DEPOSIT":
            # gains d'intérêt / staking etc. : coût d'acquisition = 0
            holdings_qty[asset] += abs(qty)
            # acquisition_cost_eur ne bouge pas
            continue

        # 1.c) Conversions crypto-crypto (neutres fiscalement)
        if side_norm == "CONVERT":
            # on ajuste juste les quantités, A ne change pas
            m = DIRECTION_RE.search(tx.note or "")
            if m:
                from_amount = float(m.group(1))
                from_sym = m.group(2).upper()
                to_amount = float(m.group(3))
                to_sym = m.group(4).upper()

                if not is_fiat(from_sym):
                    holdings_qty[from_sym] -= abs(from_amount)
                if not is_fiat(to_sym):
                    holdings_qty[to_sym] += abs(to_amount)
            else:
                # au pire, on traite le pair comme une entrée
                holdings_qty[asset] += abs(qty)
            continue

        # 1.d) Withdrawals de crypto / transferts sortants : on enlève des coins
        if side_norm == "WITHDRAWAL" and not is_fiat(asset):
            holdings_qty[asset] -= abs(qty)
            continue

        # ---------- 2) Cas imposables : SELL vers FIAT ----------
        if side_norm == "SELL" and is_taxable(tx) and not is_fiat(asset):
            # quantité vendue (on force le signe)
            if qty > 0:
                qty = -qty
            qty_sold = abs(qty)

            # Valeur de cession C en EUR (montant FIAT reçu)
            C = abs(trade_value_eur)

            # Valeur globale du portefeuille V juste AVANT la vente
            V = compute_portfolio_value_eur(
                db,
                holdings_qty,
                tx.datetime,
                price_cache,
                fx_cache,
            )
            if V <= 0 or acquisition_cost_eur <= 0:
                # pas de base de coût connue → on considère tout en PV brute
                pv = C
                allocated_cost = 0.0
            else:
                allocated_cost = acquisition_cost_eur * (C / V)
                pv = C - allocated_cost

            # Mise à jour du pool après la cession
            holdings_qty[asset] -= qty_sold
            acquisition_cost_eur = max(acquisition_cost_eur - allocated_cost, 0.0)

            # cumul annuel
            cum_pv_by_year[year] += pv
            cum_pv_year = cum_pv_by_year[year]
            est_tax = max(cum_pv_year, 0.0) * 0.30

            events_by_id[tx.id] = {
                "id": tx.id,
                "datetime": tx.datetime,
                "pair": asset,
                "side": side_norm,
                "proceeds_eur": C,
                "pv_eur": pv,
                "cum_pv_year_eur": cum_pv_year,
                "estimated_tax_eur": est_tax,
            }

            continue

        # Le reste (SELL non imposables, OTHER…) : on peut réduire le pool
        if side_norm == "SELL" and not is_fiat(asset):
            if qty > 0:
                qty = -qty
            holdings_qty[asset] += qty  # qty est négatif
            continue

    return events_by_id


def get_direction_label(tx: TransactionDB) -> str | None:
    s = normalize_side(tx)
    from_asset, to_asset = extract_from_to(tx)

    if s == "BUY" and from_asset:
        return f"Depuis {from_asset}"
    if s == "SELL" and to_asset:
        return f"Vers {to_asset}"
    if s == "CONVERT" and from_asset and to_asset:
        return f"{from_asset} → {to_asset}"
    return None

# ===================== Routes =====================

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/years")
def list_years(
        asset: str | None = Query(None),
        db: Session = Depends(get_db),
):
    query = db.query(func.extract("year", TransactionDB.datetime).label("y"))
    if asset:
        query = query.filter(TransactionDB.pair.ilike(f"%{asset}%"))

    years = query.distinct().order_by("y").all()
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

    rows = query.distinct().order_by(TransactionDB.pair.asc()).all()
    assets = [row.pair for row in rows if row.pair]
    return {"assets": assets}


@app.get("/transactions", response_model=List[TransactionOut])
def list_transactions(
        limit: int = Query(100, ge=1, le=5000),
        offset: int = Query(0, ge=0),
        year: int | None = Query(None),
        asset: str | None = Query(None),
        types: List[str] | None = Query(None),
        db: Session = Depends(get_db),
):
    query = db.query(TransactionDB)

    if year is not None:
        query = query.filter(func.extract("year", TransactionDB.datetime) == year)

    if asset:
        query = query.filter(TransactionDB.pair.ilike(f"%{asset}%"))

    rows = query.order_by(TransactionDB.datetime.desc()).all()

    normalized_rows = []
    for tx in rows:
        side_norm = normalize_side(tx)
        if side_norm == "HIDDEN":
            continue
        normalized_rows.append(tx)
    rows = normalized_rows

    if types:
        allowed = {TYPE_MAP.get(t.upper(), t.upper()) for t in types}
        rows = [tx for tx in rows if normalize_side(tx) in allowed]

    start = offset
    end = offset + limit
    page_rows = rows[start:end]

    tax_events = compute_tax_events_all_years(db)

    out: list[TransactionOut] = []
    for tx in page_rows:
        normalized = normalize_side(tx)
        tax_info = tax_events.get(tx.id)

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
                taxable=is_taxable(tx),
                direction=get_direction_label(tx),
                pv_eur=(tax_info["pv_eur"] if tax_info else None),
                cum_pv_year_eur=(
                    tax_info["cum_pv_year_eur"] if tax_info else None
                ),
                estimated_tax_eur=(
                    tax_info["estimated_tax_eur"] if tax_info else None
                ),
            )
        )
    return out


@app.get("/summary", response_model=SummaryOut)
def get_summary(
        year: int | None = Query(None),
        asset: str | None = Query(None),
        types: List[str] | None = Query(None),
        db: Session = Depends(get_db),
):
    query = db.query(TransactionDB)

    if year is not None:
        query = query.filter(func.extract("year", TransactionDB.datetime) == year)

    if asset:
        query = query.filter(TransactionDB.pair.ilike(f"%{asset}%"))

    rows = query.all()

    if types:
        allowed = {TYPE_MAP.get(t.upper(), t.upper()) for t in types}
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


@app.get("/portfolio/daily", response_model=List[DailyPortfolioOut])
def get_daily_portfolio(
        year: int | None = Query(None),
        db: Session = Depends(get_db),
):
    return compute_daily_portfolio(db, year=year)


@app.get("/tax/{year}")
def compute_tax(year: int, db: Session = Depends(get_db)):
    events_by_id = compute_tax_events_all_years(db)

    year_events = [
        e for e in events_by_id.values()
        if e["datetime"].year == year
    ]

    total_pv_eur = sum(e["pv_eur"] for e in year_events)
    flat_tax_30 = max(total_pv_eur, 0.0) * 0.30

    events_out = [
        {
            "id": e["id"],
            "datetime": e["datetime"],
            "pair": e["pair"],
            "side": e["side"],
            "proceeds_eur": e["proceeds_eur"],
            "pv_eur": e["pv_eur"],
        }
        for e in year_events
    ]

    return {
        "year": year,
        "total_pv_eur": total_pv_eur,
        "flat_tax_30": flat_tax_30,
        "events": events_out,
    }

# ===================== CRUD simple =====================

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
        taxable=is_taxable(tx_db),
        direction=get_direction_label(tx_db),
    )

# ===================== Import Binance =====================

@app.post("/import/binance")
async def import_binance(file: UploadFile = File(...), db: Session = Depends(get_db)):
    content = await file.read()
    s = content.decode("utf-8", errors="ignore")

    sample = s[:2048]
    dialect = csv.Sniffer().sniff(sample, delimiters=",;")

    rows = list(csv.DictReader(StringIO(s), dialect=dialect))

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

    if min_date is not None and max_date is not None:
        for sym in assets:
            ensure_coin_history(db, sym, min_date, max_date, base_currency="USD")

    composed_ops: dict[str, dict] = defaultdict(lambda: {
        "datetime": None,
        "account": None,
        "remark": None,
        "spends": [],
        "buys": [],
        "fees": [],
        "raw_ops": [],
    })

    simple_rows: list[dict] = []

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

        account_upper = account.upper()
        remark_upper = remark.upper()

        if "FUTURES" in account_upper or "FUTURES" in remark_upper:
            continue

        if not utc_time or not operation:
            continue

        try:
            dt = datetime.strptime(utc_time, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            try:
                dt = datetime.fromisoformat(utc_time.replace("Z", "+00:00"))
            except Exception:
                continue

        try:
            qty = float(str(change_str).replace(",", "."))
        except ValueError:
            qty = 0.0

        op_upper = operation.upper()
        remark_upper = remark.upper()

        # ---------- Cas simples ----------
        if op_upper in {"FIAT DEPOSIT", "DEPOSIT"}:
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

        if "SIMPLE EARN FLEXIBLE REDEMPTION" in op_upper:
            simple_rows.append({
                "datetime": dt,
                "side": "EARN_RETURN",
                "pair": coin,
                "quantity": qty,
                "note": f"{account} | {remark}".strip(" |"),
                "price_eur": 0.0,
                "fees_eur": 0.0,
            })
            continue

        if "SIMPLE EARN FLEXIBLE SUBSCRIPTION" in op_upper:
            simple_rows.append({
                "datetime": dt,
                "side": "SUBSCRIPTION",
                "pair": coin,
                "quantity": qty,
                "note": f"{account} | {remark}".strip(" |"),
            })
            continue

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

        # ---------- Binance Convert ----------
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

        # ---------- Cas composés (Transaction Spend / Revenue / Fee / ...) ----------
        group_key = f"{account}|{dt.strftime('%Y-%m-%d %H:%M:%S')}|{remark}"

        comp = composed_ops[group_key]
        comp["datetime"] = dt
        comp["account"] = account
        comp["remark"] = remark
        comp["raw_ops"].append(operation)

        op_u = operation.upper()

        if "TRANSACTION REVENUE" in op_u:
            # ce que tu reçois (USDT, EUR, ...)
            comp["buys"].append((coin, abs(qty)))
        elif "SPEND" in op_u or "SOLD" in op_u or op_u == "SELL":
            # ce que tu vends (LUNA, NEO, ...)
            comp["spends"].append((coin, qty))  # qty déjà négatif
        elif "BUY" in op_u:
            comp["buys"].append((coin, qty))
        elif "FEE" in op_u:
            comp["fees"].append((coin, qty))
        else:
            comp["buys"].append((coin, qty))
            comp["raw_ops"].append(f"FALLBACK_OTHER:{operation}")

    inserted = 0
    usd_stables = {"USDT", "USDC", "BUSD", "USD"}

    # ---------- insertion des simples ----------
    for r in simple_rows:
        dt = r["datetime"]
        qty = r["quantity"]
        pair = r["pair"]
        side = r["side"]

        price_eur = r.get("price_eur", 0.0)
        fees_eur = r.get("fees_eur", 0.0)

        if pair == "EUR":
            price_eur = abs(qty)

        if (
                price_eur == 0
                and pair not in {"EUR"} | usd_stables
                and side in {"DEPOSIT", "WITHDRAWAL", "INCOME"}
        ):
            price_usd = get_coin_price_usd(db, pair, dt)
            if price_usd is not None:
                fx = get_usd_eur_rate(db, dt)
                price_eur = abs(qty) * price_usd * fx

        tx = TransactionDB(
            datetime=dt,
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

    # ---------- insertion des composés ----------
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

        total_spent_eur = sum(abs(qty) for coin, qty in spends if coin == "EUR")
        total_fees_eur = sum(abs(qty) for coin, qty in fees if coin == "EUR")

        usd_spent = sum(abs(qty) for coin, qty in spends if coin in usd_stables)
        usd_fees = sum(abs(qty) for coin, qty in fees if coin in usd_stables)

        fx = get_usd_eur_rate(db, dt)

        total_spent_eur += usd_spent * fx
        total_fees_eur += usd_fees * fx

        total_received_eur = sum(abs(qty) for coin, qty in buys if coin == "EUR")
        usd_received = sum(abs(qty) for coin, qty in buys if coin in usd_stables)
        total_received_eur += usd_received * fx

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

        if (
                total_spent_eur == 0
                and from_asset is None
                and to_asset is not None
                and any("BUY CRYPTO WITH FIAT" in op.upper() for op in raw_ops)
        ):
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
        price_eur = 0.0  # on choisira ensuite

        # BUY : crypto achetée contre FIAT (EUR, USD...)
        if is_fiat(from_asset) and to_asset and not is_fiat(to_asset):
            side = "BUY"
            pair = to_asset
            quantity = to_amount
            # montant dépensé = jambe "spends" (FIAT)
            price_eur = total_spent_eur or total_received_eur

        # SELL : crypto vendue contre FIAT
        elif is_fiat(to_asset) and from_asset and not is_fiat(from_asset):
            side = "SELL"
            pair = from_asset
            quantity = from_amount
            # montant reçu = jambe "buys" (FIAT)
            price_eur = total_received_eur or total_spent_eur

        # CONVERT crypto <-> crypto (y compris stables)
        elif from_asset and to_asset:
            side = "CONVERT"
            pair = to_asset
            quantity = to_amount

        elif to_asset and not from_asset:
            # fallback : achat de crypto (fiat implicite ou autre)
            side = "BUY"
            pair = to_asset
            quantity = to_amount

        if (
                price_eur == 0
                and side in {"BUY", "SELL", "CONVERT"}
                and pair not in {"EUR"} | usd_stables
        ):
            price_usd = get_coin_price_usd(db, pair, dt)
            if price_usd is not None:
                price_eur = abs(quantity) * price_usd * fx

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

# ===================== Import FX USD/EUR =====================

@app.post("/import/fx-usdeur")
async def import_fx_usdeur(
        file: UploadFile = File(...),
        db: Session = Depends(get_db),
):
    content = await file.read()
    tree = ET.fromstring(content)

    NS = {"exr": "http://www.ecb.europa.eu/vocabulary/stats/exr/1"}
    series_list = tree.findall(".//exr:Series", NS)

    if not series_list:
        return {"inserted": 0, "detail": "Aucune série trouvée"}

    inserted = 0

    for series in series_list:
        attrs = series.attrib
        curr = attrs.get("CURRENCY")
        denom = attrs.get("CURRENCY_DENOM")

        if curr != "USD" or denom != "EUR":
            continue

        db.query(FXRate).filter(
            FXRate.base == "USD",
            FXRate.quote == "EUR",
            ).delete()

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