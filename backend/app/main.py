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

import re

DIRECTION_RE = re.compile(
    r"From\s+([-\d\.]+)\s+([A-Z0-9]+)\s+->\s+([-\d\.]+)\s+([A-Z0-9]+)"
)

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

    # 🆕
    taxable: bool
    direction: str | None = None

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

# ---------- DB utils ----------

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

import re
from collections import defaultdict

def compute_daily_portfolio(db: Session, year: int | None = None) -> list[DailyPortfolioOut]:
    """
    Reconstruit le portefeuille jour par jour à partir des transactions.

    Hypothèses :
      - On travaille en scope "Binance only" (ce que voit la DB).
      - BUY / DEPOSIT / INCOME augmentent la quantité de l'asset.
      - SELL / WITHDRAWAL la réduisent.
      - CONVERT : on parse le note "From X BTC -> Y BCH" pour décrémenter X et
        incrémenter Y (sinon on ignore le from, fallback ≈ BUY).
    """

    # 1. On récupère toutes les transactions dans l’ordre chronologique
    query = db.query(TransactionDB)
    if year is not None:
        query = query.filter(func.extract("year", TransactionDB.datetime) == year)

    txs: list[TransactionDB] = query.order_by(TransactionDB.datetime.asc()).all()
    if not txs:
        return []

    # 2. On prépare
    holdings: dict[str, float] = defaultdict(float)  # quantité par coin
    by_day: dict[date, list[TransactionDB]] = defaultdict(list)

    for tx in txs:
        d = tx.datetime.date()
        by_day[d].append(tx)

    all_days = sorted(by_day.keys())
    results: list[DailyPortfolioOut] = []

    prev_total_value = 0.0

    for d in all_days:
        day_txs = by_day[d]
        net_flow_eur = 0.0  # flux "externe" du jour (EUR qui rentre/sort)

        # 3. On applique les transactions du jour sur les holdings
        for tx in day_txs:
            side = normalize_side(tx)
            pair = (tx.pair or "").upper()
            qty = tx.quantity or 0.0

            # a) flux externes en EUR pour le PnL
            if pair == "EUR" and side in {"DEPOSIT", "WITHDRAWAL"}:
                if side == "DEPOSIT":
                    net_flow_eur += abs(tx.price_eur or 0.0)
                else:
                    net_flow_eur -= abs(tx.price_eur or 0.0)

            # b) mise à jour des quantités crypto
            if pair == "EUR" or is_fiat_or_stable(pair):
                # on ne stocke pas EUR/USDT/... dans les holdings de l’onglet portfolio
                continue

            if side in {"BUY", "DEPOSIT", "INCOME"}:
                holdings[pair] += abs(qty)

            elif side in {"SELL", "WITHDRAWAL"}:
                holdings[pair] -= abs(qty)

            elif side == "CONVERT":
                # On essaye de parser le FROM/TO depuis le note
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
                    # fallback : on considère juste que c’est un BUY du pair
                    holdings[pair] += abs(qty)

        # 4. Valorisation du portefeuille ce jour-là
        day_total_value = 0.0
        positions_out: list[PositionOut] = []

        day_dt = datetime.combine(d, datetime.min.time())

        for sym, q in list(holdings.items()):
            # on nettoie les positions quasi-nulles
            if abs(q) < 1e-12:
                holdings.pop(sym, None)
                continue

            if is_fiat_or_stable(sym) or sym == "EUR":
                # l’onglet portfolio = seulement les coins "vrais"
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

        # 5. PnL du jour (approx) : variation de valeur - flux net EUR
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

    # 6. Flux Earn internes : on les CACHE
    if raw in {"SUBSCRIPTION", "EARN_RETURN"}:
        return "HIDDEN"

    return "OTHER"

def extract_from_to(tx: TransactionDB) -> tuple[str | None, str | None]:
    """
    Extrait (from_asset, to_asset) depuis la note "From X COIN -> Y COIN2".
    """
    m = DIRECTION_RE.search(tx.note or "")
    if not m:
        return None, None
    from_asset = m.group(2)
    to_asset = m.group(4)
    return from_asset, to_asset


def is_taxable(tx: TransactionDB) -> bool:
    """
    Approximation FR : imposable seulement quand on sort vers l'EUR.
    - SELL vers EUR
    - CONVERT où l'une des jambes est EUR (cas rare, normalement classé SELL/BUY)
    """
    s = normalize_side(tx)
    if s not in {"SELL", "CONVERT"}:
        return False

    from_asset, to_asset = extract_from_to(tx)

    if from_asset == "EUR" or to_asset == "EUR":
        return True

    # Cas ultra simple : si note absente mais pair == "EUR" (devrait peu arriver)
    if tx.pair.upper() == "EUR":
        return True

    return False


def get_direction_label(tx: TransactionDB) -> str | None:
    """
    Texte "Vers/Depuis" pour affichage UI.
    """
    s = normalize_side(tx)
    from_asset, to_asset = extract_from_to(tx)

    if s == "BUY" and from_asset:
        return f"Depuis {from_asset}"
    if s == "SELL" and to_asset:
        return f"Vers {to_asset}"
    if s == "CONVERT" and from_asset and to_asset:
        return f"{from_asset} → {to_asset}"
    return None

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

    normalized_rows = []
    for tx in rows:
        side_norm = normalize_side(tx)
        if side_norm == "HIDDEN":  # 👈 ON CACHE
            continue
        normalized_rows.append(tx)
    rows = normalized_rows

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
                taxable=is_taxable(tx),          # 🆕
                direction=get_direction_label(tx) # 🆕
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

@app.get("/portfolio/daily", response_model=List[DailyPortfolioOut])
def get_daily_portfolio(
        year: int | None = Query(None),
        db: Session = Depends(get_db),
):
    """
    Time-series du portefeuille (Binance seulement), valorisé en EUR par jour.
    Utilisable pour :
      - graphe de valeur totale jour par jour
      - breakdown par asset à la date sélectionnée
      - PnL journalier approximatif
    """
    return compute_daily_portfolio(db, year=year)

@app.get("/tax/{year}")
def compute_tax(year: int, db: Session = Depends(get_db)):
    """
    Calcul simplifié des plus-values :
      - moyenne d'achat par actif (average cost per coin)
      - évènements taxables = SELL (y compris vers USDT/USDC/BUSD/EUR)
      - WITHDRAWAL = transfert (ignoré)
      - SUBSCRIPTION / EARN_RETURN / CONVERT = ignorés pour la fiscalité

    ATTENTION : ce n'est PAS la méthode exacte française (valeur globale du portefeuille),
    mais un modèle cohérent et vérifiable coin par coin.
    """

    # On prend toutes les transactions jusqu'à fin de l'année pour avoir l'historique complet
    end_dt = datetime(year, 12, 31, 23, 59, 59)

    txs = (
        db.query(TransactionDB)
        .filter(TransactionDB.datetime <= end_dt)
        .order_by(TransactionDB.datetime.asc())
        .all()
    )

    # Position et coût moyen par actif
    holdings_qty: dict[str, float] = defaultdict(float)
    holdings_cost: dict[str, float] = defaultdict(float)

    events: list[dict] = []
    total_pv_eur = 0.0

    for tx in txs:
        side = normalize_side(tx)      # BUY / SELL / DEPOSIT / WITHDRAWAL / CONVERT / OTHER / HIDDEN / TRANSFER
        asset = (tx.pair or "").upper()
        if not asset:
            continue

        # On ignore explicitement ce qu'on a marqué comme caché / transfert interne
        if side in {"HIDDEN", "TRANSFER"}:
            continue

        qty = tx.quantity or 0.0
        if qty == 0:
            continue

        trade_value = tx.price_eur or 0.0  # valeur totale en EUR (pas prix unitaire)

        # ----- ACQUISITIONS (on augmente le pool) -----
        if side in {"BUY", "DEPOSIT", "INCOME"}:
            # qty positive (on reçoit)
            if qty < 0:
                qty = -qty
            holdings_qty[asset] += qty
            holdings_cost[asset] += abs(trade_value)
            continue

        # ----- VENTES (évènements taxables) -----
        if side == "SELL":
            # on vend → qty doit être négative dans le CSV, on convertit
            if qty > 0:
                qty = -qty
            qty_sold = abs(qty)

            prev_qty = holdings_qty[asset]
            prev_cost = holdings_cost[asset]

            if prev_qty > 0:
                unit_cost = prev_cost / prev_qty
            else:
                # aucun historique → on considère coût nul (toute la vente est PV)
                unit_cost = 0.0

            cost_out = unit_cost * qty_sold
            proceeds = abs(trade_value)    # montant de la vente en EUR
            pv = proceeds - cost_out       # plus-value (peut être négative)

            # On met à jour la position résiduelle
            new_qty = max(prev_qty - qty_sold, 0.0)
            new_cost = max(prev_cost - cost_out, 0.0)

            holdings_qty[asset] = new_qty
            holdings_cost[asset] = new_cost

            # On ne comptabilise la PV que si la vente est dans l'année demandée
            if tx.datetime.year == year:
                total_pv_eur += pv
                events.append(
                    {
                        "id": tx.id,
                        "datetime": tx.datetime,
                        "pair": asset,
                        "side": side,
                        "proceeds_eur": proceeds,
                        "pv_eur": pv,
                    }
                )

            continue

        # ----- Le reste : WITHDRAWAL, CONVERT, OTHER, etc. -----
        # -> ignorés pour la fiscalité dans cette version
        continue

    flat_tax_30 = total_pv_eur * 0.30 if total_pv_eur > 0 else 0.0

    return {
        "year": year,
        "total_pv_eur": total_pv_eur,
        "flat_tax_30": flat_tax_30,
        "events": events,
    }
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

        # Simple Earn Flexible Redemption  → transfert interne (non taxable)
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
    usd_stables = {"USDT", "USDC", "BUSD", "USD"}  # tu peux remonter ça en haut si tu veux

    for r in simple_rows:
        dt = r["datetime"]
        qty = r["quantity"]
        pair = r["pair"]
        side = r["side"]

        price_eur = r.get("price_eur", 0.0)
        fees_eur = r.get("fees_eur", 0.0)

        # Si c'est de l'EUR qui rentre / sort → montant en EUR direct
        if pair == "EUR":
            price_eur = abs(qty)

        # 💰 Fallback pour dépôts / retraits / income en crypto :
        # on valorise à prix_jour(coin_USD) * fx * quantité
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

        # 🔥 Fallback : si on n’a toujours pas de prix EUR
        # (BUY / SELL / CONVERT crypto-crypto)
        # on prend prix_jour(coin en USD) * fx * quantité
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