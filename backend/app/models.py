from sqlalchemy import Column, Integer, String, Float, Date, UniqueConstraint, DateTime, Text
from .db import Base

class TransactionDB(Base):
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, index=True)
    datetime = Column(DateTime, nullable=False)
    side = Column(Text, nullable=False)
    pair = Column(Text, nullable=False)
    note = Column(Text, nullable=True)
    exchange = Column(String(50), nullable=False)
    quantity = Column(Float, nullable=False)
    price_eur = Column(Float, nullable=False)
    fees_eur = Column(Float, nullable=False)

class FXRate(Base):
    __tablename__ = "fx_rates"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, index=True)     # 2021-01-01
    base = Column(String(8), index=True)   # "USD"
    quote = Column(String(8), index=True)  # "EUR"
    rate = Column(Float)               # 1 base -> rate quote

    __table_args__ = (
        UniqueConstraint("date", "base", "quote", name="uq_fxrate_date_pair"),
    )

class CoinMeta(Base):
    """
    Mapping entre un symbole (BCH, AVAX, etc.) et l'id de l'API CCC.
    On peut aussi stocker quelques infos utiles.
    """
    __tablename__ = "coin_meta"

    id = Column(Integer, primary_key=True, index=True)
    api_id = Column(Integer, index=True)      # id = 2487 pour BCH
    symbol = Column(String(16), index=True)   # "BCH"
    name = Column(String(128))
    first_data = Column(Date, nullable=True)
    most_recent_data = Column(Date, nullable=True)
    base_currency = Column(String(8), nullable=True)  # "USD" en général

    __table_args__ = (
        UniqueConstraint("symbol", name="uq_coinmeta_symbol"),
    )


class CoinPrice(Base):
    __tablename__ = "coin_prices"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, index=True)           # jour UTC
    symbol = Column(String(16), index=True)   # "BCH"
    base = Column(String(8), index=True)      # "USD" ou "EUR"
    price = Column(Float)                     # 1 coin -> price base

    __table_args__ = (
        UniqueConstraint("date", "symbol", "base", name="uq_coinprice_date_symbol_base"),
    )