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
