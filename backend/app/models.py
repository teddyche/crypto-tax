from sqlalchemy import Column, Integer, String, DateTime, Float, Text
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
