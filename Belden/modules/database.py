from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import datetime

Base = declarative_base()

class Asset(Base):
    __tablename__ = 'assets'
    device_id = Column(String, primary_key=True)
    name = Column(String)
    device_type = Column(String) # switch, plc, hmi, etc.
    location = Column(String)
    parent_id = Column(String, ForeignKey('assets.device_id'), nullable=True)
    
    metrics = relationship("Metric", back_populates="asset")

class Metric(Base):
    __tablename__ = 'metrics'
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    device_id = Column(String, ForeignKey('assets.device_id'))
    metric_name = Column(String)
    value = Column(Float)
    
    asset = relationship("Asset", back_populates="metrics")

# Database setup
DATABASE_URL = "sqlite:///./network_diagnostics.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
