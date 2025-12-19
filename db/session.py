from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# The database URL. Format: "postgresql://user:password@host/dbname"
SQLALCHEMY_DATABASE_URL = "postgresql://postgres:password@172.17.36.136/postgres"

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()