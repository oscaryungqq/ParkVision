from fastapi import FastAPI
from .endpoints import tracking

app = FastAPI(title="ParkVision_API")

app.include_router(tracking.router, prefix="/api/v1", tags=["Tracking"])

@app.get("/")
def read_root():

    return {"message": "Welcome to the ParkVision API"}
