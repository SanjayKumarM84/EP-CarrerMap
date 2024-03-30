from fastapi import FastAPI, HTTPException, Query
from commonModules import *
from typing import Dict, Any
import uvicorn

app = FastAPI()

@app.get("/")
async def home():
    try:
        return successResponse("Hello World")
    except Exception as e:
        raise failureResponse(str(e))

@app.post("/predictJobRole")
async def predictJobRole(payload: Dict[str, Any]):
    try:
        return successResponse(payload)
    except Exception as e:
        raise failureResponse(str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)
