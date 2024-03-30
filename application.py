from fastapi import FastAPI, HTTPException, Query
from commonModules import *
from typing import Dict, Any
import uvicorn
import json
from prediction import predictMain

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

        # # Convert the data to JSON string
        # data_json = json.dumps(payload)

        prediction_result = predictMain(payload)
        print(prediction_result)
        return successResponse(prediction_result)
    except Exception as e:
        raise failureResponse(str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0")
