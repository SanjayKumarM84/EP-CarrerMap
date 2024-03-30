from fastapi.responses import JSONResponse

def successResponse(content, code=200):
    response = {
        'message': 'success',
        'content': content
    }
    jsonResponse = JSONResponse(
        content=response
    )
    jsonResponse.status_code = code
    return jsonResponse

def failureResponse(content, code=400):
    response = {
        'message': 'failure',
        'content': content
    }
    jsonResponse = JSONResponse(
        content=response
    )
    jsonResponse.status_code = code
    return jsonResponse
