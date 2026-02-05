from fastapi import FastAPI, HTTPException 

app = FastAPI()

textPosts = {
    1: {"title": "Sample 1", "content": "Content 1"},
    2: {"title": "Sample 2", "content": "Content 2"},
    3: {"title": "Sample 3", "content": "Content 3"},
    4: {"title": "Sample 4", "content": "Content 4"},
    5: {"title": "Sample 5", "content": "Content 5"},
    6: {"title": "Sample 6", "content": "Content 6"},
    7: {"title": "Sample 7", "content": "Content 7"},
    8: {"title": "Sample 8", "content": "Content 8"},
    9: {"title": "Sample 9", "content": "Content 9"},
    10: {"title": "Sample 10", "content": "Content 10"},
}

@app.get("/posts")
def getAllPosts():
    return textPosts

@app.get("/posts/{id}")
def getPost(id: int):
    if id not in textPosts:
        raise HTTPException(status_code= 404, detail="ID not found")
    return textPosts.get(id)