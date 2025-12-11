from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def sample(a:int,b:int):
    return{"Addition of Two":int(a + b)}