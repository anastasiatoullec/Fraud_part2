#pip3 install fastapi uvicorn
#pip install passlib[bcrypt]
#uvicorn part2:api --reload
#http://127.0.0.1:8000/docs ou http://localhost:8000/docs

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from passlib.context import CryptContext

api = FastAPI(
    title="Fraud detection API",
    description="This API interrogates various maching learning models.You can access performances of the algorithms on the test sets.",
    version="1.0.0")

security = HTTPBasic()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

users = {

    "Alice": {
        "username": "Alice",
        "hashed_password": pwd_context.hash('wonderland'),
    },

    "Clementine" : {
        "username" :  "Clementine",
        "hashed_password" : pwd_context.hash('mandarine'),
    },

    "Bob" : {
        "username" :  "Bob",
        "hashed_password" : pwd_context.hash('builder'),
    }

}

def get_current_user(credentials: HTTPBasicCredentials = Depends(security)):
    username = credentials.username
    if not(users.get(username)) or not(pwd_context.verify(credentials.password, users[username]['hashed_password'])):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

@api.get("/user")
def current_user(username: str = Depends(get_current_user)):
    return "Hello {}".format(username)