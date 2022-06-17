#pip3 install fastapi uvicorn
#uvicorn part2:api --reload

from fastapi import FastAPI
from fastapi import Header
from pydantic import BaseModel
from fastapi import HTTPException

api = FastAPI(
    title="Fraud detection API",
    description="This API interrogates various maching learning models.You can access performances of the algorithms on the test sets.",
    version="1.0.0")

@api.get('/welcome')
def welcome_here():
    """Returns greetings
    """
    return "Welcome"

users_db = [
    {
        'user_name': 'Alice',
        'user_password': 'wonderland'
    },
    {
        'user_name': 'Clementine',
        'user_password': 'mandarine'
    },
    {
        'user_name': 'Bob',
        'user_password': 'builder'
    }
]

class Authorization(BaseModel):
    name:str
    password:str

@api.post('/name/{name:str}/password/{password:str}')
def authorization_users(user: Authorization, name, password):
    try:
        identif_user_name = list(
            filter(lambda x: x.get('user_name') == name and x.get('user_password') == password, users_db)
            )[0]
        if identif_user_name['user_name']== name and identif_user_name['user_password']== password:
            return "Hello. You can start to explore models."
    except IndexError:
        raise HTTPException(
            status_code=404,
            detail='Unknown username or password')
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail='Bad Type'
        )