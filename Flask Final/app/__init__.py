from flask import Flask
app = Flask(__name__)

from app import views
from app import db
from app import analysis



app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config["SECRET_KEY"] = 'my unobvious secret key'