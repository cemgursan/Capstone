
from app import app
from app import db
import os

from flask import Flask, session


cwd = os.getcwd()

if __name__ == "__main__": 

    dbcon=db.create_connection(cwd+"/tutorials.db")
    
    if dbcon is not None:
            # create projects table
        db.create_table(cwd+"/tutorials.db", db.create_users_table )
    else:
        print("Error! cannot create the database connection.")

    app.run()