from app import app
import sqlite3
import pandas as pd
from flask import flash, render_template
from passlib.hash import sha256_crypt


create_users_table = """ CREATE TABLE IF NOT EXISTS users (
                                        Username text NOT NULL,
                                        Email text UNIQUE,
                                        Password text NOT NULL                                      
                                    ); """



def create_connection(db_file):
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        print(sqlite3.version)
    except Exception as e:
        print(e)
    return conn

def create_table(db_file, create_table_sql):
    try:
        conn = sqlite3.connect(db_file)
        c = conn.cursor()
        c.execute(create_table_sql)
        c.close()
    except Exception as e:
        print(e)
    
    

def add_user(db_file,name,mail,passw):

    password = sha256_crypt.encrypt(passw)

    add_user_query ="""INSERT INTO users (Username,Email,Password) VALUES ('%s', '%s','%s'); """ % (name,mail,password)
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    try:
        c.execute("SELECT * FROM users WHERE Email=?;", (mail,))
        if c.fetchone() is not None:
            flash("That email is already in use!")
            return render_template('signup.html')
        else:
            c.execute(add_user_query )
            conn.commit()
            conn.close()
    except Exception as e:
        print(e)


def login_user(db_file,mail,passw):
    logged=False
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    try:
        c.execute('SELECT Password FROM users WHERE Email=? ', (mail, ))
        result=c.fetchone() 
        if sha256_crypt.verify(passw, result[0]):

            logged=True
               
        else:
            flash("This account does not exist")
            
    except Exception as e:
        print(e)
    return logged




                                    
