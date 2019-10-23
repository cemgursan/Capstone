from app import app
from app import db
from app import analysis as ana
from bokeh.embed import components 

from bokeh.plotting import figure
from bokeh.embed import components 
from bokeh.io import output_file, output_notebook, show
import os
cwd = os.getcwd()

from io import BytesIO
import base64


from flask import flash, render_template,request,redirect,flash, session,send_file

passed="Cem"

@app.route("/")
def initialize():
    session['logged_in'] = False
    return render_template("home.html" , value=passed)
    
@app.route("/home.html")
def home():
    return render_template("home.html" , value=passed)

@app.route("/about.html")
def about():
    return render_template("about.html")

@app.route("/signin.html", methods=["GET","POST"])
def signin():
    if request.method == "POST":

        mail = request.form["email"]
        passw = request.form["password"]
        loggedBool = db.login_user(cwd+"/tutorials.db",mail,passw)
        
        if loggedBool:
            session['logged_in'] = True
            return render_template("logged.html")
        else:
            session['logged_in'] = False
            return render_template("signin.html")

    if session['logged_in']== True:
        return render_template("logged.html")   

    return render_template("signin.html")
    

@app.route("/signup.html", methods=["GET","POST"])
def signup():

    if request.method == "POST":
        
        name =  request.form["username"]
        mail = request.form["email"]
        passw = request.form["password"]
        
        ## Requested has isgned up data in dictionary put this in database
        db.add_user(cwd+"/tutorials.db",name,mail,passw)

        return render_template("signup.html")


    return render_template("signup.html")

@app.route("/jinja.html")
def jinja():
    flash("Hello from flashed message")
    passed="This value is passed to HTML by Flask"
    langs=["Python", "Flask","SQLlite","Pandas","Seaborn"]

    simplealert="<script> alert('This is a script passed by Flask') </script>"

    def Jinjafunc(stringtowrite):
        stringtowrite= "This string was written by a function from Python :" + stringtowrite
        return stringtowrite

    return render_template("jinja.html", value=passed,lang=langs,alert=simplealert,Jfunc=Jinjafunc)

@app.route("/logged.html")
def logged():
    if session['logged_in'] == False:
        return render_template("signin.html")
    else:
        return render_template("logged.html")

@app.route("/stockprices", methods=["GET","POST"])
def stockprices():
    if session['logged_in'] == False:
        return render_template("signin.html")
    else:
        df=ana.SetupStocks("WIKI/KO","WIKI/AAPL","WIKI/GOOGL")

        if request.method == "POST":
            stockname =  request.form.get("dropdown", None)
            
        else:
            stockname="Google"

        MAplot=ana.PlotMAStockPrice(df,stockname,10,20,30)  
        script1, div1 = components(MAplot)
        return render_template("stockprices.html", stock_name= stockname, the_div1=div1, the_script1=script1)

@app.route("/plots", methods=["GET","POST"])
def plots():
    if session['logged_in'] == False:
        return render_template("signin.html")
    else:
        df=ana.SetupStocks("WIKI/KO","WIKI/AAPL","WIKI/GOOGL")

        if request.method == "POST":
            stock1 =  request.form.get("dropdown_first", None)
            stock2 =  request.form.get("dropdown_second", None)
            
        else:
            stock1 =  "Google"
            stock2 =  "Apple"
        
        #Save Seaborn plots as PNG and pass it as string and decode it in flask
        figfile = BytesIO()
        JointPlot= ana.JointPlot(df,stock1,stock2)
        JointPlot.savefig(figfile, format='png',dpi=300)
        figfile.seek(0)
        figdata_png = figfile.getvalue()
        figdata_png = base64.b64encode(figdata_png)

        figfile2 = BytesIO()
        PairPlot = ana.PairPlot(df)
        PairPlot.savefig(figfile2, format='png',dpi=300)
        figfile2.seek(0)
        figdata2_png = figfile2.getvalue()
        figdata2_png = base64.b64encode(figdata2_png)


        return render_template("plots.html",stock1=stock1, stock2=stock2, jointplot=figdata_png.decode('ascii'),pairplot=figdata2_png.decode('ascii') )

@app.route("/montecarlo", methods=["GET","POST"])
def montecarlo():
    if session['logged_in'] == False:
        return render_template("signin.html")
    else:
        df=ana.SetupStocks("WIKI/KO","WIKI/AAPL","WIKI/GOOGL")
        if request.method == "POST":
            stockname =  request.form.get("dropdown", None)
            simulationdays  =  int(request.form["days"])
            simulationruns  =  int(request.form["runs"])
            confidence  =  int(request.form["confidence"])
        else:
            stockname =  "Google"
            simulationdays  =  100
            simulationruns  =  1000
            confidence  =  5
        MonteCarloPic = ana.MonteCarloGraph(df,stockname,simulationdays,simulationruns)
        script1, div1 = components(MonteCarloPic)

        figfile = BytesIO()
        MonteDistribution = ana.MonteCarloDistribution(df,stockname,simulationdays,simulationruns,confidence)
        MonteDistribution.savefig(figfile,format='png',dpi=300)
        figfile.seek(0)
        figdata_png = figfile.getvalue()
        figdata_png = base64.b64encode(figdata_png)

        return render_template("montecarlo.html", MonteDist=figdata_png.decode('ascii'), stockname=stockname, simdays=simulationdays,simruns=simulationruns, conf=confidence, the_div1=div1, the_script1=script1)

@app.route("/machinelearning", methods=["GET","POST"])
def machinelearning():
    if session['logged_in'] == False:
        return render_template("signin.html")
    else:
        if request.method == "POST":
            stockname =  request.form.get("dropdown", None)
        else:
            stockname =  "Google"
        
        if stockname=="Apple":
            wikiname="WIKI/AAPL"
        elif stockname=="Coca Cola":
            wikiname="WIKI/KO"
        else:
            wikiname="WIKI/GOOGL"
        MLPlot=ana.SetupStockML(wikiname)
        script1, div1 = components(MLPlot)
        return render_template("machinelearning.html", stockname=stockname, the_div1=div1, the_script1=script1)

@app.route("/logout")
def logout():
    session['logged_in'] = False
    return render_template("home.html" , value=passed)