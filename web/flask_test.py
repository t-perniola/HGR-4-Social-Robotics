from flask import Flask, render_template, redirect, url_for, request, session, flash, get_flashed_messages
from datetime import timedelta
  
app = Flask(__name__)
app.secret_key = "ciro"
app.permanent_session_lifetime = timedelta(minutes=5) # per quanto manteniamo la sessione?
  
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/login", methods = ["GET", "POST"])
def login():
    if request.method == "POST": # se nome stato inserito tramite form, quindi con il metodo POST...
      session.permanent = True
      user = request.form["nm"] # usiamo dizionario e chiave per accedere al valore corrispondente
      session["user"] = user # salviamo il nome utente mantenendo la sessione aperta
      return redirect(url_for("user")) # manderemo il nome inserito e reindirizzeremo alla funzione sotto
    
    else:
        if "user" in session: # se sono già registrato, non devo riandare al login
            return redirect(url_for("user"))
      
        return render_template("login.html")
    
@app.route("/user")
def user():
    if "user" in session:
        user = session["user"]
        return f"<h1>{user}</h1>"
    else:
        return redirect(url_for("login"))
    
@app.route("/logout")
def logout():
    if "user" in session:
        user = session["user"]
        flash("Logged out", "info")
    session.pop("user", None)
    return redirect(url_for("login"))
  
if __name__ == "__main__":
  app.run(debug=True)