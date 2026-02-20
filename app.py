from flask import Flask, render_template, request, redirect, url_for, session, flash
import os
import json
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "replace_with_secure_random")

USERS_FILE = "users.json"

# Load user data
def load_users():
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE, "r") as f:
        return json.load(f)

# Save user data
def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f)

# Home redirects based on session
@app.route('/')
def home():
    if 'username' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

# Login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        users = load_users()
        username = request.form['username']
        password = request.form['password']
        if username in users and check_password_hash(users[username], password):
            session['username'] = username
            return redirect(url_for('dashboard'))
        flash("Invalid username or password", "error")
    return render_template('auth.html', action='login')

# Register route
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        users = load_users()
        username = request.form['username']
        password = request.form['password']
        confirm = request.form['confirm']
        if not username or not password:
            flash("Username and password are required.", "error")
        elif password != confirm:
            flash("Passwords do not match.", "error")
        elif username in users:
            flash("Username already exists.", "error")
        else:
            users[username] = generate_password_hash(password)
            save_users(users)
            flash("Registration successful! Please login.", "success")
            return redirect(url_for('login'))
    return render_template('auth.html', action='register')

# Dashboard route
@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html', username=session['username'])

# Diagnose route
@app.route('/diagnose')
def diagnose():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('diagnose.html', username=session['username'])

# Logout route
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# Start Flask app
if __name__ == "__main__":
    app.run(debug=True, port=5000)
