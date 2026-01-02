from flask import Blueprint, render_template, redirect, url_for, request, flash
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import login_user, logout_user, login_required
from database import User, db

auth = Blueprint('auth', __name__)

@auth.route('/login')
def login():
    return render_template('login.html')

@auth.route('/login', methods=['POST'])
def login_post():
    username = request.form.get('username')
    password = request.form.get('password')
    remember = True if request.form.get('remember') else False
    
    user = User.query.filter_by(username=username).first()
    
    # Check if user exists and password is correct
    if not user or not check_password_hash(user.password_hash, password):
        flash('Please check your login details and try again.')
        return redirect(url_for('auth.login'))
    
    login_user(user, remember=remember)
    
    # Redirect to appropriate dashboard based on user type
    if user.is_teacher:
        return redirect(url_for('teacher.dashboard'))
    else:
        return redirect(url_for('student.dashboard'))

@auth.route('/signup')
def signup():
    return render_template('login.html', signup=True)

@auth.route('/signup', methods=['POST'])
def signup_post():
    username = request.form.get('username')
    email = request.form.get('email')
    password = request.form.get('password')
    is_teacher = True if request.form.get('user_type') == 'teacher' else False
    
    # Check if user already exists
    user = User.query.filter_by(email=email).first()
    if user:
        flash('Email address already exists.')
        return redirect(url_for('auth.signup'))
    
    # Create new user
    new_user = User(
        username=username,
        email=email,
        password_hash=generate_password_hash(password, method='sha256'),
        is_teacher=is_teacher
    )
    
    db.session.add(new_user)
    db.session.commit()
    
    return redirect(url_for('auth.login'))

@auth.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('auth.login'))