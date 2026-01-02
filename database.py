from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime

db = SQLAlchemy()

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)
    is_teacher = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    assignments = db.relationship('Assignment', backref='teacher', lazy=True)
    submissions = db.relationship('Submission', backref='student', lazy=True)

class Assignment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text, nullable=False)
    question_file = db.Column(db.String(200))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    deadline = db.Column(db.DateTime)
    teacher_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    
    # Relationships
    submissions = db.relationship('Submission', backref='assignment', lazy=True)
    similarity_results = db.relationship('SimilarityResult', backref='assignment', lazy=True)

class Submission(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(200), nullable=False)
    file_path = db.Column(db.String(200), nullable=False)
    submitted_at = db.Column(db.DateTime, default=datetime.utcnow)
    student_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    assignment_id = db.Column(db.Integer, db.ForeignKey('assignment.id'), nullable=False)
    
    # Relationships
    similarity_results = db.relationship('SimilarityResult', backref='submission', lazy=True)

class SimilarityResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    result_type = db.Column(db.String(20), nullable=False)  # 'student_vs_student' or 'student_vs_ai'
    result_data = db.Column(db.Text)  # JSON string
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    assignment_id = db.Column(db.Integer, db.ForeignKey('assignment.id'), nullable=False)
    submission_id = db.Column(db.Integer, db.ForeignKey('submission.id'), nullable=True)