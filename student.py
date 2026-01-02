from flask import Blueprint, render_template, request, redirect, url_for, flash, current_app
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
from database import db, Assignment, Submission
import os
import datetime

student = Blueprint('student', __name__, url_prefix='/student')

@student.route('/dashboard')
@login_required
def dashboard():
    if current_user.is_teacher:
        flash('Access denied. This page is for students only.')
        return redirect(url_for('teacher.dashboard'))
    
    # Get all assignments
    assignments = Assignment.query.all()
    
    # Get student's submissions
    submissions = Submission.query.filter_by(student_id=current_user.id).all()
    submission_dict = {s.assignment_id: s for s in submissions}
    
    return render_template('student/dashboard.html', 
                          assignments=assignments, 
                          submissions=submission_dict,
                          now=datetime.datetime.now())

@student.route('/upload/<int:assignment_id>', methods=['GET', 'POST'])
@login_required
def upload(assignment_id):
    if current_user.is_teacher:
        flash('Access denied. This page is for students only.')
        return redirect(url_for('teacher.dashboard'))
    
    assignment = Assignment.query.get_or_404(assignment_id)
    
    # Check if already submitted
    existing_submission = Submission.query.filter_by(
        student_id=current_user.id, 
        assignment_id=assignment_id
    ).first()
    
    if existing_submission:
        flash('You have already submitted this assignment.')
        return redirect(url_for('student.dashboard'))
    
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No file selected')
            return redirect(request.url)
        
        if file and file.filename.endswith('.c'):
            filename = secure_filename(file.filename)
            # Create a unique filename to avoid conflicts
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"{current_user.username}_{assignment_id}_{timestamp}_{filename}"
            file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Create submission record
            submission = Submission(
                filename=file.filename,
                file_path=file_path,
                student_id=current_user.id,
                assignment_id=assignment_id
            )
            db.session.add(submission)
            db.session.commit()
            
            flash('Assignment submitted successfully!')
            return redirect(url_for('student.dashboard'))
        else:
            flash('Please upload a .c file')
    
    return render_template('student/upload.html', assignment=assignment)

@student.route('/submissions')
@login_required
def submissions():
    if current_user.is_teacher:
        flash('Access denied. This page is for students only.')
        return redirect(url_for('teacher.dashboard'))
    
    submissions = Submission.query.filter_by(student_id=current_user.id).all()
    return render_template('student/submissions.html', submissions=submissions)