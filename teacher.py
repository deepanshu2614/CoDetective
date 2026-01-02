from flask import Blueprint, render_template, request, redirect, url_for, flash, current_app, jsonify
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
from database import db, Assignment, Submission, SimilarityResult
import os
import json
import datetime

teacher = Blueprint('teacher', __name__, url_prefix='/teacher')

@teacher.route('/dashboard')
@login_required
def dashboard():
    if not current_user.is_teacher:
        flash('Access denied. This page is for teachers only.')
        return redirect(url_for('student.dashboard'))
    
    assignments = Assignment.query.filter_by(teacher_id=current_user.id).all()

    # FIXED: safe calculations
    total_submissions = sum(len(a.submissions) for a in assignments)
    assignments_with_submissions = sum(1 for a in assignments if len(a.submissions) > 0)

    return render_template(
        'teacher/dashboard.html',
        assignments=assignments,
        total_submissions=total_submissions,
        assignments_with_submissions=assignments_with_submissions
    )

@teacher.route('/assignments')
@login_required
def assignments():
    if not current_user.is_teacher:
        flash('Access denied. This page is for teachers only.')
        return redirect(url_for('student.dashboard'))
    
    assignments = Assignment.query.filter_by(teacher_id=current_user.id).all()
    return render_template('teacher/assignments.html', assignments=assignments)

@teacher.route('/create_assignment', methods=['GET', 'POST'])
@login_required
def create_assignment():
    if not current_user.is_teacher:
        flash('Access denied. This page is for teachers only.')
        return redirect(url_for('student.dashboard'))
    
    if request.method == 'POST':
        title = request.form.get('title')
        description = request.form.get('description')
        deadline_str = request.form.get('deadline')
        
        if not title or not description:
            flash('Title and description are required')
            return redirect(request.url)
        
        deadline = None
        if deadline_str:
            try:
                deadline = datetime.datetime.strptime(deadline_str, '%Y-%m-%d')
            except ValueError:
                flash('Invalid deadline format')
                return redirect(request.url)
        
        # Handle question file upload
        question_file = None
        if 'question_file' in request.files:
            file = request.files['question_file']
            if file.filename:
                filename = secure_filename(file.filename)
                # Create a unique filename
                timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                filename = f"question_{timestamp}_{filename}"
                file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                question_file = file_path
        
        # Create assignment
        assignment = Assignment(
            title=title,
            description=description,
            question_file=question_file,
            deadline=deadline,
            teacher_id=current_user.id
        )
        db.session.add(assignment)
        db.session.commit()
        
        flash('Assignment created successfully!')
        return redirect(url_for('teacher.assignments'))
    
    return render_template('teacher/create_assignment.html')

@teacher.route('/assignment/<int:assignment_id>')
@login_required
def view_assignment(assignment_id):
    if not current_user.is_teacher:
        flash('Access denied. This page is for teachers only.')
        return redirect(url_for('student.dashboard'))
    
    assignment = Assignment.query.get_or_404(assignment_id)
    
    # Check if teacher owns this assignment
    if assignment.teacher_id != current_user.id:
        flash('Access denied')
        return redirect(url_for('teacher.dashboard'))
    
    submissions = Submission.query.filter_by(assignment_id=assignment_id).all()
    return render_template('teacher/view_assignment.html', assignment=assignment, submissions=submissions)

@teacher.route('/student_similarity/<int:assignment_id>')
@login_required
def student_similarity(assignment_id):
    if not current_user.is_teacher:
        flash('Access denied. This page is for teachers only.')
        return redirect(url_for('student.dashboard'))
    
    assignment = Assignment.query.get_or_404(assignment_id)
    
    # Check if teacher owns this assignment
    if assignment.teacher_id != current_user.id:
        flash('Access denied')
        return redirect(url_for('teacher.dashboard'))
    
    submissions = Submission.query.filter_by(assignment_id=assignment_id).all()
    return render_template('teacher/student_similarity.html', assignment=assignment, submissions=submissions)

@teacher.route('/ai_similarity/<int:assignment_id>')
@login_required
def ai_similarity(assignment_id):
    if not current_user.is_teacher:
        flash('Access denied. This page is for teachers only.')
        return redirect(url_for('student.dashboard'))
    
    assignment = Assignment.query.get_or_404(assignment_id)
    
    # Check if teacher owns this assignment
    if assignment.teacher_id != current_user.id:
        flash('Access denied')
        return redirect(url_for('teacher.dashboard'))
    
    submissions = Submission.query.filter_by(assignment_id=assignment_id).all()
    return render_template('teacher/ai_similarity.html', assignment=assignment, submissions=submissions)

@teacher.route('/get_question/<int:assignment_id>')
@login_required
def get_question(assignment_id):
    if not current_user.is_teacher:
        return jsonify({'error': 'Access denied'}), 403
    
    assignment = Assignment.query.get_or_404(assignment_id)
    
    # Check if teacher owns this assignment
    if assignment.teacher_id != current_user.id:
        return jsonify({'error': 'Access denied'}), 403
    
    # Read question file
    question_text = assignment.description
    if assignment.question_file and os.path.exists(assignment.question_file):
        with open(assignment.question_file, 'r', encoding='utf-8') as f:
            question_text = f.read()
    
    return jsonify({'question': question_text})

@teacher.route('/save_similarity_result', methods=['POST'])
@login_required
def save_similarity_result():
    if not current_user.is_teacher:
        return jsonify({'error': 'Access denied'}), 403
    
    data = request.get_json()
    assignment_id = data.get('assignment_id')
    submission_id = data.get('submission_id')
    result_type = data.get('result_type')
    result_data = json.dumps(data.get('result_data'))
    
    # Verify teacher owns the assignment
    assignment = Assignment.query.get_or_404(assignment_id)
    if assignment.teacher_id != current_user.id:
        return jsonify({'error': 'Access denied'}), 403
    
    # Create similarity result
    similarity_result = SimilarityResult(
        result_type=result_type,
        result_data=result_data,
        assignment_id=assignment_id,
        submission_id=submission_id
    )
    db.session.add(similarity_result)
    db.session.commit()
    
    return jsonify({'success': True})

@teacher.route('/assignment/<int:assignment_id>/delete', methods=['GET'])
@login_required
def delete_assignment(assignment_id):
    assignment = Assignment.query.get_or_404(assignment_id)

    from database import SimilarityResult
    SimilarityResult.query.filter_by(assignment_id=assignment_id).delete()

    # delete submissions
    for sub in assignment.submissions:
        db.session.delete(sub)

    db.session.delete(assignment)
    db.session.commit()

    return redirect(url_for('teacher.assignments'))
