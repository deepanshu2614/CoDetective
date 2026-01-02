@app.route('/api/ai-detection', methods=['POST'])
def ai_detection():
    """Endpoint to analyze student code against AI-generated code"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not file.filename.lower().endswith('.c'):
            return jsonify({'error': 'Invalid file type. Only .c files are accepted.'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        file.save(file_path)

        # Read student's code
        with open(file_path, 'r', encoding='utf-8') as f:
            student_code = f.read()

        # Ensure question is uploaded first
        question = session.get('question')
        if not question:
            return jsonify({'error': 'No question uploaded. Please upload a question first.'}), 400

        # --- STEP 1: Try Local Model ---
        try:
            local_confidence, is_ai_local = ai_detector.predict(student_code)
            local_confidence = float(local_confidence)
            is_ai_local = bool(is_ai_local)
        except Exception as e:
            logger.warning(f"Local model failed: {e}")
            local_confidence, is_ai_local = 0.0, False  # fallback to API

        # --- STEP 2: Decide model route ---
        use_local_model = local_confidence >= getattr(ai_detector, "threshold", 0.7)

        if use_local_model:
            logger.info(f"Using local model: confidence={local_confidence}")
            result = {
                'ai_probability': local_confidence,
                'is_ai_generated': is_ai_local,
                'detection_method': 'local_model',
                'features': {
                    'token_similarity': f"{local_confidence * 100:.1f}%",
                    'structure_similarity': f"{local_confidence * 100:.1f}%",
                    'variable_naming': f"{local_confidence * 100:.1f}%",
                    'logic_flow': f"{local_confidence * 100:.1f}%",
                    'comments': f"{local_confidence * 100:.1f}%"
                },
                'reference_solution': 'Local model detection - no reference solution generated.'
            }

        # --- STEP 3: Fallback to API ---
        else:
            logger.info("Local model not confident. Using OpenAI API fallback.")
            api_result = detect_ai_similarity(student_code, question)
            
            if not api_result or 'error' in api_result:
                return jsonify({
                    'error': api_result.get('error', 'OpenAI API failed to respond')
                }), 500
            
            similarity = api_result['similarity']
            result = {
                'ai_probability': float(similarity['overall']),
                'is_ai_generated': bool(similarity['overall'] >= 0.7),
                'detection_method': 'openai_api',
                'features': {
                    'token_similarity': f"{float(similarity['token']) * 100:.1f}%",
                    'structure_similarity': f"{float(similarity['structure']) * 100:.1f}%",
                    'variable_naming': f"{float(similarity['variable']) * 100:.1f}%",
                    'logic_flow': f"{float(similarity['logic']) * 100:.1f}%",
                    'comments': f"{float(similarity['comments']) * 100:.1f}%"
                },
                'reference_solution': api_result['reference_solution']
            }

        # --- STEP 4: Add conclusion ---
        result['conclusion'] = (
            "The code appears to be AI-generated."
            if result['is_ai_generated']
            else "The code appears to be human-written."
        )

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error in ai_detection: {str(e)}", exc_info=True)
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500