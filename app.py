from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
from test_case_gen import gen_requirements_pdf_to_test_case

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/generate-test-cases', methods=['POST'])
def generate_test_cases():
    try:
        # Check if file is in the request
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        # Check if the file is a PDF
        if not file.filename.lower().endswith('.pdf'):
            return jsonify({'error': 'Only PDF files are supported'}), 400
        
        # Read and encode the PDF file
        pdf_data = base64.b64encode(file.read()).decode('utf-8')
        
        # Generate test cases
        test_cases = gen_requirements_pdf_to_test_case(pdf_data)
        
        return jsonify(test_cases)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)