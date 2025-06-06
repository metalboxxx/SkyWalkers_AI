from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
from Agents.gen_test_cases import generate_test_cases_from_requirements

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/generate-test-cases', methods=['POST'])
def generate_test_cases():
    try:
        printf("PDF sended")
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
        test_cases = generate_test_cases_from_requirements(pdf_data)
        
        return jsonify(test_cases)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)