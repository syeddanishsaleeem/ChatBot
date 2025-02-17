import os
from flask import Flask, request, jsonify,render_template
import uuid
from werkzeug.utils import secure_filename
from agent import service_retrieve,grade_documents,generate,decide_to_generate,grade_generation_v_documents_and_question,GraphState
from langgraph.graph import END, StateGraph
from pprint import pprint
from embedding import embedd_and_store

app = Flask(__name__)

workflow = StateGraph(GraphState)
workflow.add_node("service_retrieve", service_retrieve)  # retrieve
workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("generate", generate)  # generatae

workflow.set_entry_point("service_retrieve")
workflow.add_edge("service_retrieve", "grade_documents")

workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "end": END,
        "generate": "generate",
    },
)

workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": END,
    },
)
# Compile
# app = workflow.compile()
app_graph = workflow.compile()

# Directory to save uploaded images
UPLOAD_FOLDER = './static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed extensions for uploaded files
ALLOWED_EXTENSIONS = {'pdf'}

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    """Check if the file has a valid extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Global Variable
qdrnt_retriever = None

@app.route('/')
def dashboard():
    return render_template('dashboard.html')


@app.route("/message", methods=["GET","POST"])
def messages():
    if request.method == "POST":
        """Handles user queries and runs the workflow."""

        data = request.json
        question = data.get("question")

        if not question:
            return jsonify({"error": "Question is required"}), 400
        # collection_name = "demo_collection1"
        global qdrnt_retriever
        # question = "What is AI?"
        inputs = {"question": question, "retrieval_model": "","qdrnt_retriever":qdrnt_retriever}
        result_data = {}

        for output in app_graph.stream(inputs):
            for key, value in output.items():
                pprint(f"Finished running: {key}:")
                result_data[key] = value  # Store results
                # pprint(f"Finished running: {result_data}:")
        pprint(value["generation"])

        return jsonify({"response":value["generation"] })
    else:
        return render_template('messages.html')



@app.route('/upload', methods=['POST'])
def upload():
    global qdrnt_retriever  # Declare it as global inside the function

    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file and allowed_file(file.filename):
        # Generate a random UUID filename with .pdf extension
        unique_id = uuid.uuid4()
        random_filename = f"{unique_id}.pdf"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(random_filename))
        file.save(file_path)
        qdrnt_retriever = embedd_and_store(file_path)

        return jsonify({'message': 'File uploaded successfully', 'filename': random_filename}), 200
    else:
        return jsonify({'message': "Only PDF files are allowed"}), 400

if __name__ == '__main__':
    # app.run(debug=True)
    app.run(host="0.0.0.0", port=5000)
