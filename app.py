import boto3
from flask import Flask, request, jsonify, render_template, redirect, url_for
from transformers import pipeline, AutoTokenizer
import json

# AWS S3 setup
S3_BUCKET = "serp-app-bucket"
ISEF_PROJECTS_FILE = "isef-projects.json"

# Configure AWS credentials
s3_client = boto3.client(
    "s3",
    aws_access_key_id="AKIAXQIQAJ6R6M4JCJO7",
    aws_secret_access_key="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"
)

# Use a smaller Hugging Face model
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
tokenizer.pad_token = tokenizer.eos_token  # Set padding token
generator = pipeline(
    "text-generation",
    model="EleutherAI/gpt-neo-125M",
    tokenizer=tokenizer,
    truncation=True,
    padding="max_length"
)

app = Flask(__name__)
current_evaluation = {}

def fetch_isef_data(query):
    try:
        response = s3_client.get_object(Bucket=S3_BUCKET, Key=ISEF_PROJECTS_FILE)
        isef_projects = json.loads(response["Body"].read())
        relevant_projects = [
            project for project in isef_projects if query.lower() in project["title"].lower()
        ]
        return relevant_projects
    except Exception as e:
        print(f"Error fetching ISEF data: {e}")
        return []

def evaluate_project_idea(title, description, objectives, methods, feasibility, impact, pathway):
    prompt = (
        f"Evaluate the following SERP project idea:\n\n"
        f"Title: {title}\n\nDescription: {description}\n\n"
        f"Objectives: {', '.join(objectives)}\n\nMethods: {', '.join(methods)}\n\n"
        f"Feasibility: {feasibility}\n\nImpact: {impact}\n\n"
        f"Pathway: {pathway}\n\nProvide detailed evaluation."
    )
    try:
        # Truncate input to prevent model overload
        max_input_tokens = 1024
        if len(prompt.split()) > max_input_tokens:
            prompt = " ".join(prompt.split()[:max_input_tokens])

        generated = generator(
            prompt,
            max_new_tokens=50,  # Generate up to 50 tokens
            num_return_sequences=1,
            truncation=True
        )
        return generated[0]["generated_text"]
    except MemoryError:
        print("MemoryError: The model ran out of resources.")
        return "Error: The evaluation could not be completed due to resource constraints."
    except Exception as e:
        print(f"Error generating evaluation: {e}")
        return "Error: The evaluation could not be completed. Please try again."

@app.route("/")
def home():
    return render_template("project_evaluator_ui.html")

@app.route("/results", methods=["POST"])
def results():
    global current_evaluation
    title = request.form.get("title", "")
    description = request.form.get("description", "")
    objectives = request.form.get("objectives", "").split(",")
    methods = request.form.get("methods", "").split(",")
    feasibility = request.form.get("feasibility", "")
    impact = request.form.get("impact", "")
    pathway = request.form.get("pathway", "")
    current_evaluation = {
        "evaluation": evaluate_project_idea(
            title, description, objectives, methods, feasibility, impact, pathway
        )
    }
    return redirect(url_for("show_results"))

@app.route("/results-data", methods=["GET"])
def results_data():
    global current_evaluation
    if "evaluation" not in current_evaluation:
        print("Error: No evaluation data available.")
        return jsonify({"evaluation": "Failed to generate evaluation."}), 400
    print(f"Returning evaluation: {current_evaluation}")
    return jsonify(current_evaluation)

@app.route("/results")
def show_results():
    return render_template("results_page.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
