import boto3
from flask import Flask, request, jsonify, render_template, redirect, url_for
from transformers import pipeline, AutoTokenizer
import json
import time

# AWS S3 setup
S3_BUCKET = "serp-app-bucket"
ISEF_PROJECTS_FILE = "isef-projects.json"

# Configure AWS credentials
s3_client = boto3.client(
    "s3",
    aws_access_key_id="AKIAXQIQAJ6R6M4JCJO7",
    aws_secret_access_key="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"
)

# Hugging Face model setup
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
tokenizer.pad_token = tokenizer.eos_token  # Set EOS token as padding token
generator = pipeline(
    "text-generation",
    model="EleutherAI/gpt-neo-1.3B",
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
        generated = generator(
            prompt,
            max_length=300,
            num_return_sequences=1,
            truncation=True
        )
        return generated[0]["generated_text"]
    except Exception as e:
        print(f"Error generating evaluation: {e}")
        return "An error occurred during evaluation. Please try again."

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

@app.route("/results-data")
def results_data():
    global current_evaluation
    if "evaluation" not in current_evaluation:
        return jsonify({"evaluation": "No evaluation data available"}), 400
    return jsonify(current_evaluation)

@app.route("/results")
def show_results():
    return render_template("results_page.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
