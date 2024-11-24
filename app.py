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
    # Construct the prompt
    prompt = (
        f"Evaluate the following SERP project idea:\n\n"
        f"Title: {title}\n\nDescription: {description}\n\n"
        f"Objectives: {', '.join(objectives)}\n\nMethods: {', '.join(methods)}\n\n"
        f"Feasibility: {feasibility}\n\nImpact: {impact}\n\n"
        f"Pathway: {pathway}\n\nProvide detailed evaluation."
    )

    # Truncate prompt to fit within model limits
    max_prompt_tokens = 1500  # Ensure room for model's response
    if len(prompt.split()) > max_prompt_tokens:
        print("Prompt exceeds token limit. Truncating...")
        prompt = " ".join(prompt.split()[:max_prompt_tokens])

    # Generate evaluation
    try:
        print(f"Prompt size: {len(prompt.split())} tokens")
        generated = generator(
            prompt,
            max_new_tokens=200,  # Generate up to 200 tokens for response
            num_return_sequences=1,
            truncation=True
        )
        evaluation = generated[0]["generated_text"]
        print("Generated evaluation:", evaluation)
        return evaluation
    except Exception as e:
        print(f"Error generating evaluation: {e}")
        return f"Error: {e}"


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

    print(f"Received form data: Title={title}, Description={description}, Objectives={objectives}, Methods={methods}, Feasibility={feasibility}, Impact={impact}, Pathway={pathway}")

    try:
        evaluation = evaluate_project_idea(
            title, description, objectives, methods, feasibility, impact, pathway
        )
        current_evaluation = {"evaluation": evaluation}
        print(f"Generated evaluation: {current_evaluation}")
    except Exception as e:
        print(f"Error during evaluation: {e}")
        current_evaluation = {"evaluation": f"Error: {e}"}

    return redirect(url_for("show_results"))



@app.route("/results-data", methods=["GET"])
def results_data():
    global current_evaluation
    print("Request received at /results-data")
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
