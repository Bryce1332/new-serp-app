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

def truncate_prompt(prompt, max_tokens=1500):
    """Truncate the prompt to fit within the model's token limit."""
    if len(prompt.split()) > max_tokens:
        print("Prompt exceeds token limit. Truncating...")
        return " ".join(prompt.split()[:max_tokens])
    return prompt

def evaluate_project_idea(title, description, objectives, methods, feasibility, impact, pathway):
    # Construct the prompt
    prompt = (
        f"Evaluate the following SERP project idea:\n\n"
        f"Title: {title}\n\nDescription: {description}\n\n"
        f"Objectives: {', '.join(objectives)}\n\nMethods: {', '.join(methods)}\n\n"
        f"Feasibility: {feasibility}\n\nImpact: {impact}\n\n"
        f"Pathway: {pathway}\n\nProvide detailed evaluation."
    )

    # Truncate the prompt to fit within safe limits
    prompt = truncate_prompt(prompt, max_tokens=1500)

    # Debug tokenized input length
    tokenized_input = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)
    print(f"Tokenized input length: {tokenized_input['input_ids'].shape[1]} tokens")

    # Fallback truncation if tokenized length exceeds safe threshold
    if tokenized_input["input_ids"].shape[1] > 1800:
        print("Tokenized input exceeds safe limit. Truncating further...")
        prompt = " ".join(prompt.split()[:1200])

    # Generate evaluation
    try:
        generated = generator(
            prompt,
            max_new_tokens=150,  # Limit generated tokens
            num_return_sequences=1,
            truncation=True
        )
        evaluation = generated[0]["generated_text"]
        print("Generated evaluation:", evaluation)
        return evaluation
    except RuntimeError as e:
        if "size of tensor" in str(e):
            print("Tensor size mismatch error:", e)
            return "Error: The evaluation prompt is too long or caused an internal model error."
        else:
            print(f"Error generating evaluation: {e}")
            return f"Error: {e}"
    except Exception as e:
        print(f"Unexpected error during evaluation: {e}")
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
