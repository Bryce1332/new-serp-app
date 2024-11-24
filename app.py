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
    aws_access_key_id="YOUR_AWS_ACCESS_KEY_ID",
    aws_secret_access_key="YOUR_AWS_SECRET_ACCESS_KEY"
)

# Use a smaller Hugging Face model like DistilGPT-2
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
tokenizer.pad_token = tokenizer.eos_token  # Ensure padding token is set

generator = pipeline(
    "text-generation",
    model="distilgpt2",
    tokenizer=tokenizer,
    truncation=True,
    padding=True
)

app = Flask(__name__)
current_evaluation = {}

def fetch_isef_data(query):
    """
    Fetch relevant ISEF database entries for a project idea to assist in evaluation.
    """
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

def evaluate_inquiry_question(inquiry_question):
    """
    Evaluate an inquiry question based on the ISEF database and provide scores for each criterion.
    """
    relevant_projects = fetch_isef_data(inquiry_question)
    print(f"Relevant ISEF projects: {len(relevant_projects)} found.")

    if not relevant_projects:
        return {
            "scores": {
                "Originality": 5,
                "Impactfulness": 5,
                "Feasibility": 5,
                "Quantifiable Data": 5,
                "Specificity": 5,
            },
            "average_score": 5.0
        }

    scores = {
        "Originality": min(10, len(relevant_projects)),  # More projects = less original
        "Impactfulness": 10 if len(relevant_projects) >= 5 else 7,
        "Feasibility": 10 if len(relevant_projects) >= 3 else 6,
        "Quantifiable Data": 8 if len(relevant_projects) >= 2 else 5,
        "Specificity": 10 if len(inquiry_question.split()) > 10 else 7,
    }

    average_score = sum(scores.values()) / len(scores)

    evaluation = {
        "scores": scores,
        "average_score": round(average_score, 2)
    }
    print(f"Generated evaluation: {evaluation}")
    return evaluation

def suggest_improvements(inquiry_question, scores):
    """
    Use AI to suggest three edited versions of the inquiry question addressing the lowest-scoring criteria.
    """
    sorted_criteria = sorted(scores.items(), key=lambda x: x[1])[:2]
    lowest_criteria = [criterion for criterion, _ in sorted_criteria]

    prompt = (
        f"The following research inquiry question scored low on the criteria {', '.join(lowest_criteria)}:\n\n"
        f"Inquiry Question: {inquiry_question}\n\n"
        f"Provide 3 improved versions of the question to address these weaknesses."
    )

    try:
        generated = generator(
            prompt,
            max_new_tokens=150,
            num_return_sequences=1,
            truncation=True
        )
        suggestions = generated[0]["generated_text"].strip().split("\n")[:3]
        print("Generated suggestions:", suggestions)
        return suggestions
    except Exception as e:
        print(f"Error generating suggestions: {e}")
        return ["Error: Could not generate suggestions."]

@app.route("/")
def home():
    return render_template("project_evaluator_ui.html")

@app.route("/results", methods=["POST"])
def results():
    global current_evaluation
    inquiry_question = request.form.get("inquiry_question", "")

    print(f"Received inquiry question: {inquiry_question}")

    try:
        evaluation = evaluate_inquiry_question(inquiry_question)
        suggestions = suggest_improvements(inquiry_question, evaluation["scores"])
        current_evaluation = {
            "scores": evaluation["scores"],
            "average_score": evaluation["average_score"],
            "suggestions": suggestions
        }
    except Exception as e:
        print(f"Error during evaluation: {e}")
        current_evaluation = {"error": f"Error: {e}"}

    return redirect(url_for("show_results"))

@app.route("/results-data", methods=["GET"])
def results_data():
    global current_evaluation
    print("Request received at /results-data")
    if "scores" not in current_evaluation:
        print("Error: No evaluation data available.")
        return jsonify({"error": "Failed to generate evaluation."}), 400
    print(f"Returning evaluation: {current_evaluation}")
    return jsonify(current_evaluation)

@app.route("/results")
def show_results():
    return render_template("results_page.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
