import boto3
from flask import Flask, request, jsonify, render_template, redirect, url_for
from transformers import pipeline, AutoTokenizer
import json

# AWS S3 setup
S3_BUCKET = "serp-app-bucket"
ISEF_PROJECTS_FILE = "isef_projects.json"

s3_client = boto3.client(
    "s3",
    aws_access_key_id="AKIAXQIQAJ6R6M4JCJO7",
    aws_secret_access_key="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"
)

# Configure AI model
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
tokenizer.pad_token = tokenizer.eos_token
generator = pipeline("text-generation", model="distilgpt2", tokenizer=tokenizer)

app = Flask(__name__)
current_evaluation = {}
ISEF_PROJECTS = []

# Fetch the ISEF data from S3
def fetch_isef_data_from_s3():
    global ISEF_PROJECTS
    try:
        print("Fetching ISEF data from S3...")
        response = s3_client.get_object(Bucket=S3_BUCKET, Key=ISEF_PROJECTS_FILE)
        data = json.loads(response["Body"].read())
        ISEF_PROJECTS = [{"title": proj["title"], "abstract": proj["abstract"]} for proj in data]
        print(f"Loaded {len(ISEF_PROJECTS)} projects from S3.")
    except Exception as e:
        print(f"Error fetching ISEF data: {e}")
        ISEF_PROJECTS = []

# Evaluate Inquiry Question
def evaluate_inquiry_question(inquiry_question):
    relevant_projects = [
        proj for proj in ISEF_PROJECTS if inquiry_question.lower() in proj["abstract"].lower()
    ]
    scores = {
        "Originality": 10 - len(relevant_projects) if len(relevant_projects) < 10 else 1,
        "Impactfulness": 8 if len(relevant_projects) >= 5 else 6,
        "Feasibility": 7 if len(relevant_projects) >= 3 else 5,
        "Quantifiable Data": 8 if len(relevant_projects) >= 2 else 6,
        "Specificity": 10 if len(inquiry_question.split()) > 10 else 7,
    }
    average_score = sum(scores.values()) / len(scores)
    return {"scores": scores, "average_score": average_score}

def suggest_improvements(inquiry_question, scores):
    """
    Use AI to suggest three edited versions of the inquiry question addressing the lowest-scoring criteria.
    """
    # Identify the two lowest-scoring criteria
    lowest_criteria = sorted(scores, key=scores.get)[:2]

    # Construct the prompt
    prompt = (
        f"The following inquiry question scored low on the criteria: {', '.join(lowest_criteria)}.\n\n"
        f"Inquiry Question: {inquiry_question}\n\n"
        f"Provide three improved versions of this question to address these weaknesses, clearly labeled as 1, 2, and 3."
    )

    try:
        # Generate AI response
        generated = generator(
            prompt,
            max_new_tokens=150,
            num_return_sequences=1,
            truncation=True
        )

        # Parse the AI response
        response = generated[0]["generated_text"]
        print("Raw AI response:", response)

        # Extract suggestions based on numbering (1., 2., 3.)
        suggestions = []
        for line in response.split("\n"):
            if line.strip().startswith(("1.", "2.", "3.")):
                suggestions.append(line.strip())

        # Ensure we return exactly three suggestions
        if len(suggestions) < 3:
            while len(suggestions) < 3:
                suggestions.append("No additional suggestion available.")

        print("Parsed suggestions:", suggestions)
        return suggestions[:3]
    except Exception as e:
        print(f"Error generating suggestions: {e}")
        return [f"Error: {e}"]


@app.route("/")
def home():
    return render_template("project_evaluator_ui.html")

@app.route("/results", methods=["POST"])
def results():
    global current_evaluation
    inquiry_question = request.form.get("inquiry_question", "")
    print(f"Received inquiry question: {inquiry_question}")

    # Fetch data and evaluate the question
    fetch_isef_data_from_s3()
    evaluation = evaluate_inquiry_question(inquiry_question)
    suggestions = suggest_improvements(inquiry_question, evaluation["scores"])

    current_evaluation = {
        "scores": evaluation["scores"],
        "average_score": evaluation["average_score"],
        "suggestions": suggestions
    }
    return redirect(url_for("show_results"))

@app.route("/results-data", methods=["GET"])
def results_data():
    global current_evaluation
    if not current_evaluation:
        return jsonify({"error": "No evaluation data available."}), 400
    return jsonify(current_evaluation)

@app.route("/results")
def show_results():
    return render_template("results_page.html", evaluation=current_evaluation)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
