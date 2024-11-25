import os
import re
from flask import Flask, request, jsonify, render_template, redirect, url_for
from transformers import pipeline, AutoTokenizer
import boto3
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# AWS S3 Configuration
S3_BUCKET = os.getenv("serp-app-bucket   ")
AWS_ACCESS_KEY = os.getenv("AKIAXQIQAJ6R6M4JCJO7")
AWS_SECRET_KEY = os.getenv("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")
ISEF_PROJECTS_FILE = "isef_projects.json"

# S3 Client
s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
)

# Hugging Face Model Configuration
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Ensure padding token is set
generator = pipeline(
    "text-generation",
    model="gpt2",
    tokenizer=tokenizer,
    truncation=True,
    padding=True,
)

app = Flask(__name__)
current_evaluation = {}

def fetch_isef_data():
    """Fetch and validate ISEF data from S3."""
    try:
        response = s3_client.get_object(Bucket=S3_BUCKET, Key=ISEF_PROJECTS_FILE)
        isef_projects = json.loads(response["Body"].read())
        
        # Validate the data: Ensure all projects have titles and abstracts
        valid_projects = [
            proj for proj in isef_projects
            if isinstance(proj.get("title"), str) and isinstance(proj.get("abstract"), str)
        ]
        
        print(f"Loaded {len(valid_projects)} valid projects from ISEF database.")
        return valid_projects
    except Exception as e:
        print(f"Error fetching ISEF data: {e}")
        return []


def evaluate_inquiry_question(inquiry_question):
    """Evaluate an inquiry question using AI for each criterion."""
    isef_projects = fetch_isef_data()
    if not isef_projects:
        return {"scores": {}, "average_score": 0, "reasons": {}}

    # Define the criteria and their prompts
    criteria_prompts = {
        "Originality": (
            f"Evaluate the originality of the following research inquiry question:\n\n"
            f"Inquiry Question: {inquiry_question}\n\n"
            f"Originality refers to how novel and unique the question is compared to common scientific research topics.\n"
            f"Score the originality on a scale of 1 to 10 and provide a brief reason for the score."
        ),
        "Impactfulness": (
            f"Evaluate the impactfulness of the following research inquiry question:\n\n"
            f"Inquiry Question: {inquiry_question}\n\n"
            f"Impactfulness refers to how significant the potential scientific, societal, or environmental impact of answering this question would be.\n"
            f"Score the impactfulness on a scale of 1 to 10 and provide a brief reason for the score."
        ),
        "Feasibility": (
            f"Evaluate the feasibility of the following research inquiry question:\n\n"
            f"Inquiry Question: {inquiry_question}\n\n"
            f"Feasibility refers to whether a high school student could realistically complete a project addressing this question within two months using commonly available resources.\n"
            f"Score the feasibility on a scale of 1 to 10 and provide a brief reason for the score."
        ),
        "Quantifiable Data": (
            f"Evaluate the ability of the following research inquiry question to generate quantifiable data:\n\n"
            f"Inquiry Question: {inquiry_question}\n\n"
            f"Quantifiable data refers to measurable outcomes or results that can be analyzed scientifically.\n"
            f"Score the question on a scale of 1 to 10 and provide a brief reason for the score."
        ),
        "Specificity": (
            f"Evaluate the specificity of the following research inquiry question:\n\n"
            f"Inquiry Question: {inquiry_question}\n\n"
            f"Specificity refers to how focused and well-defined the question is.\n"
            f"Score the specificity on a scale of 1 to 10 and provide a brief reason for the score."
        ),
    }

    scores = {}
    reasons = {}

    try:
        for criterion, prompt in criteria_prompts.items():
            # Generate AI response for each criterion
            response = generator(
                prompt,
                max_new_tokens=150,
                num_return_sequences=1,
                truncation=True,
            )
            ai_output = response[0]["generated_text"]

            # Extract the score and reason from the AI response
            score_match = re.search(r"\b(\d{1,2})\b", ai_output)
            reason_match = re.search(r"\b(?:Reason|Explanation|Because): (.+)", ai_output, re.IGNORECASE)

            score = int(score_match.group(1)) if score_match else 0
            reason = reason_match.group(1).strip() if reason_match else "No reason provided."

            # Store the score and reason
            scores[criterion] = min(max(score, 1), 10)  # Ensure score is between 1 and 10
            reasons[criterion] = reason

        # Calculate average score
        average_score = sum(scores.values()) / len(scores)

        return {"scores": scores, "average_score": average_score, "reasons": reasons}

    except Exception as e:
        print(f"Error during AI-driven evaluation: {e}")
        return {"scores": {}, "average_score": 0, "reasons": {}}

def suggest_improvements(inquiry_question, scores):
    """Generate three improved versions of the inquiry question."""
    # Find the two lowest-scoring criteria
    lowest_criteria = sorted(scores, key=scores.get)[:2]
    prompt = (
        f"The following research inquiry question needs improvement:\n\n"
        f"Inquiry Question: {inquiry_question}\n\n"
        f"Criteria scoring low: {', '.join(lowest_criteria)}\n\n"
        f"Instructions:\n"
        f"- Generate exactly 3 improved versions of the question.\n"
        f"- Each version must address the weaknesses in {', '.join(lowest_criteria)}.\n"
        f"- Be concise (less than 25 words).\n"
        f"- Use clear and actionable language.\n\n"
        f"Now, provide your 3 suggestions labeled as '1.', '2.', and '3.'."
    )

    try:
        generated = generator(
            prompt,
            max_new_tokens=150,
            num_return_sequences=1,
            truncation=True,
        )
        response = generated[0]["generated_text"]

        # Extract suggestions labeled '1.', '2.', and '3.'
        suggestions = [
            line.strip() for line in response.split("\n") if line.strip().startswith(("1.", "2.", "3."))
        ]

        # If AI fails to generate enough suggestions, fill the rest
        while len(suggestions) < 3:
            suggestions.append("No additional suggestion available.")

        return suggestions[:3]
    except Exception as e:
        print(f"Error generating suggestions: {e}")
        return ["Error: Could not generate suggestions."] * 3

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
        current_evaluation = {"evaluation": evaluation, "suggestions": suggestions}
        print(f"Scores: {evaluation['scores']}")
        print(f"Reasons: {evaluation['reasons']}")
        print(f"Average Score: {evaluation['average_score']}")
    except Exception as e:
        print(f"Error during evaluation: {e}")
        current_evaluation = {
            "evaluation": {"scores": {}, "average_score": 0, "reasons": {}},
            "suggestions": ["Error: Could not generate suggestions."] * 3,
        }

    return redirect(url_for("show_results"))

@app.route("/results-data", methods=["GET"])
def results_data():
    global current_evaluation
    if "evaluation" not in current_evaluation:
        return jsonify({"error": "No evaluation data available."}), 400
    return jsonify(current_evaluation)

@app.route("/results")
def show_results():
    return render_template("results_page.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
