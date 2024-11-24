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

# Use a smaller model to reduce memory usage
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
tokenizer.pad_token = tokenizer.eos_token
generator = pipeline("text-generation", model="EleutherAI/gpt-neo-125M", tokenizer=tokenizer)

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
    """
    Critically evaluate the inquiry question against the ISEF database abstracts.
    """
    # Match inquiry question against abstracts
    relevant_projects = [
        proj for proj in ISEF_PROJECTS if inquiry_question.lower() in proj["abstract"].lower()
    ]

    # Criteria scoring logic
    originality_score = 10 - min(len(relevant_projects), 10)  # More matches = lower originality
    impact_score = 5 if len(inquiry_question.split()) <= 8 else 7  # Penalize very short questions
    feasibility_score = (
        4 if len(relevant_projects) >= 8 else 6 if len(relevant_projects) >= 5 else 8
    )  # Fewer matches = more feasible
    quantifiable_data_score = (
        5 if "measure" not in inquiry_question.lower() else 8
    )  # Reward measurable language
    specificity_score = (
        4 if len(inquiry_question.split()) <= 10 else 6 if len(inquiry_question.split()) <= 15 else 8
    )  # Penalize vague or overly broad questions

    # Aggregate scores
    scores = {
        "Originality": originality_score,
        "Impactfulness": impact_score,
        "Feasibility": feasibility_score,
        "Quantifiable Data": quantifiable_data_score,
        "Specificity": specificity_score,
    }
    average_score = sum(scores.values()) / len(scores)

    print(f"Scores: {scores}")
    print(f"Average Score: {average_score}")
    return {"scores": scores, "average_score": average_score}

def suggest_improvements(inquiry_question, scores):
    """
    Generate three improved versions of the inquiry question addressing the lowest-scoring criteria.
    """
    lowest_criteria = sorted(scores, key=scores.get)[:2]
    prompt = (
        f"The inquiry question scored low on: {', '.join(lowest_criteria)}.\n\n"
        f"Inquiry Question: {inquiry_question}\n\n"
        f"Provide three improved versions of this question to address these weaknesses. "
        f"Each version must be labeled as '1.', '2.', and '3.' and should focus on improving {lowest_criteria[0]} and/or {lowest_criteria[1]}.\n"
        f"Keep each suggestion concise and actionable."
    )

    try:
        # Generate response from AI
        generated = generator(
            prompt,
            max_new_tokens=150,  # Limit response length
            num_return_sequences=1,
            truncation=True
        )
        response = generated[0]["generated_text"]
        print("Raw AI response:", response)

        # Extract suggestions from response
        suggestions = [
            line.strip() for line in response.split("\n") if line.strip().startswith(("1.", "2.", "3."))
        ]

        # Ensure there are exactly three suggestions
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
    inquiry_question = request.form.get("inquiry_question", "").strip()

    print(f"Received inquiry question: {inquiry_question}")

    try:
        # Fetch scores and average score
        evaluation = evaluate_inquiry_question(inquiry_question)
        print(f"Evaluation results: {evaluation}")

        # Generate suggestions based on scores
        suggestions = suggest_improvements(inquiry_question, evaluation["scores"])
        print(f"Generated suggestions: {suggestions}")

        # Populate current_evaluation
        current_evaluation = {
            "scores": evaluation["scores"],
            "average_score": evaluation["average_score"],
            "suggestions": suggestions
        }
    except Exception as e:
        print(f"Error during evaluation: {e}")
        current_evaluation = {
            "scores": {},
            "average_score": 0,
            "suggestions": ["Error: Could not generate suggestions."] * 3
        }

    return redirect(url_for("show_results"))



@app.route("/results-data", methods=["GET"])
def results_data():
    global current_evaluation
    print("Request received at /results-data")
    if not current_evaluation or "scores" not in current_evaluation:
        print("Error: No evaluation data available.")
        return jsonify({"error": "No evaluation data available."}), 400
    print(f"Returning evaluation: {current_evaluation}")
    return jsonify(current_evaluation)


@app.route("/results")
def show_results():
    return render_template("results_page.html", evaluation=current_evaluation)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
