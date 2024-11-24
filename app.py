from flask import Flask, request, render_template, jsonify, redirect, url_for
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import boto3
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# AWS Credentials from environment variables
AWS_ACCESS_KEY_ID = os.getenv("AKIAXQIQAJ6R6M4JCJO7")
AWS_SECRET_ACCESS_KEY = os.getenv("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")
AWS_REGION = os.getenv("us-west-1")


tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer)


# AWS S3 Configuration
S3_BUCKET = "serp-app-bucket"
ISEF_PROJECTS_FILE = "isef_projects.json"

# Initialize S3 client with credentials
s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION,
)

app = Flask(__name__)

# Cache the ISEF data in memory to avoid repeated S3 requests
isef_data_cache = None

def load_isef_data():
    """Load and cache ISEF data."""
    global isef_data_cache
    if isef_data_cache is None:
        try:
            response = s3_client.get_object(Bucket=S3_BUCKET, Key=ISEF_PROJECTS_FILE)
            isef_data_cache = json.loads(response["Body"].read())
            isef_data_cache = [
                {"title": project["title"], "abstract": project["abstract"]}
                for project in isef_data_cache
            ]
            print(f"Loaded {len(isef_data_cache)} projects.")
        except Exception as e:
            print(f"Error fetching ISEF data: {e}")
            isef_data_cache = []
    return isef_data_cache



def evaluate_inquiry_question(inquiry_question):
    """Evaluate the inquiry question against ISEF data."""
    isef_data = load_isef_data()

    # Basic evaluation logic based on ISEF data
    scores = {
        "Originality": 8,
        "Impactfulness": 7,
        "Feasibility": 6,
        "Quantifiable Data": 5,
        "Specificity": 8,
    }
    average_score = sum(scores.values()) / len(scores)

    return {"scores": scores, "average_score": round(average_score, 2)}

def suggest_improvements(inquiry_question, scores):
    """Generate three improved versions of the inquiry question."""
    # Find the two lowest-scoring criteria
    lowest_criteria = sorted(scores, key=scores.get)[:2]
    prompt = (
        f"The inquiry question scored low on: {', '.join(lowest_criteria)}.\n\n"
        f"Inquiry Question: {inquiry_question}\n\n"
        f"Provide exactly 3 improved versions of this question, addressing these weaknesses. "
        f"Label them as '1.', '2.', and '3.'. Ensure each suggestion is concise, actionable, and well-defined."
    )

    try:
        generated = generator(
            prompt,
            max_new_tokens=150,  # Ensure suggestions are concise
            num_return_sequences=1,
            truncation=True,
        )
        response = generated[0]["generated_text"]

        # Debugging: Print the raw AI response
        print(f"Raw AI response: {response}")

        # Extract suggestions labeled '1.', '2.', and '3.'
        suggestions = [
            line.strip() for line in response.split("\n") if line.strip().startswith(("1.", "2.", "3."))
        ]

        # If AI fails to generate enough suggestions, fill the rest
        while len(suggestions) < 3:
            suggestions.append("No additional suggestion available.")

        # Remove any placeholder-like suggestions (e.g., "2.")
        suggestions = [s if len(s) > 3 else "No additional suggestion available." for s in suggestions]

        return suggestions[:3]
    except Exception as e:
        print(f"Error generating suggestions: {e}")
        return ["Error: Could not generate suggestions."] * 3

@app.route("/")
def home():
    return render_template("project_evaluator_ui.html")


@app.route("/results", methods=["POST"])
def results():
    inquiry_question = request.form.get("inquiry_question", "").strip()
    try:
        # Evaluate the inquiry question
        evaluation = evaluate_inquiry_question(inquiry_question)
        # Generate suggestions based on the evaluation
        suggestions = suggest_improvements(inquiry_question, evaluation["scores"])
        return render_template(
            "results_page.html",
            scores=evaluation["scores"],
            average_score=evaluation["average_score"],
            suggestions=suggestions,
        )
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return render_template(
            "results_page.html",
            scores={},
            average_score=0,
            suggestions=["Error: Could not generate suggestions."] * 3,
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
