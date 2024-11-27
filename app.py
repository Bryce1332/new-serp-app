import os
import json
import gzip
from flask import Flask, request, jsonify, render_template, redirect, url_for
import openai
import boto3
from dotenv import load_dotenv
from difflib import get_close_matches

# Load environment variables
load_dotenv()

# AWS S3 Configuration
S3_BUCKET = os.getenv("S3_BUCKET")
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
ISEF_PROJECTS_FILE = "isef_projects.json.gz"  # Use compressed format

# S3 Client
s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
)

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)
current_evaluation = {}


def fetch_isef_data(query):
    """Fetch relevant ISEF database entries for a project idea."""
    if not query:
        print("Error: Query is required to fetch ISEF data.")
        return []

    try:
        bucket_name = os.getenv("S3_BUCKET")
        print(f"Using bucket: {bucket_name}")

        response = s3_client.get_object(Bucket=bucket_name, Key=ISEF_PROJECTS_FILE)
        isef_projects = json.loads(gzip.decompress(response["Body"].read()))
        print(f"Loaded {len(isef_projects)} projects.")

        # Improved matching logic using difflib
        titles = [project.get("title", "") for project in isef_projects]
        close_matches = get_close_matches(query, titles, n=5, cutoff=0.3)
        print(f"Close matches: {close_matches}")

        relevant_projects = [proj for proj in isef_projects if proj.get("title", "") in close_matches]

        # Truncate abstracts to reduce size
        for proj in relevant_projects:
            proj["abstract"] = proj.get("abstract", "")[:300] + "..."  # Limit abstract length to 300 characters

        print(f"Found {len(relevant_projects)} relevant projects.")
        return relevant_projects
    except Exception as e:
        print(f"Error fetching ISEF data: {e}")
        return []



def evaluate_project_idea(title, description, inquiry_question, pathway):
    """Evaluate a project idea based on various criteria using OpenAI."""
    isef_projects = fetch_isef_data(title)
    if not isef_projects:
        return "No relevant projects were found. Please refine your query or try again with a different title."

    # Limit the number of projects included in the summary
    isef_summary = "\n".join([
        f"Title: {proj['title']}\nAbstract: {proj['abstract']}"
        for proj in isef_projects[:5]  # Limit to top 5 projects
    ])

    # Construct evaluation prompt
    prompt = (
        f"Evaluate the following research project idea:\n\n"
        f"Title: {title}\nDescription: {description}\nInquiry Question: {inquiry_question}\n\n"
        f"Pathway: {pathway}\n\n"
        f"Here are similar projects from a database:\n{isef_summary}\n\n"
        f"Provide a score (1-10) for the following criteria:\n"
        f"1. Originality\n2. Impactfulness\n3. Feasibility\n4. Quantifiable Data\n5. Specificity\n\n"
        f"Provide a reason for each score. Finally, generate three improved versions of the inquiry question."
    )

    try:
        print("Sending prompt to OpenAI API:")
        print(prompt)  # Debugging: Print the full prompt to verify correctness

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "system", "content": "You are an AI project evaluator."},
                      {"role": "user", "content": prompt}],
            max_tokens=700,  # Limit token usage
            temperature=0.7,
        )
        print("OpenAI API response received.")
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Error generating evaluation: {e}")
        return "Error: Could not generate evaluation."



@app.route("/")
def home():
    return render_template("project_evaluator_ui.html")


@app.route("/results", methods=["POST"])
def results():
    global current_evaluation
    title = request.form.get("title", "").strip()
    description = request.form.get("description", "").strip()
    inquiry_question = request.form.get("inquiry_question", "").strip()
    pathway = request.form.get("pathway", "").strip()

    print(f"Received project data: Title={title}, Description={description}, Inquiry Question={inquiry_question}, Pathway={pathway}")

    try:
        evaluation = evaluate_project_idea(title, description, inquiry_question, pathway)
        current_evaluation = {"evaluation": evaluation}
    except Exception as e:
        print(f"Error during evaluation: {e}")
        current_evaluation = {"evaluation": "Error: Could not generate evaluation."}

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
