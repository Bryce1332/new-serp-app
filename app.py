import boto3
from flask import Flask, request, jsonify, render_template
from transformers import pipeline
import json

# AWS S3 setup
S3_BUCKET = "serp-app-bucket"
ISEF_PROJECTS_FILE = "isef-projects.json"
s3_client = boto3.client("s3")

# Hugging Face model setup
generator = pipeline("text-generation", model="EleutherAI/gpt-neo-1.3B")

def fetch_isef_data(query):
    """
    Fetch relevant ISEF database entries for a project idea to assist in evaluation.

    Args:
        query (str): Search query related to the project idea.

    Returns:
        list: Relevant ISEF projects and their details.
    """
    try:
        # Download the JSON file from S3
        response = s3_client.get_object(Bucket=S3_BUCKET, Key=ISEF_PROJECTS_FILE)
        isef_projects = json.loads(response["Body"].read())

        # Search for relevant projects
        relevant_projects = [
            project for project in isef_projects if query.lower() in project["title"].lower()
        ]

        return relevant_projects
    except Exception as e:
        print(f"Error fetching ISEF data: {e}")
        return []

def evaluate_project_idea(title, description, objectives, methods, feasibility, impact, pathway):
    """
    Evaluates a SERP project idea based on various criteria using Hugging Face and ISEF data.

    Args:
        title (str): Title of the project idea.
        description (str): Brief description of the project.
        objectives (list): Key objectives of the project.
        methods (list): Methods or approaches to be used in the project.
        feasibility (str): Assessment of feasibility (resources, timeline, skills).
        impact (str): Expected scientific, societal, or environmental impact.
        pathway (str): The pathway of the project ("science", "engineering", or "computer science").

    Returns:
        str: Evaluation results for the project.
    """
    # Fetch similar projects from the ISEF database
    similar_projects = fetch_isef_data(title)

    # Summarize fetched ISEF data
    isef_summary = "\n".join([
        f"Title: {proj['title']}, Category: {proj['category']}, Score: {proj.get('score', 'N/A')}"
        for proj in similar_projects
    ])

    # Tailor suggestions based on the pathway
    if pathway == "science":
        pathway_criteria = (
            "Emphasize forming a clear hypothesis, designing robust experiments, and collecting measurable data."
        )
    elif pathway == "engineering":
        pathway_criteria = (
            "Focus on problem-solving, designing and testing prototypes, and evaluating the efficiency and scalability of the solution."
        )
    elif pathway == "computer science":
        pathway_criteria = (
            "Concentrate on algorithm design, software development, and evaluating performance metrics and usability."
        )
    else:
        pathway_criteria = "Pathway not specified. General evaluation criteria applied."

    # Construct the evaluation prompt
    prompt = (
        f"Evaluate the following SERP project idea:\n\n"
        f"Title: {title}\n\n"
        f"Description: {description}\n\n"
        f"Objectives: {', '.join(objectives)}\n\n"
        f"Methods: {', '.join(methods)}\n\n"
        f"Feasibility: {feasibility}\n\n"
        f"Impact: {impact}\n\n"
        f"Here are similar projects from the ISEF database:\n{isef_summary}\n\n"
        f"Pathway: {pathway}\n"
        f"Suggestions based on pathway: {pathway_criteria}\n\n"
        "Provide a detailed evaluation based on the following criteria:\n"
        "1. Originality: How novel is the idea?\n"
        "2. Impactfulness: What is the potential scientific, societal, or environmental impact?\n"
        "3. Feasibility: Is the project achievable within a 2-month timeframe and suitable for a high school student?\n"
        "4. Quantifiable Data: Does the project provide measurable outcomes or results?\n"
        "5. Specificity: Are the objectives and methods clearly defined and specific?\n"
        "Provide a summary of strengths, weaknesses, and an overall score (out of 100)."
    )

    # Use Hugging Face to generate the evaluation
    generated = generator(prompt, max_length=300, num_return_sequences=1)
    evaluation = generated[0]["generated_text"]

    return evaluation

# Flask app setup
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("project_evaluator_ui.html")
    
@app.route("/evaluate", methods=["POST"])
def evaluate():
    data = request.get_json()
    title = data.get("title", "")
    description = data.get("description", "")
    objectives = data.get("objectives", [])  # Fixed the string
    methods = data.get("methods", [])
    feasibility = data.get("feasibility", "")
    impact = data.get("impact", "")
    pathway = data.get("pathway", "")

    result = evaluate_project_idea(title, description, objectives, methods, feasibility, impact, pathway)
    return jsonify({"evaluation": result})

