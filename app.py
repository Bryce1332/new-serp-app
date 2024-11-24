import boto3
from flask import Flask, request, jsonify, render_template, redirect, url_for
from transformers import pipeline, AutoTokenizer

# Hugging Face model setup
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
tokenizer.pad_token = tokenizer.eos_token  # Set EOS token as padding token
generator = pipeline(
    "text-generation",
    model="EleutherAI/gpt-neo-125M",
    tokenizer=tokenizer,
    truncation=True,
    padding=True
)

app = Flask(__name__)
current_evaluation = {}

def evaluate_inquiry_question(inquiry_question):
    """
    Evaluate the inquiry question based on the 5 criteria:
    - Originality
    - Impactfulness
    - Feasibility
    - Quantifiable Data
    - Specificity
    """
    # Construct the evaluation prompt
    prompt = (
        f"Evaluate the following research inquiry question:\n\n"
        f"Inquiry Question: {inquiry_question}\n\n"
        f"Evaluate the question based on these criteria:\n"
        f"1. Originality: Is the question novel and unique?\n"
        f"2. Impactfulness: Does the question address a problem with scientific, societal, or environmental significance?\n"
        f"3. Feasibility: Can the project be realistically completed within two months by a high school student?\n"
        f"4. Quantifiable Data: Does the question suggest measurable outcomes or results?\n"
        f"5. Specificity: Is the question well-defined and focused?\n\n"
        f"Provide detailed feedback on each criterion and assign an overall score (out of 100)."
    )

    try:
        # Generate the evaluation
        print(f"Evaluating inquiry question: {inquiry_question}")
        generated = generator(
            prompt,
            max_new_tokens=200,  # Limit response length
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
    inquiry_question = request.form.get("inquiry_question", "")

    print(f"Received inquiry question: {inquiry_question}")

    try:
        evaluation = evaluate_inquiry_question(inquiry_question)
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
