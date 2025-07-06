from flask import Flask, render_template, request
import PyPDF2
import os
from dotenv import load_dotenv
import requests

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

document_text = ""  # store the PDF text globally

def extract_text(pdf_stream):
    reader = PyPDF2.PdfReader(pdf_stream)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

def summarize_text(text):
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "model": "llama3-8b-8192",
            "messages": [
                {"role": "system", "content": "Summarize this document in 5 clear bullet points."},
                {"role": "user", "content": text[:6000]}
            ],
            "temperature": 0.3,
        }
        r = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=data)
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error summarizing: {e}"

def answer_question(text, question):
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        prompt = (
            f"Document:\n{text[:6000]}\n\n"
            f"Question: {question}\n\n"
            "Answer using the document only. If unsure, say 'I don't know.'"
        )
        data = {
            "model": "llama3-8b-8192",
            "messages": [
                {"role": "system", "content": "You are a precise business assistant."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2,
        }
        r = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=data)
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error answering: {e}"

@app.route("/", methods=["GET", "POST"])
def index():
    global document_text
    summary = None
    answer = None

    if request.method == "POST":
        if "pdf_file" in request.files:
            pdf = request.files["pdf_file"]
            document_text = extract_text(pdf)
            summary = summarize_text(document_text)
        elif request.form.get("question_mode"):
            question = request.form.get("user_question")
            answer = answer_question(document_text, question)

    return render_template("index.html", summary=summary, answer=answer)

if __name__ == "__main__":
    app.run()