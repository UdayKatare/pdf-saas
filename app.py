from flask import Flask, render_template, request
import PyPDF2
import os
from dotenv import load_dotenv
import requests

# ============================================================
# FLASK APP CONFIGURATION
# ============================================================

app = Flask(__name__)

# Allow PDF uploads up to 500 MB
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024


# ============================================================
# ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

# Current Groq model
MODEL = "openai/gpt-oss-120b"

# Groq API endpoint
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


# ============================================================
# GLOBAL DOCUMENT STORAGE
# ============================================================

document_text = ""


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def extract_text(pdf_stream):
    """
    Extract text from an uploaded PDF.
    """

    try:
        reader = PyPDF2.PdfReader(pdf_stream)

        text = ""

        for page in reader.pages:

            try:
                page_text = page.extract_text()

                if page_text:
                    text += page_text + "\n"

            except Exception as e:
                print(f"Error extracting page: {e}")

        return text

    except Exception as e:
        print(f"PDF extraction error: {e}")
        return ""


# ============================================================
# GROQ API HELPER
# ============================================================

def call_groq(messages, temperature=0.2):
    """
    Send a request to Groq and safely process the response.
    """

    # --------------------------------------------------------
    # CHECK API KEY
    # --------------------------------------------------------

    if not api_key:
        return "Error: GROQ_API_KEY is not configured."

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    data = {
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
        "include_reasoning": False
    }

    try:

        # ----------------------------------------------------
        # SEND REQUEST
        # ----------------------------------------------------

        response = requests.post(
            GROQ_API_URL,
            headers=headers,
            json=data,
            timeout=60
        )

        # ----------------------------------------------------
        # HTTP ERROR
        # ----------------------------------------------------

        if not response.ok:

            try:
                error_data = response.json()

                print(
                    "Groq API Error:",
                    response.status_code,
                    error_data
                )

                error_message = (
                    error_data
                    .get("error", {})
                    .get("message")
                )

                if error_message:
                    return (
                        f"Groq API Error "
                        f"({response.status_code}): "
                        f"{error_message}"
                    )

                return (
                    f"Groq API Error "
                    f"({response.status_code}): "
                    f"{error_data}"
                )

            except Exception:

                print(
                    "Groq API Error:",
                    response.status_code,
                    response.text
                )

                return (
                    f"Groq API Error "
                    f"({response.status_code}): "
                    f"{response.text}"
                )

        # ----------------------------------------------------
        # PARSE RESPONSE
        # ----------------------------------------------------

        try:
            result = response.json()

        except ValueError:

            print(
                "Invalid JSON returned by Groq:",
                response.text
            )

            return "Error: Groq returned an invalid response."

        # ----------------------------------------------------
        # CHECK FOR CHOICES
        # ----------------------------------------------------

        if "choices" not in result:

            print(
                "Unexpected Groq response:",
                result
            )

            return (
                "Error: Groq returned an unexpected "
                "response format."
            )

        # ----------------------------------------------------
        # CHECK CHOICES
        # ----------------------------------------------------

        if not result["choices"]:

            print(
                "Groq returned an empty choices array:",
                result
            )

            return (
                "Error: Groq returned an empty response."
            )

        # ----------------------------------------------------
        # GET MESSAGE
        # ----------------------------------------------------

        message = result["choices"][0].get(
            "message",
            {}
        )

        content = message.get("content")

        # ----------------------------------------------------
        # CHECK CONTENT
        # ----------------------------------------------------

        if not content:

            print(
                "Groq response has no content:",
                result
            )

            return (
                "Error: Groq returned no response content."
            )

        return content

    # --------------------------------------------------------
    # TIMEOUT
    # --------------------------------------------------------

    except requests.exceptions.Timeout:

        return (
            "Error: The Groq request timed out. "
            "Please try again."
        )

    # --------------------------------------------------------
    # CONNECTION ERROR
    # --------------------------------------------------------

    except requests.exceptions.ConnectionError as e:

        print(
            "Groq connection error:",
            e
        )

        return (
            "Error: Could not connect to Groq."
        )

    # --------------------------------------------------------
    # REQUEST ERROR
    # --------------------------------------------------------

    except requests.exceptions.RequestException as e:

        print(
            "Groq request error:",
            e
        )

        return (
            f"Error communicating with Groq: {e}"
        )

    # --------------------------------------------------------
    # UNKNOWN ERROR
    # --------------------------------------------------------

    except Exception as e:

        print(
            "Unexpected Groq error:",
            e
        )

        return (
            f"Unexpected error: {e}"
        )


# ============================================================
# SUMMARIZATION
# ============================================================

def summarize_text(text):
    """
    Generate a 5-point summary using GPT-OSS 120B.
    """

    if not text or not text.strip():
        return "Error: No text was extracted from the PDF."

    messages = [
        {
            "role": "system",
            "content": (
                "You are a professional document summarizer. "
                "Summarize the provided document clearly and accurately.\n\n"
                "Return exactly 5 concise bullet points.\n\n"
                "Focus on the most important:\n"
                "- Facts\n"
                "- Key decisions\n"
                "- Numbers\n"
                "- Dates\n"
                "- Requirements\n"
                "- Business-relevant information\n\n"
                "Do not invent information. "
                "Use only information contained in the document."
            )
        },
        {
            "role": "user",
            "content": (
                "Summarize the following document:\n\n"
                + text[:6000]
            )
        }
    ]

    return call_groq(
        messages,
        temperature=0.2
    )


# ============================================================
# QUESTION ANSWERING
# ============================================================

def answer_question(text, question):
    """
    Answer a question using only the uploaded document.
    """

    if not text or not text.strip():
        return "Error: No document has been uploaded yet."

    if not question or not question.strip():
        return "Please enter a question."

    messages = [
        {
            "role": "system",
            "content": (
                "You are a precise document assistant.\n\n"
                "Answer the user's question using ONLY "
                "information contained in the provided document.\n\n"
                "Rules:\n"
                "1. Do not invent information.\n"
                "2. Do not use outside knowledge.\n"
                "3. If the answer cannot be found in the document, "
                "say: I don't know.\n"
                "4. Keep the answer clear and concise."
            )
        },
        {
            "role": "user",
            "content": (
                "DOCUMENT:\n"
                "-------------------------\n"
                f"{text[:6000]}\n"
                "-------------------------\n\n"
                f"QUESTION:\n{question}"
            )
        }
    ]

    return call_groq(
        messages,
        temperature=0.2
    )


# ============================================================
# MAIN ROUTE
# ============================================================

@app.route("/", methods=["GET", "POST"])
def index():

    global document_text

    summary = None
    answer = None

    # --------------------------------------------------------
    # POST REQUEST
    # --------------------------------------------------------

    if request.method == "POST":

        # ====================================================
        # PDF UPLOAD
        # ====================================================

        if "pdf_file" in request.files:

            pdf = request.files["pdf_file"]

            if not pdf or not pdf.filename:

                summary = "Please select a PDF file."

            else:

                print(
                    f"Processing PDF: {pdf.filename}"
                )

                # Extract PDF text
                document_text = extract_text(pdf)

                # Check extracted text
                if not document_text.strip():

                    summary = (
                        "Error: Could not extract text "
                        "from this PDF. It may be scanned "
                        "or image-based."
                    )

                else:

                    print(
                        f"Extracted {len(document_text)} "
                        f"characters from PDF."
                    )

                    # Generate summary
                    summary = summarize_text(
                        document_text
                    )

        # ====================================================
        # QUESTION MODE
        # ====================================================

        elif request.form.get("question_mode"):

            question = request.form.get(
                "user_question",
                ""
            )

            answer = answer_question(
                document_text,
                question
            )

    # --------------------------------------------------------
    # RENDER TEMPLATE
    # --------------------------------------------------------

    return render_template(
        "index.html",
        summary=summary,
        answer=answer
    )


# ============================================================
# APPLICATION START
# ============================================================

if __name__ == "__main__":

    # Render provides PORT through environment variable.
    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port
    )
