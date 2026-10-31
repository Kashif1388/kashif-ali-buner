# 🎤 AI Mock Interview Coach

Practise a realistic interview and get every answer scored and coached. Built with **Streamlit** and the **Groq API**.

## How it works

1. Enter the **role** you're interviewing for, your seniority level and the interview type (behavioral, technical or mixed).
2. Optionally upload your **resume** (PDF, DOCX or TXT) and paste a **job description** so the questions are tailored to you.
3. The AI prepares 3–10 questions, ordered from easy to hard.
4. Answer them one at a time in a chat. After each answer you get:
   - a score out of 10 and a one-line verdict
   - what worked and how to improve
   - a stronger sample answer built from *your* content (with placeholders instead of invented facts)
5. At the end you get an overall score out of 100, a readiness level, top strengths, top improvements, next practice steps, and a downloadable Markdown report.

You can skip questions, end the interview early, or start over at any time. If an API call fails, your answer is kept so you can retry without retyping.

> Scores are AI estimates meant for practice. They are not a prediction of real hiring decisions.

## Project structure

```
mock-interview-coach/
├── app.py            # Streamlit app
├── requirements.txt  # Python dependencies
└── README.md
```

## Run locally

1. **Get a free Groq API key** at <https://console.groq.com/keys>.
2. **Install dependencies** (Python 3.9+; 3.11 or 3.12 recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. **Provide your key** using ONE of these options:
   - Paste it in the app's sidebar when it runs (simplest), or
   - Set an environment variable:
     ```bash
     export GROQ_API_KEY="gsk_..."          # Windows PowerShell: $env:GROQ_API_KEY="gsk_..."
     ```
   - Or create `.streamlit/secrets.toml`:
     ```toml
     GROQ_API_KEY = "gsk_..."
     ```
4. **Start the app:**
   ```bash
   streamlit run app.py
   ```
   It opens at <http://localhost:8501>.

## Deploy on Streamlit Community Cloud

1. Push the three files to a **GitHub repository** (repo root, public).
2. Go to <https://share.streamlit.io> and sign in with GitHub.
3. Click **Create app** → **Deploy a public app from GitHub**.
4. Pick your repository, branch `main`, main file path `app.py`.
5. Open **Advanced settings**, choose Python 3.11 or 3.12, and paste in **Secrets**:
   ```toml
   GROQ_API_KEY = "gsk_your_key_here"
   ```
6. Click **Deploy**.

## Security and privacy

- **Never commit your API key** to GitHub. Use Streamlit Secrets, an environment variable, or the sidebar field.
- If you create `.streamlit/secrets.toml` locally, add it to a `.gitignore` file.
- If a key is ever exposed, delete it at <https://console.groq.com/keys> and create a new one.
- The app does not store resumes or answers, but their text is sent to Groq for analysis. Don't upload anything you aren't comfortable sharing.
- A public deployment using your secret key lets visitors spend your Groq quota. To avoid that, leave the secret out and let each user paste their own key in the sidebar.

## Troubleshooting

| Problem | Fix |
|---|---|
| "API key was rejected" | Re-copy the key from the Groq console; no extra spaces or quotes. |
| "Rate limit reached" | Wait a minute, or choose GPT-OSS 20B in the sidebar. |
| "Model isn't available to your account" (404) | Some models (e.g. Llama 3.x) are enterprise-only. The sidebar lists only models your key can use; pick another. |
| Request too large (413) | Your plan's per-minute token limit was hit. Use GPT-OSS 20B or a shorter resume/job description. |
| "Almost no text could be extracted" | The PDF is probably a scanned image. Upload a text-based PDF or DOCX. |
| An answer failed to submit | Your text is saved on screen. Click **Retry submitting this answer**. |

## Customising

- **Models:** edit the `MODELS` dictionary at the top of `app.py`. See <https://console.groq.com/docs/models>.
- **Interview styles / levels:** edit `LEVELS` and `INTERVIEW_TYPES`.
- **Question and scoring behaviour:** edit `QUESTIONS_SYSTEM`, `EVAL_SYSTEM` and `SUMMARY_SYSTEM`.

## License

MIT: use it however you like.
