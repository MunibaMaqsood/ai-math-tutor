# AI Mathematics Tutor (Stage 1)

## Setup (VS Code terminal)
    python -m venv .venv
    .venv\Scripts\activate          # Windows
    source .venv/bin/activate       # macOS/Linux
    pip install -r requirements.txt
    copy .env.example .env          # Windows (use cp on macOS/Linux)
    # edit .env and paste your Google AI Studio key

## Test and run
    pytest
    streamlit run app.py
