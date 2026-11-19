"""Stage 2 app: Home, Upload PDF, Chatbot, Solver, Graphs.
Run: streamlit run app.py"""
import csv
import io
import time

import streamlit as st

import config
from src.pdf_processor import process_pdf
from src.plotting import plot_functions
from src.llm import extract_text, get_llm
from src.rag_chain import answer_question
from src.render import render_math
from src.solver import (derivative, euclid_steps, explain_verified, integral,
                        matrix_info, modular, solve_equation, solve_system,
                        solve_word_problem)
from src.storage import (apply_settings, clear_history, get_events, load_settings,
                         log_event, quiz_scores, save_settings)
from src.study_tools import (find_formulas, make_flashcards, make_mcqs,
                             make_written, strip_dollars, summarize)
from src.vector_store import add_chunks, delete_source, list_sources

st.set_page_config(page_title="AI Math Tutor", page_icon="📐", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 2rem;}
.card {background:#f4f6ff; border-radius:12px; padding:1.2rem; border:1px solid #dfe3f7;}
</style>""", unsafe_allow_html=True)

if not config.GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY is missing. Check your .env file.")
    st.stop()

apply_settings(load_settings())  # model, sources and answer length saved in Settings

page = st.sidebar.radio("Navigation", [
    "🏠 Home", "📄 Upload PDF", "💬 Chatbot", "🧮 Solver", "📈 Graphs",
    "📝 Quiz", "🃏 Flashcards", "📚 Summary", "📐 Formulas",
    "🕘 History", "⚙️ Settings"])
st.sidebar.caption("Final polish and deployment come next.")


def show_calc(res: dict, label: str, explain: bool):
    """Show a SymPy result, then optionally an AI explanation of it."""
    log_event("solver", label)
    st.latex(res["problem"])
    st.markdown("**Result (computed by SymPy):**")
    st.latex(res["result"])
    if res.get("checked"):
        st.success("✅ Verified: the answer was checked by substitution/differentiation.")
    if explain:
        with st.spinner("Writing the explanation..."):
            render_math(explain_verified(label, res["plain"]))


def study_inputs(key: str):
    """Document + topic pickers shared by the study pages."""
    sources = list_sources()
    if not sources:
        st.info("Upload and index a PDF first (📄 Upload PDF).")
        st.stop()
    c1, c2 = st.columns(2)
    choice = c1.selectbox("Document", ["All documents"] + sources, key=f"{key}_src")
    topic = c2.text_input("Topic (leave empty for the main ideas)", key=f"{key}_topic",
                          placeholder="e.g. Euclidean algorithm")
    return (None if choice == "All documents" else choice), topic


# ------------------------------------------------------------------ Home
if page == "🏠 Home":
    st.title("📐 AI Mathematics Tutor")
    st.write("Upload your textbooks and notes, ask questions, solve problems and plot graphs.")
    c1, c2 = st.columns(2)
    c1.markdown(f'<div class="card"><h3>{len(list_sources())}</h3>Documents indexed</div>',
                unsafe_allow_html=True)
    c2.markdown(f'<div class="card"><h3>{config.GEMINI_MODEL}</h3>Active model</div>',
                unsafe_allow_html=True)

# ------------------------------------------------------------------ Upload
elif page == "📄 Upload PDF":
    st.title("📄 Upload PDF")
    files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
    if files and st.button("Process and index"):
        for f in files:
            path = config.UPLOAD_DIR / f.name
            path.write_bytes(f.getbuffer())
            with st.spinner(f"Processing {f.name}..."):
                chunks = process_pdf(path)
                if not chunks:
                    st.warning(f"No text found in {f.name} (scanned PDF? OCR needed).")
                    continue
                add_chunks(chunks)
            st.success(f"{f.name}: {len(chunks)} chunks indexed.")
    st.subheader("Indexed documents")
    for name in list_sources():
        col1, col2 = st.columns([5, 1])
        col1.write(name)
        if col2.button("Delete", key=name):
            delete_source(name)
            st.rerun()

# ------------------------------------------------------------------ Chatbot
elif page == "💬 Chatbot":
    st.title("💬 Mathematics Chatbot")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            render_math(m["content"])
            for s in m.get("sources", []):
                st.caption(f"📖 {s['source']} — p. {s['page']}")
    if q := st.chat_input("Ask about your uploaded materials..."):
        st.session_state.messages.append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, sources = answer_question(q)
            log_event("question", q)
            render_math(answer)
            with st.expander("Sources"):
                for s in sources:
                    st.markdown(f"**{s['source']}, p. {s['page']}**")
                    st.text(s["snippet"])
        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources})

# ------------------------------------------------------------------ Solver
elif page == "🧮 Solver":
    st.title("🧮 Step-by-Step Solver")
    st.caption("The calculators use SymPy for the answer, so numbers are reliable. "
               "The AI only explains the steps.")
    tabs = st.tabs(["Word problem (AI)", "Equation", "Linear system", "Derivative",
                    "Integral", "Matrix", "Euclid & modular"])

    with tabs[0]:
        problem = st.text_area("Type any math problem", key="wp",
                               placeholder="A train travels 120 km in 1.5 hours...")
        st.info("This tab is answered by the AI alone and is not machine-checked. "
                "Use the other tabs when you need a verified answer.")
        if st.button("Solve", key="wp_btn") and problem.strip():
            with st.spinner("Solving..."):
                render_math(solve_word_problem(problem))

    with tabs[1]:
        eq = st.text_input("Equation", "x^2 - 5x + 6 = 0", key="eq")
        var = st.text_input("Solve for", "x", key="eq_var")
        ex = st.checkbox("Explain step by step with AI", True, key="eq_ex")
        if st.button("Solve equation"):
            try:
                show_calc(solve_equation(eq, var), f"Solve {eq} for {var}", ex)
            except Exception as e:
                st.error(f"Could not solve: {e}")

    with tabs[2]:
        sysin = st.text_area("One equation per line", "2x + y = 5\nx - y = 1", key="sys")
        ex = st.checkbox("Explain step by step with AI", True, key="sys_ex")
        if st.button("Solve system"):
            try:
                show_calc(solve_system(sysin), f"Solve the system:\n{sysin}", ex)
            except Exception as e:
                st.error(f"Could not solve: {e}")

    with tabs[3]:
        f = st.text_input("Function", "x^3 + 2x", key="d_f")
        c1, c2 = st.columns(2)
        v = c1.text_input("Variable", "x", key="d_v")
        order = c2.number_input("Order", 1, 5, 1, key="d_o")
        ex = st.checkbox("Explain step by step with AI", True, key="d_ex")
        if st.button("Differentiate"):
            try:
                show_calc(derivative(f, v, int(order)),
                          f"Find the derivative (order {int(order)}) of {f} with respect to {v}", ex)
            except Exception as e:
                st.error(f"Could not differentiate: {e}")

    with tabs[4]:
        f = st.text_input("Function", "x^2", key="i_f")
        c1, c2, c3 = st.columns(3)
        v = c1.text_input("Variable", "x", key="i_v")
        lo = c2.text_input("Lower limit (optional)", "", key="i_lo")
        hi = c3.text_input("Upper limit (optional)", "", key="i_hi")
        ex = st.checkbox("Explain step by step with AI", True, key="i_ex")
        if st.button("Integrate"):
            try:
                show_calc(integral(f, v, lo, hi),
                          f"Integrate {f} d{v}" + (f" from {lo} to {hi}" if lo and hi else ""), ex)
            except Exception as e:
                st.error(f"Could not integrate: {e}")

    with tabs[5]:
        mat = st.text_area("Matrix (rows separated by ; entries by commas)",
                           "1, 2; 3, 4", key="mat")
        ex = st.checkbox("Explain step by step with AI", True, key="mat_ex")
        if st.button("Analyse matrix"):
            try:
                show_calc(matrix_info(mat), f"Analyse the matrix {mat}: determinant, "
                          "inverse, rank and eigenvalues", ex)
            except Exception as e:
                st.error(f"Could not analyse: {e}")

    with tabs[6]:
        st.subheader("Euclidean algorithm")
        c1, c2 = st.columns(2)
        a = c1.number_input("a", 1, value=48, key="ea")
        b = c2.number_input("b", 1, value=18, key="eb")
        if st.button("Run Euclid"):
            r = euclid_steps(int(a), int(b))
            for s in r["steps"]:
                st.write(s["equation"])
            st.success(f"gcd({int(a)}, {int(b)}) = {r['gcd']}")
            st.write(f"Bézout identity: {r['bezout']}")
        st.subheader("Modular arithmetic")
        c1, c2, c3 = st.columns(3)
        ma = c1.number_input("a", 1, value=3, key="ma")
        mm = c2.number_input("modulus m", 2, value=7, key="mm")
        me = c3.number_input("exponent (optional, 0 = skip)", 0, value=0, key="me")
        if st.button("Compute modular"):
            r = modular(int(ma), int(mm), int(me) if me else None)
            st.write(f"gcd(a, m) = {r['gcd']}")
            st.write(f"Inverse of a mod m: {r['inverse'] if r['inverse'] is not None else 'does not exist'}")
            if "power" in r:
                st.write(f"a^e mod m = {r['power']}")

# ------------------------------------------------------------------ Graphs
elif page == "📈 Graphs":
    st.title("📈 Function Graphs")
    funcs = st.text_input("Functions (separate with commas)", "sin(x), x^2/10")
    c1, c2, c3 = st.columns(3)
    xmin = c1.number_input("x min", value=-10.0)
    xmax = c2.number_input("x max", value=10.0)
    var = c3.text_input("Variable", "x")
    fix_y = st.checkbox("Set y-range manually (useful for tan(x), 1/x)")
    ylim = None
    if fix_y:
        d1, d2 = st.columns(2)
        ylim = (d1.number_input("y min", value=-5.0), d2.number_input("y max", value=5.0))
    if st.button("Plot"):
        try:
            names = [f.strip() for f in funcs.split(",") if f.strip()]
            st.pyplot(plot_functions(names, var, xmin, xmax, ylim))
        except Exception as e:
            st.error(f"Could not plot: {e}")


# ------------------------------------------------------------------ Quiz
elif page == "📝 Quiz":
    st.title("📝 Quiz Generator")
    source, topic = study_inputs("quiz")
    mode = st.radio("Question type", ["Multiple choice", "Short questions", "Long questions"],
                    horizontal=True)
    c1, c2 = st.columns(2)
    n = c1.slider("Number of questions", 3, 10, 5)
    difficulty = c2.selectbox("Difficulty", ["Easy", "Medium", "Hard"], index=1)
    if st.button("Generate quiz"):
        with st.spinner("Writing questions from your documents..."):
            try:
                if mode == "Multiple choice":
                    items = make_mcqs(topic, source, n, difficulty)
                    kind = "mcq"
                else:
                    items = make_written(topic, source, n,
                                         "short" if mode == "Short questions" else "long")
                    kind = "written"
                st.session_state.quiz = {"id": time.time_ns(), "kind": kind, "items": items}
                if kind == "written":
                    log_event("written quiz", f"{mode}: {topic or 'main ideas'}")
            except Exception as e:
                st.error(f"Could not generate the quiz: {e}")
    quiz = st.session_state.get("quiz")
    if quiz and not quiz["items"]:
        st.warning("No valid questions were produced. Try a broader topic or regenerate.")
    elif quiz and quiz["kind"] == "mcq":
        with st.form("quiz_form"):
            answers = []
            for i, q in enumerate(quiz["items"]):
                st.markdown(f"**Q{i + 1}.** {q.question}")
                answers.append(st.radio("Choose one", q.options, index=None,
                                        key=f"q{quiz['id']}_{i}", label_visibility="collapsed"))
            submitted = st.form_submit_button("Submit answers")
        if submitted:
            score = 0
            for i, (q, a) in enumerate(zip(quiz["items"], answers)):
                correct = q.options[q.answer_index]
                if a == correct:
                    score += 1
                    st.success(f"Q{i + 1}: correct ✅")
                else:
                    st.error(f"Q{i + 1}: your answer: {a or 'none'}  |  correct: {correct}")
                page_note = f" (p. {q.page})" if q.page else ""
                st.markdown(f"_{q.explanation}_{page_note}")
            st.subheader(f"Score: {score} / {len(quiz['items'])}")
            log_event("quiz", f"{topic or 'main ideas'} ({difficulty})", score, len(quiz["items"]))
    elif quiz:
        for i, q in enumerate(quiz["items"]):
            st.markdown(f"**Q{i + 1}.** {q.question}")
            with st.expander("Show model answer"):
                st.markdown(q.model_answer + (f" (p. {q.page})" if q.page else ""))

# ------------------------------------------------------------------ Flashcards
elif page == "🃏 Flashcards":
    st.title("🃏 Flashcards")
    source, topic = study_inputs("fc")
    n = st.slider("Number of cards", 5, 20, 10)
    if st.button("Generate flashcards"):
        with st.spinner("Creating flashcards..."):
            try:
                st.session_state.cards = make_flashcards(topic, source, n)
                log_event("flashcards", topic or "main ideas")
                st.session_state.card_i = 0
                st.session_state.show_back = False
            except Exception as e:
                st.error(f"Could not create flashcards: {e}")
    cards = st.session_state.get("cards")
    if cards:
        def _move(step):
            st.session_state.card_i = (st.session_state.card_i + step) % len(cards)
            st.session_state.show_back = False

        def _flip():
            st.session_state.show_back = not st.session_state.show_back

        i = min(st.session_state.card_i, len(cards) - 1)
        card = cards[i]
        st.caption(f"Card {i + 1} of {len(cards)}")
        with st.container(border=True):
            st.markdown("#### " + (card.back if st.session_state.show_back else card.front))
            if st.session_state.show_back and card.page:
                st.caption(f"Source page {card.page}")
        b1, b2, b3 = st.columns(3)
        b1.button("◀ Previous", on_click=_move, args=(-1,))
        b2.button("Hide answer" if st.session_state.show_back else "Show answer", on_click=_flip)
        b3.button("Next ▶", on_click=_move, args=(1,))
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["Front", "Back", "Page"])
        for c in cards:
            writer.writerow([c.front, c.back, c.page])
        st.download_button("⬇ Download as CSV", buf.getvalue(), "flashcards.csv", "text/csv")

# ------------------------------------------------------------------ Summary
elif page == "📚 Summary":
    st.title("📚 Chapter Summary")
    source, topic = study_inputs("sum")
    if st.button("Summarise"):
        with st.spinner("Reading and summarising..."):
            try:
                st.session_state.summary = summarize(topic, source)
                log_event("summary", topic or "main ideas")
            except Exception as e:
                st.error(f"Could not summarise: {e}")
    if st.session_state.get("summary"):
        render_math(st.session_state.summary)
        st.download_button("⬇ Download summary", st.session_state.summary,
                           "summary.md", "text/markdown")

# ------------------------------------------------------------------ Formulas
elif page == "📐 Formulas":
    st.title("📐 Formula Library")
    source, topic = study_inputs("fm")
    if st.button("Find formulas and definitions"):
        with st.spinner("Scanning your documents..."):
            try:
                st.session_state.formulas = find_formulas(topic, source)
                log_event("formulas", topic or "main ideas")
            except Exception as e:
                st.error(f"Could not extract formulas: {e}")
    for f in st.session_state.get("formulas", []):
        with st.container(border=True):
            st.markdown(f"**{f.name}**" + (f"  ·  p. {f.page}" if f.page else ""))
            st.latex(strip_dollars(f.latex))
            st.caption(f.meaning)


# ------------------------------------------------------------------ History
elif page == "🕘 History":
    st.title("🕘 Study History")
    events = get_events(500)
    if not events:
        st.info("Nothing here yet. Ask questions, solve problems or take a quiz and they will appear.")
    else:
        kinds = [e["type"] for e in events]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Questions asked", kinds.count("question"))
        c2.metric("Quizzes taken", kinds.count("quiz"))
        c3.metric("Problems solved", kinds.count("solver"))
        c4.metric("All activities", len(events))
        scores = quiz_scores()
        if scores:
            st.subheader("Quiz scores (%), oldest to newest")
            st.bar_chart ({"Score %": scores})
            st.caption(f"Average: {sum(scores) / len(scores):.0f}%   |   Best: {max(scores):.0f}%")
        st.subheader("Recent activity")
        st.dataframe(events, hide_index=True)
        if st.checkbox("I want to delete all history"):
            if st.button("Delete history"):
                clear_history()
                st.rerun()

# ------------------------------------------------------------------ Settings
elif page == "⚙️ Settings":
    st.title("⚙️ Settings")
    current = load_settings()
    model = st.text_input("Gemini model", current["model"],
                          help="Use the exact name from Google AI Studio.")
    st.caption("If you see a 404 (model not found) or 429 (daily limit) error, try another "
               "Flash or Flash-Lite model name here, then click Save.")
    top_k = st.slider("Sources retrieved per question", 2, 12, int(current["top_k"]))
    style = st.radio("Chatbot answer length", ["Brief", "Normal", "Detailed"],
                     index=["Brief", "Normal", "Detailed"].index(current["answer_style"]),
                     horizontal=True)
    if st.button("Save settings"):
        saved = save_settings({"model": model, "top_k": top_k, "answer_style": style})
        apply_settings(saved)
        st.success(f"Saved. Now using {saved['model']}.")

    st.divider()
    st.subheader("Status")
    st.write("API key: ✅ found" if config.GOOGLE_API_KEY else "API key: ❌ missing")
    st.write(f"Indexed documents: {len(list_sources())}")
    if st.button("Test AI connection (uses 1 request)"):
        with st.spinner("Contacting Gemini..."):
            try:
                reply = extract_text(get_llm(0).invoke("Reply with the single word OK.").content)
                st.success(f"✅ {config.GEMINI_MODEL} answered: {reply.strip()[:40]}")
            except Exception as e:
                st.error(f"❌ {str(e)[:400]}")
