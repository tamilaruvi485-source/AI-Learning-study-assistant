import streamlit as st
import sqlite3
import re
import random
from pathlib import Path
from datetime import datetime

from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Learning & Study Assistant",
    page_icon="📚",
    layout="wide"
)


# ============================================================
# CONSTANTS
# ============================================================

DB_NAME = "assistant_memory.db"
DEFAULT_KB = "knowledge_base.txt"


# ============================================================
# DATABASE / LONG-TERM MEMORY
# ============================================================

def init_database():

    connection = sqlite3.connect(DB_NAME)

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            question TEXT,
            answer TEXT,
            tool_used TEXT
        )
    """)

    connection.commit()
    connection.close()


def save_memory(question, answer, tool_used):

    connection = sqlite3.connect(DB_NAME)

    cursor = connection.cursor()

    cursor.execute("""
        INSERT INTO chat_history
        (timestamp, question, answer, tool_used)
        VALUES (?, ?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        question,
        answer,
        tool_used
    ))

    connection.commit()
    connection.close()


def get_memory(limit=50):

    connection = sqlite3.connect(DB_NAME)

    cursor = connection.cursor()

    cursor.execute("""
        SELECT timestamp, question, answer, tool_used
        FROM chat_history
        ORDER BY id DESC
        LIMIT ?
    """, (limit,))

    rows = cursor.fetchall()

    connection.close()

    return rows


def clear_memory():

    connection = sqlite3.connect(DB_NAME)

    cursor = connection.cursor()

    cursor.execute("DELETE FROM chat_history")

    connection.commit()
    connection.close()


init_database()


# ============================================================
# DOCUMENT PROCESSING
# ============================================================

def read_pdf(uploaded_file):

    reader = PdfReader(uploaded_file)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


def split_text(text, chunk_size=100):

    words = text.split()

    chunks = []

    for i in range(0, len(words), chunk_size):

        chunk = " ".join(
            words[i:i + chunk_size]
        )

        if chunk.strip():

            chunks.append(chunk)

    return chunks


# ============================================================
# RAG RETRIEVAL
# ============================================================

def retrieve_context(question, documents, top_k=4):

    if not documents:

        return ""

    try:

        vectorizer = TfidfVectorizer(
            stop_words="english"
        )

        document_vectors = vectorizer.fit_transform(
            documents
        )

        question_vector = vectorizer.transform(
            [question]
        )

        similarities = cosine_similarity(
            question_vector,
            document_vectors
        ).flatten()

        best_indexes = similarities.argsort()[
            -top_k:
        ][::-1]

        selected = []

        for index in best_indexes:

            if similarities[index] > 0:

                selected.append(
                    documents[index]
                )

        return "\n\n".join(selected)

    except Exception:

        return ""


# ============================================================
# QUESTION BANK
# ============================================================

QUESTION_BANK = {

    "Artificial Intelligence": [

        {
            "question": "What is Artificial Intelligence?",
            "options": [
                "A field that creates systems capable of intelligent tasks",
                "A database system",
                "A programming language",
                "A computer network"
            ],
            "answer": "A field that creates systems capable of intelligent tasks"
        },

        {
            "question": "Which of the following is an AI task?",
            "options": [
                "Reasoning",
                "Only printing",
                "Only typing",
                "Only file copying"
            ],
            "answer": "Reasoning"
        },

        {
            "question": "Which technology helps computers understand human language?",
            "options": [
                "Natural Language Processing",
                "Computer Graphics",
                "Operating Systems",
                "Networking"
            ],
            "answer": "Natural Language Processing"
        },

        {
            "question": "Which is an example of Generative AI?",
            "options": [
                "A system that generates new text",
                "A keyboard",
                "A printer",
                "A hard disk"
            ],
            "answer": "A system that generates new text"
        },

        {
            "question": "AI systems can be designed to perform:",
            "options": [
                "Reasoning and decision making",
                "Only calculations",
                "Only file storage",
                "Only printing"
            ],
            "answer": "Reasoning and decision making"
        }

    ],


    "Machine Learning": [

        {
            "question": "Machine Learning is a subset of:",
            "options": [
                "Artificial Intelligence",
                "Computer Networks",
                "Operating Systems",
                "Database Management"
            ],
            "answer": "Artificial Intelligence"
        },

        {
            "question": "Which learning method uses labelled data?",
            "options": [
                "Supervised Learning",
                "Unsupervised Learning",
                "Random Learning",
                "Manual Learning"
            ],
            "answer": "Supervised Learning"
        },

        {
            "question": "Which is an example of unsupervised learning?",
            "options": [
                "Clustering",
                "Classification with labels",
                "Regression with labels",
                "Manual sorting"
            ],
            "answer": "Clustering"
        },

        {
            "question": "What does a machine learning model learn from?",
            "options": [
                "Data",
                "Only images",
                "Only keyboards",
                "Only monitors"
            ],
            "answer": "Data"
        },

        {
            "question": "Regression is commonly used to predict:",
            "options": [
                "Continuous values",
                "Only categories",
                "Only images",
                "Only text files"
            ],
            "answer": "Continuous values"
        }

    ],


    "RAG": [

        {
            "question": "What does RAG stand for?",
            "options": [
                "Retrieval Augmented Generation",
                "Random AI Generation",
                "Rapid Agent Gateway",
                "Retrieval Agent Graph"
            ],
            "answer": "Retrieval Augmented Generation"
        },

        {
            "question": "What does RAG retrieve information from?",
            "options": [
                "A knowledge source",
                "Only a calculator",
                "Only an image",
                "Only a keyboard"
            ],
            "answer": "A knowledge source"
        },

        {
            "question": "What is a major benefit of RAG?",
            "options": [
                "It can answer using specific documents",
                "It removes all data",
                "It replaces the operating system",
                "It creates hardware"
            ],
            "answer": "It can answer using specific documents"
        },

        {
            "question": "RAG combines retrieval with:",
            "options": [
                "Generation",
                "Printing",
                "Networking",
                "File compression"
            ],
            "answer": "Generation"
        },

        {
            "question": "RAG is useful for:",
            "options": [
                "Document-based question answering",
                "Keyboard repair",
                "Screen replacement",
                "Hardware manufacturing"
            ],
            "answer": "Document-based question answering"
        }

    ],


    "Agentic AI": [

        {
            "question": "What can an AI agent use to perform tasks?",
            "options": [
                "Tools",
                "Only images",
                "Only a monitor",
                "Only a keyboard"
            ],
            "answer": "Tools"
        },

        {
            "question": "What helps an agent remember previous interactions?",
            "options": [
                "Memory",
                "Monitor",
                "Keyboard",
                "Printer"
            ],
            "answer": "Memory"
        },

        {
            "question": "What does an agent router do?",
            "options": [
                "Selects the appropriate task or tool",
                "Deletes all data",
                "Formats a computer",
                "Creates hardware"
            ],
            "answer": "Selects the appropriate task or tool"
        },

        {
            "question": "Agentic AI can perform:",
            "options": [
                "Multi-step tasks",
                "Only printing",
                "Only typing",
                "Only file copying"
            ],
            "answer": "Multi-step tasks"
        },

        {
            "question": "Which is an important part of an agentic system?",
            "options": [
                "Decision making",
                "Only a monitor",
                "Only a printer",
                "Only a keyboard"
            ],
            "answer": "Decision making"
        }

    ],


    "Natural Language Processing": [

        {
            "question": "What does NLP stand for?",
            "options": [
                "Natural Language Processing",
                "Network Language Protocol",
                "New Learning Program",
                "Natural Logic Processor"
            ],
            "answer": "Natural Language Processing"
        },

        {
            "question": "NLP mainly deals with:",
            "options": [
                "Human language",
                "Computer hardware",
                "Network cables",
                "Printers"
            ],
            "answer": "Human language"
        },

        {
            "question": "Which is an NLP application?",
            "options": [
                "Chatbots",
                "Keyboard manufacturing",
                "Hard disk repair",
                "Monitor assembly"
            ],
            "answer": "Chatbots"
        },

        {
            "question": "Text summarization is an example of:",
            "options": [
                "NLP",
                "Computer hardware",
                "Networking",
                "Database storage"
            ],
            "answer": "NLP"
        },

        {
            "question": "Language translation is related to:",
            "options": [
                "NLP",
                "Computer graphics",
                "Operating systems",
                "Hardware design"
            ],
            "answer": "NLP"
        }

    ]

}


# ============================================================
# QUIZ TOOL - KNOWLEDGE BASE
# ============================================================

def generate_quiz(topic, number=5):

    questions = QUESTION_BANK.get(
        topic,
        []
    )

    if not questions:

        return []

    number = min(
        number,
        len(questions)
    )

    return random.sample(
        questions,
        number
    )


# ============================================================
# QUIZ TOOL - UPLOADED MATERIAL
# ============================================================

def generate_pdf_quiz(pdf_text, number=5):

    """
    Generate MCQs from uploaded PDF/TXT material.

    This version uses the actual uploaded material
    and creates simple content-based MCQs without
    changing the existing Knowledge Base quiz.
    """

    if not pdf_text.strip():

        return []

    # Split uploaded material into sentences
    sentences = re.split(
        r'(?<=[.!?])\s+',
        pdf_text
    )

    sentences = [
        sentence.strip()
        for sentence in sentences
        if len(sentence.strip().split()) >= 6
    ]

    # Remove duplicate sentences
    sentences = list(
        dict.fromkeys(sentences)
    )

    if not sentences:

        return []

    random.shuffle(sentences)

    quiz = []

    # Common words that are not useful as answers
    ignored_words = {
        "this",
        "that",
        "these",
        "those",
        "which",
        "where",
        "there",
        "their",
        "about",
        "using",
        "from",
        "with",
        "into",
        "have",
        "has",
        "will",
        "also",
        "than",
        "then",
        "they",
        "them",
        "only",
        "such",
        "more",
        "some",
        "many",
        "what",
        "when"
    }

    for sentence in sentences:

        if len(quiz) >= number:

            break

        words = re.findall(
            r'\b[A-Za-z][A-Za-z-]{3,}\b',
            sentence
        )

        # Remove common words
        useful_words = [
            word
            for word in words
            if word.lower() not in ignored_words
        ]

        # Need enough words for question/options
        if len(useful_words) < 5:

            continue

        # Prefer longer words because they are
        # generally better content terms
        unique_words = list(
            dict.fromkeys(
                useful_words
            )
        )

        unique_words.sort(
            key=len,
            reverse=True
        )

        # Select answer from important words
        answer_candidates = unique_words[:5]

        answer = random.choice(
            answer_candidates
        )

        # Create fill-in-the-blank question
        question = re.sub(
            r'\b' + re.escape(answer) + r'\b',
            "_____",
            sentence,
            count=1,
            flags=re.IGNORECASE
        )

        # Generate distractors
        distractor_candidates = [
            word
            for word in unique_words
            if word.lower() != answer.lower()
        ]

        if len(distractor_candidates) < 3:

            continue

        wrong_options = random.sample(
            distractor_candidates,
            3
        )

        options = [
            answer,
            *wrong_options
        ]

        random.shuffle(options)

        quiz.append({

            "question": question,

            "options": options,

            "answer": answer

        })

    return quiz


# ============================================================
# STUDY PLAN TOOL - KNOWLEDGE BASE
# ============================================================

def create_study_plan(
    topic,
    days,
    hours_per_day
):

    days = max(
        1,
        min(days, 30)
    )

    plans = []

    topic_units = [

        "Understand the basic concepts",

        "Study important terminology",

        "Learn core principles",

        "Study practical examples",

        "Practice questions",

        "Work on a small practical exercise",

        "Revise previously studied concepts",

        "Take a self-test",

        "Identify weak areas",

        "Final revision and assessment"

    ]

    for day in range(
        1,
        days + 1
    ):

        if day == 1:

            task = (
                f"Introduction and fundamentals "
                f"of {topic}"
            )

        elif day == days:

            task = (
                f"Final revision, practice test "
                f"and review of {topic}"
            )

        else:

            task = (
                topic_units[
                    (day - 2) % len(topic_units)
                ]
                + f" related to {topic}"
            )

        plans.append({

            "day": day,

            "task": task,

            "hours": hours_per_day

        })

    return plans


# ============================================================
# STUDY PLAN TOOL - UPLOADED MATERIAL
# ============================================================

def create_pdf_study_plan(
    pdf_text,
    days,
    hours_per_day
):

    """
    Creates a study plan based on topics/headings
    found inside uploaded PDF/TXT material.
    """

    days = max(
        1,
        min(days, 30)
    )

    if not pdf_text.strip():

        return []

    # --------------------------------------------------------
    # Extract lines
    # --------------------------------------------------------

    lines = [
        line.strip()
        for line in pdf_text.splitlines()
        if line.strip()
    ]

    topics = []

    # --------------------------------------------------------
    # Detect possible headings
    # --------------------------------------------------------

    for line in lines:

        words = line.split()

        # Short lines without full stops
        # are treated as possible headings
        if (
            1 <= len(words) <= 12
            and not line.endswith(".")
        ):

            topics.append(line)

    # Remove duplicates
    topics = list(
        dict.fromkeys(topics)
    )

    # --------------------------------------------------------
    # If headings are not available,
    # extract meaningful sentences
    # --------------------------------------------------------

    if not topics:

        sentences = re.split(
            r'(?<=[.!?])\s+',
            pdf_text
        )

        topics = [

            sentence.strip()

            for sentence in sentences

            if len(
                sentence.strip().split()
            ) >= 5

        ]

    # --------------------------------------------------------
    # Final fallback
    # --------------------------------------------------------

    if not topics:

        topics = [
            "Review the uploaded study material"
        ]

    # --------------------------------------------------------
    # Create plan
    # --------------------------------------------------------

    plans = []

    for day in range(
        1,
        days + 1
    ):

        topic = topics[
            (day - 1) % len(topics)
        ]

        if day == 1:

            task = (
                f"Read and understand: {topic}"
            )

        elif day == days:

            task = (
                f"Final revision and self-test "
                f"based on: {topic}"
            )

        else:

            task = (
                f"Study and revise: {topic}"
            )

        plans.append({

            "day": day,

            "task": task,

            "hours": hours_per_day

        })

    return plans


# ============================================================
# AGENT ROUTER
# ============================================================

def classify_question(question):

    q = question.lower()

    if any(word in q for word in [
        "quiz",
        "mcq",
        "multiple choice",
        "test me"
    ]):

        return "quiz"

    if any(word in q for word in [
        "study plan",
        "study schedule",
        "timetable",
        "learning plan",
        "schedule"
    ]):

        return "study_plan"

    if any(word in q for word in [
        "previous",
        "earlier",
        "history",
        "remember",
        "last question",
        "what did i ask"
    ]):

        return "memory"

    return "rag"


# ============================================================
# RAG ANSWER
# ============================================================

def generate_rag_answer(
    question,
    context
):

    if not context:

        return (
            "I could not find relevant information in "
            "the available course material.\n\n"
            "Please upload a relevant PDF/TXT document."
        )

    sentences = re.split(
        r'(?<=[.!?])\s+',
        context
    )

    question_words = set(
        re.findall(
            r'\b[a-zA-Z]{3,}\b',
            question.lower()
        )
    )

    scored = []

    for sentence in sentences:

        words = set(
            re.findall(
                r'\b[a-zA-Z]{3,}\b',
                sentence.lower()
            )
        )

        score = len(
            question_words.intersection(words)
        )

        scored.append(
            (score, sentence)
        )

    scored.sort(
        key=lambda x: x[0],
        reverse=True
    )

    useful = [

        sentence

        for score, sentence in scored[:6]

        if score > 0

    ]

    if not useful:

        useful = sentences[:5]

    answer = " ".join(
        useful
    )

    return (
        "### 📚 Answer from Learning Material\n\n"
        + answer
        + "\n\n"
        "📌 **Source:** Uploaded course material"
    )


# ============================================================
# AGENT
# ============================================================

def run_agent(
    question,
    documents
):

    route = classify_question(
        question
    )

    # --------------------------------------------------------
    # QUIZ
    # --------------------------------------------------------

    if route == "quiz":

        topic = "Artificial Intelligence"

        for possible_topic in QUESTION_BANK:

            if possible_topic.lower() in question.lower():

                topic = possible_topic

                break

        quiz = generate_quiz(
            topic,
            5
        )

        answer = (
            f"### 📝 Quiz on {topic}\n\n"
        )

        for i, item in enumerate(
            quiz,
            start=1
        ):

            answer += (
                f"**{i}. {item['question']}**\n\n"
            )

            for j, option in enumerate(
                item["options"],
                start=1
            ):

                answer += (
                    f"{j}. {option}\n"
                )

            answer += "\n"

        return answer, "Quiz Generator Tool"

    # --------------------------------------------------------
    # STUDY PLAN
    # --------------------------------------------------------

    if route == "study_plan":

        numbers = re.findall(
            r'\d+',
            question
        )

        days = (
            int(numbers[0])
            if numbers
            else 7
        )

        topic = "Artificial Intelligence"

        for possible_topic in QUESTION_BANK:

            if possible_topic.lower() in question.lower():

                topic = possible_topic

                break

        plan = create_study_plan(
            topic,
            days,
            2
        )

        answer = (
            f"### 📅 Personalized {days}-Day Study Plan\n\n"
            f"**Topic:** {topic}\n\n"
        )

        for item in plan:

            answer += (
                f"**Day {item['day']}** — "
                f"{item['hours']} hours\n"
                f"{item['task']}\n\n"
            )

        return answer, "Study Plan Tool"

    # --------------------------------------------------------
    # MEMORY
    # --------------------------------------------------------

    if route == "memory":

        history = get_memory()

        if not history:

            return (
                "### 🧠 Learning Memory\n\n"
                "No previous learning interactions "
                "have been stored yet."
            ), "Memory Tool"

        answer = (
            "### 🧠 Your Recent Learning History\n\n"
        )

        for (
            timestamp,
            question_text,
            answer_text,
            tool
        ) in history[:10]:

            answer += (
                f"**🕐 {timestamp}**\n\n"
                f"**Question:** {question_text}\n\n"
                f"**Tool:** {tool}\n\n"
                f"**Answer:** "
                f"{answer_text[:300]}...\n\n"
                "---\n\n"
            )

        return answer, "Memory Tool"

    # --------------------------------------------------------
    # RAG
    # --------------------------------------------------------

    context = retrieve_context(
        question,
        documents
    )

    answer = generate_rag_answer(
        question,
        context
    )

    return answer, "RAG Retrieval Tool"


# ============================================================
# LOAD DEFAULT KNOWLEDGE BASE
# ============================================================

documents = []

default_file = Path(
    DEFAULT_KB
)

if default_file.exists():

    text = default_file.read_text(
        encoding="utf-8"
    )

    documents.extend(
        split_text(text)
    )


# ============================================================
# SESSION STATE
# ============================================================

if "current_quiz" not in st.session_state:

    st.session_state.current_quiz = []


if "quiz_topic" not in st.session_state:

    st.session_state.quiz_topic = ""


if "quiz_submitted" not in st.session_state:

    st.session_state.quiz_submitted = False


# NEW:
# Store uploaded document content separately
# so Quiz and Study Planner can use it.

if "uploaded_documents" not in st.session_state:

    st.session_state.uploaded_documents = []


if "uploaded_file_names" not in st.session_state:

    st.session_state.uploaded_file_names = []


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "📚 AI Learning Assistant"
)

st.sidebar.markdown(
    """
### Agent Capabilities

🤖 **Agent Router**

📚 **RAG**

🧠 **Memory**

📝 **Quiz Tool**

📅 **Study Plan Tool**

🗄️ **SQLite**
"""
)

st.sidebar.divider()


uploaded_files = st.sidebar.file_uploader(
    "📄 Upload Course Materials",
    type=["pdf", "txt"],
    accept_multiple_files=True
)


if uploaded_files:

    # Clear previous uploaded materials
    st.session_state.uploaded_documents = []

    st.session_state.uploaded_file_names = []

    for uploaded_file in uploaded_files:

        if uploaded_file.name.lower().endswith(".pdf"):

            text = read_pdf(
                uploaded_file
            )

        else:

            text = uploaded_file.read().decode(
                "utf-8",
                errors="ignore"
            )

        if text.strip():

            # Save full uploaded document
            st.session_state.uploaded_documents.append(
                text
            )

            st.session_state.uploaded_file_names.append(
                uploaded_file.name
            )

            # Add chunks to RAG documents
            documents.extend(
                split_text(text)
            )

    st.sidebar.success(
        f"{len(uploaded_files)} file(s) loaded"
    )

    for file_name in st.session_state.uploaded_file_names:

        st.sidebar.caption(
            f"📄 {file_name}"
        )


st.sidebar.divider()


st.sidebar.metric(
    "Learning Memories",
    len(get_memory())
)


# ============================================================
# HEADER
# ============================================================

st.title(
    "📚 AI Learning & Study Assistant"
)

st.markdown(
    """
**An Agentic AI assistant that helps students learn from
course materials, generate quizzes, create study plans,
and maintain learning memory.**
"""
)


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "💬 AI Assistant",
        "📝 Quiz Generator",
        "📅 Study Planner",
        "🧠 Memory"
    ]
)


# ============================================================
# TAB 1 - AI ASSISTANT
# ============================================================

with tab1:

    st.subheader(
        "💬 Ask Your AI Learning Assistant"
    )

    question = st.text_input(
        "What would you like to learn?",
        placeholder=(
            "Example: Explain RAG in simple words"
        )
    )

    st.caption(
        "The agent automatically selects RAG, "
        "Memory, Quiz or Study Plan capability."
    )

    if st.button(
        "🤖 Ask Assistant",
        type="primary"
    ):

        if question.strip():

            with st.spinner(
                "Agent is analyzing your request..."
            ):

                answer, tool_used = run_agent(
                    question,
                    documents
                )

            st.success(
                f"Agent selected: **{tool_used}**"
            )

            st.markdown(
                answer
            )

            save_memory(
                question,
                answer,
                tool_used
            )

        else:

            st.warning(
                "Please enter a question."
            )


# ============================================================
# TAB 2 - QUIZ
# ============================================================

with tab2:

    st.subheader(
        "📝 Adaptive Quiz Generator"
    )

    # --------------------------------------------------------
    # NEW SOURCE SELECTION
    # --------------------------------------------------------

    quiz_source = st.radio(
        "Choose Quiz Source",
        [
            "📚 Knowledge Base",
            "📄 Uploaded Material"
        ],
        horizontal=True
    )

    quiz_count = st.slider(
        "Number of questions",
        3,
        5,
        5
    )

    # --------------------------------------------------------
    # KNOWLEDGE BASE
    # --------------------------------------------------------

    if quiz_source == "📚 Knowledge Base":

        quiz_topic = st.selectbox(
            "Select a topic",
            list(QUESTION_BANK.keys())
        )

        if st.button(
            "🎯 Generate New Quiz"
        ):

            st.session_state.current_quiz = generate_quiz(
                quiz_topic,
                quiz_count
            )

            st.session_state.quiz_topic = quiz_topic

            st.session_state.quiz_submitted = False

    # --------------------------------------------------------
    # UPLOADED MATERIAL
    # --------------------------------------------------------

    else:

        if not st.session_state.uploaded_documents:

            st.warning(
                "📄 Please upload a PDF/TXT study material "
                "from the sidebar first."
            )

        else:

            st.success(
                f"📄 {len(st.session_state.uploaded_documents)} "
                f"uploaded material(s) available."
            )

            if st.button(
                "🎯 Generate Quiz from Uploaded Material"
            ):

                combined_text = "\n\n".join(
                    st.session_state.uploaded_documents
                )

                generated_quiz = generate_pdf_quiz(
                    combined_text,
                    quiz_count
                )

                st.session_state.current_quiz = (
                    generated_quiz
                )

                st.session_state.quiz_topic = (
                    "Uploaded Study Material"
                )

                st.session_state.quiz_submitted = False

                if not generated_quiz:

                    st.warning(
                        "Could not generate enough questions "
                        "from the uploaded material. "
                        "Please upload a text-based PDF "
                        "with sufficient content."
                    )

    # --------------------------------------------------------
    # DISPLAY QUIZ
    # --------------------------------------------------------

    quiz = st.session_state.current_quiz

    if quiz:

        st.info(
            f"Quiz Source: "
            f"{st.session_state.quiz_topic}"
        )

        for i, item in enumerate(
            quiz,
            start=1
        ):

            st.markdown(
                f"### Question {i}"
            )

            st.write(
                item["question"]
            )

            st.radio(
                "Choose your answer:",
                item["options"],
                key=f"answer_{i}",
                index=None
            )

        if st.button(
            "✅ Submit Quiz"
        ):

            score = 0

            for i, item in enumerate(
                quiz,
                start=1
            ):

                selected = st.session_state.get(
                    f"answer_{i}"
                )

                if selected == item["answer"]:

                    score += 1

            st.session_state.quiz_submitted = True

            percentage = (
                score / len(quiz)
            ) * 100

            st.success(
                f"🎉 Your Score: "
                f"**{score}/{len(quiz)} "
                f"({percentage:.0f}%)**"
            )

            for i, item in enumerate(
                quiz,
                start=1
            ):

                st.write(
                    f"**Q{i} Correct Answer:** "
                    f"{item['answer']}"
                )


# ============================================================
# TAB 3 - STUDY PLANNER
# ============================================================

with tab3:

    st.subheader(
        "📅 Personalized Study Planner"
    )

    # --------------------------------------------------------
    # NEW SOURCE SELECTION
    # --------------------------------------------------------

    study_source = st.radio(
        "Choose Study Plan Source",
        [
            "📚 Knowledge Base",
            "📄 Uploaded Material"
        ],
        horizontal=True
    )

    study_days = st.slider(
        "Study duration (days)",
        1,
        30,
        7
    )

    hours = st.slider(
        "Study hours per day",
        1,
        8,
        2
    )

    # --------------------------------------------------------
    # KNOWLEDGE BASE STUDY PLAN
    # --------------------------------------------------------

    if study_source == "📚 Knowledge Base":

        study_topic = st.selectbox(
            "Choose your study topic",
            list(QUESTION_BANK.keys()),
            key="planner_topic"
        )

        if st.button(
            "📅 Generate My Study Plan"
        ):

            plan = create_study_plan(
                study_topic,
                study_days,
                hours
            )

            st.success(
                "Personalized study plan created!"
            )

            plan_text = (
                f"AI Learning & Study Assistant\n"
                f"Study Plan: {study_topic}\n\n"
            )

            for item in plan:

                st.markdown(
                    f"### 📌 Day {item['day']}"
                )

                st.write(
                    item["task"]
                )

                st.caption(
                    f"Recommended study time: "
                    f"{item['hours']} hour(s)"
                )

                plan_text += (
                    f"Day {item['day']}: "
                    f"{item['task']} "
                    f"({item['hours']} hours)\n"
                )

            st.download_button(
                "⬇️ Download Study Plan",
                data=plan_text,
                file_name="study_plan.txt",
                mime="text/plain"
            )

    # --------------------------------------------------------
    # UPLOADED MATERIAL STUDY PLAN
    # --------------------------------------------------------

    else:

        if not st.session_state.uploaded_documents:

            st.warning(
                "📄 Please upload a PDF/TXT study material "
                "from the sidebar first."
            )

        else:

            st.success(
                f"📄 {len(st.session_state.uploaded_documents)} "
                f"uploaded material(s) available."
            )

            if st.button(
                "📅 Generate Study Plan from Uploaded Material"
            ):

                combined_text = "\n\n".join(
                    st.session_state.uploaded_documents
                )

                plan = create_pdf_study_plan(
                    combined_text,
                    study_days,
                    hours
                )

                if not plan:

                    st.warning(
                        "Could not create a study plan "
                        "from the uploaded material."
                    )

                else:

                    st.success(
                        "Study plan created from "
                        "uploaded material!"
                    )

                    plan_text = (
                        "AI Learning & Study Assistant\n"
                        "Study Plan: Uploaded Material\n\n"
                    )

                    for item in plan:

                        st.markdown(
                            f"### 📌 Day {item['day']}"
                        )

                        st.write(
                            item["task"]
                        )

                        st.caption(
                            f"Recommended study time: "
                            f"{item['hours']} hour(s)"
                        )

                        plan_text += (
                            f"Day {item['day']}: "
                            f"{item['task']} "
                            f"({item['hours']} hours)\n"
                        )

                    st.download_button(
                        "⬇️ Download Study Plan",
                        data=plan_text,
                        file_name="uploaded_material_study_plan.txt",
                        mime="text/plain"
                    )


# ============================================================
# TAB 4 - MEMORY
# ============================================================

with tab4:

    st.subheader(
        "🧠 Long-Term Learning Memory"
    )

    st.write(
        "The assistant stores previous learning "
        "interactions in SQLite."
    )

    history = get_memory()

    if history:

        st.metric(
            "Total Stored Interactions",
            len(history)
        )

        for (
            timestamp,
            question_text,
            answer_text,
            tool
        ) in history:

            with st.expander(
                f"🕐 {timestamp} — {tool}"
            ):

                st.markdown(
                    f"**Question:** {question_text}"
                )

                st.markdown(
                    "**Answer:**"
                )

                st.write(
                    answer_text[:1000]
                )

    else:

        st.info(
            "No learning history yet. "
            "Ask the assistant a question first."
        )

    st.divider()

    if st.button(
        "🗑️ Clear Learning Memory"
    ):

        clear_memory()

        st.success(
            "Learning memory cleared."
        )

        st.rerun()


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI Learning & Study Assistant | "
    "Agentic AI • RAG • Memory • Tools • Streamlit • SQLite"
)
