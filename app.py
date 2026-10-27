import streamlit as st
import pandas as pd
import re
import joblib
import os
from datetime import datetime

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Intelligent Software Bug Reporter",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM WHITE + PURPLE DESIGN
# ============================================================

st.markdown("""
<style>

.stApp {
    background: #ffffff;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}

.main-title {
    font-size: 42px;
    font-weight: 700;
    color: #5B21B6;
    margin-bottom: 5px;
    letter-spacing: -1px;
}

.subtitle {
    font-size: 17px;
    color: #6B7280;
    margin-bottom: 25px;
}

.purple-line {
    height: 4px;
    width: 80px;
    background: linear-gradient(
        90deg,
        #7C3AED,
        #A855F7
    );
    border-radius: 10px;
    margin-bottom: 30px;
}

h1, h2, h3 {
    color: #4C1D95 !important;
}

p, label {
    color: #374151 !important;
}

textarea {
    border: 2px solid #DDD6FE !important;
    border-radius: 12px !important;
    background-color: #FFFFFF !important;
    color: #1F2937 !important;
    font-size: 16px !important;
}

textarea:focus {
    border-color: #7C3AED !important;
    box-shadow: 0 0 0 2px #EDE9FE !important;
}

.stButton > button {
    background: linear-gradient(
        135deg,
        #7C3AED,
        #6D28D9
    ) !important;

    color: white !important;
    border: none !important;
    border-radius: 10px !important;

    padding: 10px 28px !important;

    font-size: 16px !important;
    font-weight: 600 !important;
}

.stButton > button:hover {
    background: linear-gradient(
        135deg,
        #6D28D9,
        #5B21B6
    ) !important;
}

[data-testid="stMetric"] {
    background: #FAF9FF;
    border: 1px solid #E9D5FF;
    border-radius: 14px;
    padding: 18px;
    box-shadow: 0 3px 12px rgba(91, 33, 182, 0.07);
}

[data-testid="stMetricLabel"] {
    color: #6B7280 !important;
}

[data-testid="stMetricValue"] {
    color: #5B21B6 !important;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #FAF9FF 0%,
        #F5F3FF 100%
    );

    border-right: 1px solid #E9D5FF;
}

section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #5B21B6 !important;
}

hr {
    border-color: #E9D5FF !important;
}

.stDownloadButton > button {
    background: white !important;
    color: #6D28D9 !important;
    border: 2px solid #7C3AED !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
}

.stDownloadButton > button:hover {
    background: #F5F3FF !important;
}

.report-card {
    background: #FAF9FF;
    border: 1px solid #E9D5FF;
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 15px;
}

.priority-highest {
    color: #991B1B;
    font-weight: 700;
}

.priority-high {
    color: #C2410C;
    font-weight: 700;
}

.priority-medium {
    color: #A16207;
    font-weight: 700;
}

.priority-low {
    color: #15803D;
    font-weight: 700;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# FILE SETTINGS
# ============================================================

REPORT_FILE = "reports.csv"


# ============================================================
# LOAD MODELS
# ============================================================

try:

    type_model = joblib.load(
        "bug_type_model.pkl"
    )

    severity_model = joblib.load(
        "severity_model.pkl"
    )

    bug_data = pd.read_csv(
        "bugs.csv"
    )

except Exception as e:

    st.error(
        "Unable to load the project files."
    )

    st.code(
        str(e)
    )

    st.stop()


# ============================================================
# REPORT STORAGE
# ============================================================

REPORT_COLUMNS = [
    "bug_id",
    "date",
    "description",
    "bug_type",
    "severity",
    "priority",
    "developer_priority",
    "keywords",
    "duplicate_similarity",
    "status"
]


def create_report_file():

    if not os.path.exists(REPORT_FILE):

        empty_data = pd.DataFrame(
            columns=REPORT_COLUMNS
        )

        empty_data.to_csv(
            REPORT_FILE,
            index=False
        )

    else:

        # Check whether the existing file is empty
        if os.path.getsize(REPORT_FILE) == 0:

            empty_data = pd.DataFrame(
                columns=REPORT_COLUMNS
            )

            empty_data.to_csv(
                REPORT_FILE,
                index=False
            )


def load_reports():

    create_report_file()

    try:

        reports = pd.read_csv(
            REPORT_FILE
        )

    except (
        pd.errors.EmptyDataError,
        FileNotFoundError
    ):

        reports = pd.DataFrame(
            columns=REPORT_COLUMNS
        )

        reports.to_csv(
            REPORT_FILE,
            index=False
        )

    return reports



def save_report(report):

    create_report_file()

    reports = load_reports()

    new_report = pd.DataFrame(
        [report]
    )

    reports = pd.concat(
        [reports, new_report],
        ignore_index=True
    )

    reports.to_csv(
        REPORT_FILE,
        index=False
    )


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    text = text.lower()

    text = re.sub(
        r"[^a-zA-Z\s]",
        "",
        text
    )

    words = text.split()

    stop_words = {
        "the",
        "is",
        "a",
        "an",
        "to",
        "and",
        "in",
        "on",
        "of",
        "for",
        "when",
        "with",
        "this",
        "that",
        "it",
        "my"
    }

    words = [
        word
        for word in words
        if word not in stop_words
    ]

    return words


# ============================================================
# KEYWORD EXTRACTION
# ============================================================

def extract_keywords(text):

    words = clean_text(text)

    keywords = []

    for word in words:

        if len(word) > 3 and word not in keywords:

            keywords.append(word)

    return keywords[:8]


# ============================================================
# DUPLICATE DETECTION
# ============================================================

def find_duplicate(text):

    descriptions = bug_data[
        "bug_description"
    ].tolist()

    vectorizer = TfidfVectorizer(
        stop_words="english"
    )

    all_text = descriptions + [text]

    vectors = vectorizer.fit_transform(
        all_text
    )

    similarity = cosine_similarity(
        vectors[-1],
        vectors[:-1]
    )[0]

    index = similarity.argmax()

    return (
        descriptions[index],
        similarity[index]
    )


# ============================================================
# PRIORITY FUNCTIONS
# ============================================================

def get_priority(severity):

    if severity in [
        "Critical",
        "High"
    ]:

        return "High"

    elif severity == "Medium":

        return "Medium"

    return "Low"


def get_developer_priority(
    severity,
    bug_type
):

    if severity == "Critical":

        if bug_type in [
            "Security",
            "Crash"
        ]:

            return "Highest"

        return "High"

    elif severity == "High":

        return "High"

    elif severity == "Medium":

        return "Medium"

    return "Low"


def priority_number(priority):

    order = {
        "Highest": 1,
        "High": 2,
        "Medium": 3,
        "Low": 4
    }

    return order.get(
        priority,
        5
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    'Intelligent Software Bug Reporter'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'NLP and Machine Learning based Software Bug Analysis System'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="purple-line"></div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## Navigation")

    page = st.radio(
        "Select View",
        [
            "Report Bug",
            "Developer Dashboard"
        ]
    )

    st.divider()

    st.markdown("## About the Project")

    st.write(
        "This system uses Natural Language Processing "
        "and Machine Learning to analyze software bug "
        "reports automatically."
    )

    st.divider()

    st.markdown("### Technologies")

    st.write("Python")
    st.write("Natural Language Processing")
    st.write("TF-IDF")
    st.write("Logistic Regression")
    st.write("Scikit-learn")
    st.write("Streamlit")
    st.write("Cosine Similarity")


# ============================================================
# PAGE 1 — REPORT BUG
# ============================================================

if page == "Report Bug":

    st.markdown(
        "## Report a Software Bug"
    )

    bug_description = st.text_area(
        "Describe the problem",
        height=170,
        placeholder=(
            "Example: When I click the login button, "
            "the application crashes and shows a blank screen."
        )
    )

    st.write("")

    analyze = st.button(
        "Analyze Bug"
    )


    # ========================================================
    # ANALYZE BUG
    # ========================================================

    if analyze:

        if not bug_description.strip():

            st.warning(
                "Please enter a bug description."
            )

        else:

            # ----------------------------------------------
            # MACHINE LEARNING PREDICTIONS
            # ----------------------------------------------

            bug_type = type_model.predict(
                [bug_description]
            )[0]

            severity = severity_model.predict(
                [bug_description]
            )[0]

            priority = get_priority(
                severity
            )

            developer_priority = get_developer_priority(
                severity,
                bug_type
            )

            keywords = extract_keywords(
                bug_description
            )

            duplicate_bug, similarity = find_duplicate(
                bug_description
            )

            bug_id = "BUG-" + str(
                abs(hash(
                    bug_description +
                    str(datetime.now())
                )) % 10000
            )


            # ----------------------------------------------
            # STORE REPORT
            # ----------------------------------------------

            report = {
                "bug_id": bug_id,

                "date": datetime.now().strftime(
                    "%Y-%m-%d %H:%M"
                ),

                "description": bug_description,

                "bug_type": bug_type,

                "severity": severity,

                "priority": priority,

                "developer_priority":
                    developer_priority,

                "keywords":
                    ", ".join(keywords),

                "duplicate_similarity":
                    round(
                        similarity * 100,
                        2
                    ),

                "status": "New"
            }

            save_report(
                report
            )


            # ----------------------------------------------
            # SUCCESS
            # ----------------------------------------------

            st.success(
                "Bug analyzed and report saved successfully."
            )

            st.markdown(
                "## Analysis Results"
            )


            # ----------------------------------------------
            # RESULT CARDS
            # ----------------------------------------------

            col1, col2, col3, col4 = st.columns(4)

            with col1:

                st.metric(
                    "Bug ID",
                    bug_id
                )

            with col2:

                st.metric(
                    "Bug Type",
                    bug_type
                )

            with col3:

                st.metric(
                    "Severity",
                    severity
                )

            with col4:

                st.metric(
                    "Developer Priority",
                    developer_priority
                )


            # ----------------------------------------------
            # KEYWORDS
            # ----------------------------------------------

            st.divider()

            st.markdown(
                "### Extracted Keywords"
            )

            if keywords:

                st.write(
                    ", ".join(keywords)
                )

            else:

                st.write(
                    "No keywords found."
                )


            # ----------------------------------------------
            # DUPLICATE DETECTION
            # ----------------------------------------------

            st.markdown(
                "### Duplicate Bug Detection"
            )

            st.write(
                f"Similarity Score: "
                f"{similarity * 100:.2f}%"
            )

            if similarity >= 0.70:

                st.warning(
                    "Possible duplicate bug found."
                )

                st.write(
                    "Similar bug:"
                )

                st.info(
                    duplicate_bug
                )

            else:

                st.success(
                    "No highly similar bug found."
                )


            # ----------------------------------------------
            # NLP PROCESSING
            # ----------------------------------------------

            st.divider()

            st.markdown(
                "### NLP Processing"
            )

            cleaned_words = clean_text(
                bug_description
            )

            st.write(
                "Processed Tokens"
            )

            st.code(
                " ".join(cleaned_words)
            )


            # ----------------------------------------------
            # FINAL REPORT
            # ----------------------------------------------

            st.divider()

            st.markdown(
                "### Generated Bug Report"
            )

            report_text = f"""
BUG REPORT
==========

Bug ID:
{bug_id}

Date:
{report["date"]}

Description:
{bug_description}

Bug Type:
{bug_type}

Severity:
{severity}

Priority:
{priority}

Developer Priority:
{developer_priority}

Keywords:
{", ".join(keywords)}

Duplicate Similarity:
{similarity * 100:.2f}%

Status:
New
"""

            st.code(
                report_text,
                language="text"
            )


            # ----------------------------------------------
            # DOWNLOAD
            # ----------------------------------------------

            st.download_button(
                "Download Bug Report",
                report_text,
                file_name=f"{bug_id}.txt",
                mime="text/plain"
            )


# ============================================================
# PAGE 2 — DEVELOPER DASHBOARD
# ============================================================

else:

    st.markdown(
        "## Developer Dashboard"
    )

    st.write(
        "View, prioritize and manage submitted software bug reports."
    )

    st.divider()


    # ========================================================
    # LOAD REPORTS
    # ========================================================

    reports = load_reports()


    # ========================================================
    # DASHBOARD METRICS
    # ========================================================

    total_reports = len(
        reports
    )

    if total_reports > 0:

        highest_count = len(
            reports[
                reports[
                    "developer_priority"
                ] == "Highest"
            ]
        )

        high_count = len(
            reports[
                reports[
                    "developer_priority"
                ] == "High"
            ]
        )

        medium_count = len(
            reports[
                reports[
                    "developer_priority"
                ] == "Medium"
            ]
        )

        open_count = len(
            reports[
                reports[
                    "status"
                ] != "Resolved"
            ]
        )

    else:

        highest_count = 0
        high_count = 0
        medium_count = 0
        open_count = 0


    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Total Reports",
            total_reports
        )

    with col2:

        st.metric(
            "Highest Priority",
            highest_count
        )

    with col3:

        st.metric(
            "High Priority",
            high_count
        )

    with col4:

        st.metric(
            "Open Reports",
            open_count
        )


    st.divider()


    # ========================================================
    # NO REPORTS
    # ========================================================

    if reports.empty:

        st.info(
            "No bug reports have been submitted yet."
        )

    else:

        # ====================================================
        # FILTERS
        # ====================================================

        st.markdown(
            "### Filter Reports"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            priority_filter = st.selectbox(
                "Developer Priority",
                [
                    "All",
                    "Highest",
                    "High",
                    "Medium",
                    "Low"
                ]
            )

        with col2:

            severity_filter = st.selectbox(
                "Severity",
                [
                    "All",
                    "Critical",
                    "High",
                    "Medium",
                    "Low"
                ]
            )

        with col3:

            status_filter = st.selectbox(
                "Status",
                [
                    "All",
                    "New",
                    "In Progress",
                    "Resolved"
                ]
            )


        # ====================================================
        # APPLY FILTERS
        # ====================================================

        filtered = reports.copy()


        if priority_filter != "All":

            filtered = filtered[
                filtered[
                    "developer_priority"
                ] == priority_filter
            ]


        if severity_filter != "All":

            filtered = filtered[
                filtered[
                    "severity"
                ] == severity_filter
            ]


        if status_filter != "All":

            filtered = filtered[
                filtered[
                    "status"
                ] == status_filter
            ]


        # ====================================================
        # SORT BY DEVELOPER PRIORITY
        # ====================================================

        filtered["priority_order"] = (
            filtered[
                "developer_priority"
            ].map(
                priority_number
            )
        )

        filtered = filtered.sort_values(
            by=[
                "priority_order",
                "date"
            ],
            ascending=[
                True,
                False
            ]
        )


        st.divider()

        st.markdown(
            "### Reports Requiring Attention"
        )


        # ====================================================
        # REPORT TABLE
        # ====================================================

        display_data = filtered[
            [
                "bug_id",
                "date",
                "bug_type",
                "severity",
                "developer_priority",
                "status"
            ]
        ].copy()

        display_data.columns = [
            "Bug ID",
            "Date",
            "Bug Type",
            "Severity",
            "Developer Priority",
            "Status"
        ]

        st.dataframe(
            display_data,
            use_container_width=True,
            hide_index=True
        )


        # ====================================================
        # SELECT REPORT
        # ====================================================

        if not filtered.empty:

            st.divider()

            st.markdown(
                "### Manage Report"
            )

            selected_bug = st.selectbox(
                "Select a bug report",
                filtered[
                    "bug_id"
                ].tolist()
            )


            selected_index = reports[
                reports[
                    "bug_id"
                ] == selected_bug
            ].index[0]


            selected_report = reports.loc[
                selected_index
            ]


            # =================================================
            # REPORT DETAILS
            # =================================================

            st.markdown(
                "### Report Details"
            )

            col1, col2 = st.columns(2)

            with col1:

                st.write(
                    "**Bug ID:**",
                    selected_report["bug_id"]
                )

                st.write(
                    "**Bug Type:**",
                    selected_report["bug_type"]
                )

                st.write(
                    "**Severity:**",
                    selected_report["severity"]
                )

                st.write(
                    "**Developer Priority:**",
                    selected_report[
                        "developer_priority"
                    ]
                )

            with col2:

                st.write(
                    "**Date:**",
                    selected_report["date"]
                )

                st.write(
                    "**Priority:**",
                    selected_report["priority"]
                )

                st.write(
                    "**Current Status:**",
                    selected_report["status"]
                )

                st.write(
                    "**Duplicate Similarity:**",
                    f'{selected_report["duplicate_similarity"]}%'
                )


            st.markdown(
                "#### Bug Description"
            )

            st.info(
                selected_report["description"]
            )


            st.markdown(
                "#### Keywords"
            )

            st.write(
                selected_report["keywords"]
            )


            # =================================================
            # STATUS MANAGEMENT
            # =================================================

            st.markdown(
                "### Update Status"
            )

            current_status = selected_report[
                "status"
            ]

            status_options = [
                "New",
                "In Progress",
                "Resolved"
            ]

            new_status = st.selectbox(
                "Report Status",
                status_options,
                index=status_options.index(
                    current_status
                )
            )


            if st.button(
                "Update Report Status"
            ):

                reports.loc[
                    selected_index,
                    "status"
                ] = new_status

                reports.to_csv(
                    REPORT_FILE,
                    index=False
                )

                st.success(
                    f"Report {selected_bug} "
                    f"updated to {new_status}."
                )

                st.rerun()


            # =================================================
            # DOWNLOAD SELECTED REPORT
            # =================================================

            report_download = f"""
BUG REPORT
==========

Bug ID:
{selected_report["bug_id"]}

Date:
{selected_report["date"]}

Description:
{selected_report["description"]}

Bug Type:
{selected_report["bug_type"]}

Severity:
{selected_report["severity"]}

Priority:
{selected_report["priority"]}

Developer Priority:
{selected_report["developer_priority"]}

Keywords:
{selected_report["keywords"]}

Duplicate Similarity:
{selected_report["duplicate_similarity"]}%

Status:
{new_status}
"""

            st.download_button(
                "Download Selected Report",
                report_download,
                file_name=f'{selected_report["bug_id"]}.txt',
                mime="text/plain"
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown("""
<div style="
    text-align: center;
    padding: 25px 0 10px 0;
    color: #6B7280;
    font-size: 13px;
">
    <strong style="color: #5B21B6;">
        Intelligent Software Bug Reporter
    </strong>
    <br>
    Natural Lauguage Processing -Based system
    <br>
    © 2026 Rutuja Sunil Bhosale. All Rights Reserved.
</div>
""", unsafe_allow_html=True)