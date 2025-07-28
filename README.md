# 📄 Persona-Driven Document Intelligence System

*Adobe India Hackathon 2025 — Round 1B Submission*
**"Connect What Matters — For the User Who Matters"**

A smart document analysis system that extracts and ranks relevant content from PDFs based on user roles and tasks.

---

## 🚀 Overview

This system works like a personal document analyst:

* Processes 3–10 PDFs
* Understands *who* the user is (persona)
* Knows *what* the user wants to achieve (job-to-be-done)
* Extracts and ranks *only the most relevant content*

---

## 🔍 Key Capabilities

### ✅ Multi-Type PDF Analysis

* Supports research papers, financial reports, textbooks, manuals
* Adapts to academic, business, educational, and technical formats

### 🎯 Persona-Based Extraction

* Tailors content for roles like *researchers, **students, **analysts, **entrepreneurs*
* Prioritizes sections based on the user’s task

### 🧠 Intelligent Processing

* Uses *sentence transformers, **TF-IDF, and **keyword matching*
* Ranks and refines extracted content based on document type and job requirements

---

## 🧪 Supported Use Cases

| Test Case           | Persona              | Task                                         |
| ------------------- | -------------------- | -------------------------------------------- |
| Academic Research   | PhD Researcher       | Literature review on GNNs for drug discovery |
| Business Analysis   | Investment Analyst   | Compare revenue, R\&D, and strategies        |
| Educational Support | UG Chemistry Student | Extract key concepts on reaction kinetics    |

---

## 🛠 Setup & Installation

### 🔧 Requirements

bash
Python 3.8+
pip install -r requirements.txt


### 📦 Dependencies

bash
pip install PyPDF2 sentence-transformers scikit-learn numpy


### 🚀 Run

bash
git clone https://github.com/jhaaj08/Adobe-India_Hackathon25.git
cd Adobe-India_Hackathon25
python enhanced_doc_intelligence.py


---

## 📤 Input Format (JSON)

json
{
  "challenge_info": {
    "test_case_name": "academic_research"
  },
  "documents": [{ "filename": "paper1.pdf", "title": "Paper Title" }],
  "persona": { "role": "PhD Researcher" },
  "job_to_be_done": { "task": "Literature review" }
}


## 📥 Output Format (JSON)

json
{
  "metadata": { "persona": "PhD Researcher", "job_to_be_done": "Literature review" },
  "extracted_sections": [
    { "document": "paper1.pdf", "section_title": "Methodology", "importance_rank": 1, "page_number": 3 }
  ],
  "subsection_analysis": [
    { "document": "paper1.pdf", "refined_text": "Deep analysis text...", "page_number": 3 }
  ]
}


---

## 🐳 Docker (Optional)

bash
docker build -t persona-doc-intelligence .
docker run -v $(pwd)/data:/app/data persona-doc-intelligence


---

## ⚙ Performance

* *CPU-only*: No GPU required
* *Model size*: Under 1GB
* *Speed*: Processes 3–5 PDFs in ≤ 60 seconds
* *Offline*: Works without internet

---

## 🧠 Inside the System

### Content Matching Strategy

* *Semantic Analysis* using sentence-transformers
* *TF-IDF & Keywords* for statistical + contextual relevance
* *Document Type Optimization* for domain-specific extraction
* *Persona + Task Matching* for precision ranking

---

## ❗ Limitations

* Focused on *PDFs only*
* Designed for *CPU environments*
* No support for *real-time online search*

---

## 📬 Contact

* GitHub: [Adobe-India\_Hackathon25](https://github.com/jhaaj08/Adobe-India_Hackathon25)
* Issues & Suggestions → Open a GitHub Issue

---

## 📄 License

Built for Adobe India Hackathon 2025
*"Connect What Matters — For the User Who Matters"*
