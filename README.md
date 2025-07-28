# Persona-Driven Document Intelligence System

## Round 1B: "Connect What Matters — For the User Who Matters"

A sophisticated document analysis system that intelligently extracts and prioritizes the most relevant sections from PDF collections based on specific personas and their job-to-be-done tasks.

## 🎯 Challenge Overview

This system acts as an intelligent document analyst, capable of:
- Processing diverse document collections (3-10 PDFs)
- Understanding different persona types and their expertise areas
- Extracting relevant content based on specific job-to-be-done tasks
- Providing ranked, refined analysis tailored to user needs

## 🚀 Key Features

### Universal Document Processing
- *Multi-domain Support*: Research papers, financial reports, textbooks, technical manuals
- *Adaptive Content Recognition*: Automatic detection of academic, business, educational, and technical content
- *Intelligent Segmentation*: Context-aware text parsing based on document type

### Persona-Driven Analysis
- *Role-Specific Processing*: Tailored extraction for researchers, students, analysts, entrepreneurs
- *Task-Oriented Ranking*: Content prioritization based on job-to-be-done requirements
- *Expertise-Aware Refinement*: Content filtering matching persona expertise levels

### Advanced Intelligence Features
- *Semantic Understanding*: Uses sentence transformers for deep content comprehension
- *Multi-Method Scoring*: Combines TF-IDF, semantic similarity, and keyword matching
- *Content Type Optimization*: Specialized processing pipelines for different document types

## 📋 Test Cases Supported

### Test Case 1: Academic Research
- *Documents*: 4 research papers on "Graph Neural Networks for Drug Discovery"
- *Persona*: PhD Researcher in Computational Biology
- *Job*: Literature review focusing on methodologies, datasets, and performance benchmarks

### Test Case 2: Business Analysis
- *Documents*: 3 annual reports from competing tech companies
- *Persona*: Investment Analyst
- *Job*: Analyze revenue trends, R&D investments, and market positioning strategies

### Test Case 3: Educational Content
- *Documents*: 5 organic chemistry textbook chapters
- *Persona*: Undergraduate Chemistry Student
- *Job*: Identify key concepts and mechanisms for exam preparation on reaction kinetics

## 🛠 Installation & Setup

### Prerequisites
bash
python >= 3.8
pip install -r requirements.txt


### Dependencies
bash
pip install PyPDF2 sentence-transformers scikit-learn numpy


### Quick Start
1. *Clone Repository*
   bash
   git clone https://github.com/jhaaj08/Adobe-India_Hackathon25.git
   cd Adobe-India_Hackathon25
   

2. *Install Dependencies*
   bash
   pip install -r requirements.txt
   

3. *Prepare Documents*
   - Place PDF files in the project directory or create subdirectories (PDFs/, documents/)
   - The system automatically searches multiple locations

4. *Run Analysis*
   bash
   python enhanced_doc_intelligence.py
   

## 📁 Project Structure


├── enhanced_doc_intelligence.py    # Main processing system
├── requirements.txt               # Dependencies
├── README.md                     # This file
├── approach_explanation.md       # Methodology documentation
├── Dockerfile                    # Container setup
├── Collection 1/                 # Academic research test case
│   ├── challenge1b_input.json
│   └── challenge1b_output.json
├── Collection 2/                 # Business analysis test case
│   ├── challenge1b_input.json
│   └── challenge1b_output.json
└── Collection 3/                 # Educational content test case
    ├── challenge1b_input.json
    └── challenge1b_output.json


## 🔧 Configuration

### Input Format (challenge1b_input.json)
json
{
  "challenge_info": {
    "challenge_id": "round_1b_001",
    "test_case_name": "academic_research",
    "description": "Graph Neural Networks for Drug Discovery Literature Review"
  },
  "documents": [
    {
      "filename": "research_paper.pdf",
      "title": "Paper Title"
    }
  ],
  "persona": {
    "role": "PhD Researcher in Computational Biology"
  },
  "job_to_be_done": {
    "task": "Prepare comprehensive literature review"
  }
}


### Output Format (challenge1b_output.json)
json
{
  "metadata": {
    "input_documents": ["doc1.pdf", "doc2.pdf"],
    "persona": "PhD Researcher in Computational Biology",
    "job_to_be_done": "Prepare comprehensive literature review",
    "processing_timestamp": "2025-01-XX"
  },
  "extracted_sections": [
    {
      "document": "research_paper.pdf",
      "section_title": "Methodology Section",
      "importance_rank": 1,
      "page_number": 3
    }
  ],
  "subsection_analysis": [
    {
      "document": "research_paper.pdf",
      "refined_text": "Detailed analysis text...",
      "page_number": 3
    }
  ]
}


## 🐳 Docker Deployment

### Build Container
bash
docker build -t persona-doc-intelligence .


### Run Container
bash
docker run -v $(pwd)/data:/app/data persona-doc-intelligence


## ⚡ Performance Specifications

- *CPU Only*: No GPU requirements
- *Model Size*: ≤ 1GB (using efficient sentence transformers)
- *Processing Time*: ≤ 60 seconds for 3-5 documents
- *Offline Operation*: No internet access required during execution

## 🎯 Scoring Criteria

| Criteria | Max Points | Description |
|----------|------------|-------------|
| *Section Relevance* | 60 | Quality of section matching persona + job requirements with proper ranking |
| *Sub-Section Relevance* | 40 | Granular subsection extraction and ranking quality |

## 🧠 System Architecture

### Content Type Detection
- *Academic*: Methodologies, datasets, benchmarks, research findings
- *Business*: Financial metrics, strategies, market analysis, competitive positioning
- *Educational*: Key concepts, mechanisms, exam-relevant material
- *Technical*: Procedures, instructions, feature descriptions
- *Recipe*: Ingredients, instructions, cooking methods
- *Travel*: Destinations, recommendations, cultural information

### Relevance Scoring Algorithm
1. *Semantic Similarity*: Deep understanding using sentence transformers
2. *TF-IDF Analysis*: Statistical relevance measurement
3. *Keyword Matching*: Persona and task-specific term identification
4. *Content Type Scoring*: Domain-specific relevance calculation
5. *Task Bonuses*: Job-specific content prioritization
6. *Structure Analysis*: Document organization and length considerations

### Content Refinement Pipeline
- *Academic*: Preserves methodological details and performance metrics
- *Business*: Focuses on quantitative data and strategic insights
- *Educational*: Emphasizes concepts and exam preparation material
- *Technical*: Highlights procedural information and instructions

## 🔍 Advanced Features

### Intelligent Text Processing
- *Unicode Normalization*: Handles special characters and formatting
- *Pattern Recognition*: Domain-specific content identification
- *Semantic Segmentation*: Context-aware text chunking
- *Quality Filtering*: Substantial content validation

### Adaptive Algorithms
- *Dynamic Weighting*: Content-type specific scoring adjustments
- *Multi-criteria Optimization*: Balanced relevance assessment
- *Persona Matching*: Role-specific content prioritization
- *Task Alignment*: Job-focused section ranking

## 📊 Usage Examples

### Academic Research Analysis
python
# Processes research papers for literature reviews
# Focuses on methodologies, datasets, and benchmarks
# Optimized for PhD-level content extraction


### Business Intelligence
python
# Analyzes annual reports and financial documents
# Extracts revenue trends and strategic insights
# Tailored for investment analysis workflows


### Educational Support
python
# Processes textbooks and study materials
# Identifies key concepts and exam-relevant content
# Optimized for student learning objectives


## 🛡 Constraints & Limitations

- *CPU-Only Processing*: Optimized for standard hardware
- *Model Size Limit*: Uses efficient transformers within 1GB constraint
- *Processing Time*: Designed for real-time analysis (≤60 seconds)
- *Offline Operation*: No external API dependencies
- *PDF Format*: Primary focus on PDF document processing

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (git checkout -b feature/amazing-feature)
3. Commit changes (git commit -m 'Add amazing feature')
4. Push to branch (git push origin feature/amazing-feature)
5. Open Pull Request

## 📄 License

This project is developed for Adobe India Hackathon 2025 - Round 1B.
