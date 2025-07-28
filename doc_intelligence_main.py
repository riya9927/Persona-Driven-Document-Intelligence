import json
import os
import time
from datetime import datetime
from typing import List, Dict, Any, Tuple
import re
from pathlib import Path

# Using lightweight libraries for CPU-only processing
try:
    import PyPDF2
    from sentence_transformers import SentenceTransformer
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError as e:
    print(f"Missing required library: {e}")
    print("Install with: pip install PyPDF2 sentence-transformers scikit-learn numpy")
    exit(1)


class DocumentIntelligenceSystem:
    def _init_(self):
        """Initialize the system with lightweight models for CPU processing"""
        print("Initializing Document Intelligence System...")
        
        # Use a small, efficient sentence transformer model (<500MB)
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.tfidf = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
    def extract_text_from_pdf(self, pdf_path: str) -> Dict[int, str]:
        """Extract text from PDF with page-wise organization"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                pages_text = {}
                
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    try:
                        text = page.extract_text()
                        if text.strip():
                            pages_text[page_num] = text.strip()
                    except Exception as e:
                        print(f"Error extracting page {page_num} from {pdf_path}: {e}")
                        continue
                        
                return pages_text
        except Exception as e:
            print(f"Error reading PDF {pdf_path}: {e}")
            return {}
    
    def segment_text_into_sections(self, text: str, page_num: int) -> List[Dict]:
        """Segment text into logical sections with titles"""
        sections = []
        
        # Split by common section indicators
        section_patterns = [
            r'\n\s*(?:CHAPTER|Chapter)\s+\d+[:\.]?\s*([^\n]+)',
            r'\n\s*(?:SECTION|Section)\s+\d+[:\.]?\s*([^\n]+)',
            r'\n\s*\d+\.\s+([A-Z][^\n]+)',
            r'\n\s*([A-Z][A-Z\s]{10,})\s*\n',
            r'\n\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+))\s\n(?=[A-Z])'
        ]
        
        # Try to find sections using patterns
        found_sections = []
        for pattern in section_patterns:
            matches = list(re.finditer(pattern, text))
            if matches:
                found_sections.extend([(m.start(), m.group(1).strip()) for m in matches])
        
        if found_sections:
            # Sort by position and create sections
            found_sections.sort()
            for i, (start_pos, title) in enumerate(found_sections):
                end_pos = found_sections[i + 1][0] if i + 1 < len(found_sections) else len(text)
                content = text[start_pos:end_pos].strip()
                
                if len(content) > 100:  # Only include substantial sections
                    sections.append({
                        'title': title[:100],  # Limit title length
                        'content': content,
                        'page_number': page_num,
                        'start_pos': start_pos
                    })
        else:
            # Fallback: split by paragraphs for documents without clear sections
            paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 50]
            for i, para in enumerate(paragraphs[:10]):  # Limit to first 10 paragraphs
                # Use first line as title
                lines = para.split('\n')
                title = lines[0][:50] + "..." if len(lines[0]) > 50 else lines[0]
                
                sections.append({
                    'title': title,
                    'content': para,
                    'page_number': page_num,
                    'start_pos': i * 1000  # Approximate position
                })
        
        return sections
    
    def calculate_relevance_score(self, section_content: str, persona: str, job_description: str) -> float:
        """Calculate relevance score using multiple methods"""
        
        # Create query from persona and job
        query = f"{persona} {job_description}"
        
        # Method 1: TF-IDF similarity
        try:
            corpus = [section_content, query]
            tfidf_matrix = self.tfidf.fit_transform(corpus)
            tfidf_similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        except:
            tfidf_similarity = 0.0
        
        # Method 2: Semantic similarity using sentence transformers
        try:
            section_embedding = self.sentence_model.encode([section_content[:512]])  # Limit length
            query_embedding = self.sentence_model.encode([query])
            semantic_similarity = cosine_similarity(section_embedding, query_embedding)[0][0]
        except:
            semantic_similarity = 0.0
        
        # Method 3: Keyword matching
        persona_keywords = persona.lower().split()
        job_keywords = job_description.lower().split()
        all_keywords = set(persona_keywords + job_keywords)
        
        section_lower = section_content.lower()
        keyword_matches = sum(1 for keyword in all_keywords if keyword in section_lower)
        keyword_score = min(keyword_matches / len(all_keywords), 1.0) if all_keywords else 0.0
        
        # Combine scores with weights
        final_score = (
            0.4 * semantic_similarity +
            0.35 * tfidf_similarity +
            0.25 * keyword_score
        )
        
        return final_score
    
    def process_documents(self, input_config: Dict) -> Dict:
        """Main processing function"""
        start_time = time.time()
        
        # Extract input parameters
        documents = input_config['documents']
        persona = input_config['persona']['role']
        job_to_be_done = input_config['job_to_be_done']['task']
        
        print(f"Processing {len(documents)} documents for persona: {persona}")
        print(f"Job to be done: {job_to_be_done}")
        
        all_sections = []
        
        # Process each document
        for doc_info in documents:
            filename = doc_info['filename']
            doc_title = doc_info.get('title', filename)
            
            # Construct full path (assuming PDFs are in PDFs/ subdirectory)
            pdf_path = os.path.join('PDFs', filename)
            
            if not os.path.exists(pdf_path):
                print(f"Warning: File {pdf_path} not found, skipping...")
                continue
            
            print(f"Processing: {filename}")
            
            # Extract text from PDF
            pages_text = self.extract_text_from_pdf(pdf_path)
            
            if not pages_text:
                print(f"Warning: No text extracted from {filename}")
                continue
            
            # Process each page
            for page_num, page_text in pages_text.items():
                sections = self.segment_text_into_sections(page_text, page_num)
                
                for section in sections:
                    # Calculate relevance score
                    relevance_score = self.calculate_relevance_score(
                        section['content'], persona, job_to_be_done
                    )
                    
                    section['document'] = filename
                    section['relevance_score'] = relevance_score
                    all_sections.append(section)
        
        # Sort sections by relevance score (descending)
        all_sections.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        # Select top sections (limit to avoid overwhelming output)
        top_sections = all_sections[:20]  # Top 20 most relevant sections
        
        # Prepare extracted sections output
        extracted_sections = []
        for i, section in enumerate(top_sections, 1):
            extracted_sections.append({
                "document": section['document'],
                "section_title": section['title'],
                "importance_rank": i,
                "page_number": section['page_number']
            })
        
        # Prepare subsection analysis (refined text)
        subsection_analysis = []
        for section in top_sections[:10]:  # Top 10 for detailed analysis
            # Refine text by extracting key sentences
            refined_text = self.refine_section_text(section['content'], persona, job_to_be_done)
            
            subsection_analysis.append({
                "document": section['document'],
                "refined_text": refined_text,
                "page_number": section['page_number']
            })
        
        # Prepare final output
        output = {
            "metadata": {
                "input_documents": [doc['filename'] for doc in documents],
                "persona": persona,
                "job_to_be_done": job_to_be_done,
                "processing_timestamp": datetime.now().isoformat(),
                "processing_time_seconds": round(time.time() - start_time, 2)
            },
            "extracted_sections": extracted_sections,
            "subsection_analysis": subsection_analysis
        }
        
        print(f"Processing completed in {output['metadata']['processing_time_seconds']} seconds")
        print(f"Found {len(extracted_sections)} relevant sections")
        
        return output
    
    def refine_section_text(self, content: str, persona: str, job_description: str) -> str:
        """Extract and refine the most important sentences from a section"""
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        if len(sentences) <= 3:
            return content[:500]  # Return truncated content if too few sentences
        
        # Score each sentence for relevance
        query = f"{persona} {job_description}".lower()
        query_words = set(query.split())
        
        scored_sentences = []
        for sentence in sentences:
            sentence_words = set(sentence.lower().split())
            overlap = len(query_words.intersection(sentence_words))
            score = overlap / len(query_words) if query_words else 0
            scored_sentences.append((sentence, score))
        
        # Sort by score and take top sentences
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        top_sentences = [s[0] for s in scored_sentences[:3]]
        
        refined = '. '.join(top_sentences)
        return refined[:500] + "..." if len(refined) > 500 else refined


def main():
    """Main execution function"""
    # Initialize the system
    system = DocumentIntelligenceSystem()
    
    # Process all collections
    collections = ['Collection 1', 'Collection 2', 'Collection 3']
    
    for collection in collections:
        print(f"\n{'='*50}")
        print(f"Processing {collection}")
        print('='*50)
        
        collection_path = Path(collection)
        input_file = collection_path / 'challenge1b_input.json'
        output_file = collection_path / 'challenge1b_output.json'
        
        if not input_file.exists():
            print(f"Input file {input_file} not found, skipping...")
            continue
        
        # Change to collection directory for processing
        original_dir = os.getcwd()
        os.chdir(collection_path)
        
        try:
            # Load input configuration
            with open('challenge1b_input.json', 'r') as f:
                input_config = json.load(f)
            
            # Process documents
            result = system.process_documents(input_config)
            
            # Save output
            with open('challenge1b_output.json', 'w') as f:
                json.dump(result, f, indent=2)
            
            print(f"Results saved to {output_file}")
            
        except Exception as e:
            print(f"Error processing {collection}: {e}")
        finally:
            os.chdir(original_dir)
    
    print(f"\n{'='*50}")
    print("All collections processed!")
    print('='*50)


if _name_ == "_main_":
    main()
