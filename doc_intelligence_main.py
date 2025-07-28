#!/usr/bin/env python3
"""
Universal Document Intelligence System
Efficiently processes any type of document collection for persona-driven analysis
"""

import json
import os
import time
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
import re
from pathlib import Path
import unicodedata
import logging

# Using lightweight libraries for CPU-only processing
try:
    import PyPDF2
    from sentence_transformers import SentenceTransformer
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import KMeans
except ImportError as e:
    print(f"Missing required library: {e}")
    print("Install with: pip install PyPDF2 sentence-transformers scikit-learn numpy")
    exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UniversalDocumentIntelligence:
    def _init_(self):
        """Initialize the system with optimized models"""
        logger.info("Initializing Universal Document Intelligence System...")
        
        # Use efficient sentence transformer model
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.tfidf = TfidfVectorizer(
            max_features=2000,
            stop_words='english',
            ngram_range=(1, 3),
            min_df=1,
            max_df=0.95
        )
        
        # Enhanced content type keywords for better classification
        self.content_patterns = {
            'recipe': ['ingredients', 'preparation', 'cooking', 'recipe', 'cook', 'bake', 'fry', 'vegetarian', 'vegan', 'chickpeas', 'eggplant'],
            'travel': ['visit', 'attraction', 'hotel', 'restaurant', 'tour', 'location', 'destination', 'beach', 'city', 'culture', 'things to do'],
            'technical': ['acrobat', 'pdf', 'form', 'fillable', 'signature', 'convert', 'edit', 'export', 'create', 'document'],
            'business': ['management', 'strategy', 'analysis', 'report', 'planning', 'operation', 'professional', 'compliance'],
            'academic': ['research', 'study', 'analysis', 'theory', 'methodology', 'conclusion']
        }
    
    def clean_text(self, text: str) -> str:
        """Enhanced text cleaning with better Unicode handling"""
        if not text:
            return ""
        
        # Unicode replacements
        replacements = {
            '\u2022': '• ',    # Bullet point
            '\u2013': '- ',    # En dash
            '\u2014': '-- ',   # Em dash
            '\u2018': "'",     # Left single quote
            '\u2019': "'",     # Right single quote
            '\u201c': '"',     # Left double quote
            '\u201d': '"',     # Right double quote
            '\u00a0': ' ',     # Non-breaking space
            '\u00ae': '(R)',   # Registered trademark
            '\u00a9': '(C)',   # Copyright
            '\u2026': '...',   # Ellipsis
            '\u00b7': '• ',    # Middle dot
            '\u2023': '• ',    # Triangular bullet
        }
        
        for unicode_char, replacement in replacements.items():
            text = text.replace(unicode_char, replacement)
        
        # Normalize and clean
        text = unicodedata.normalize('NFKD', text)
        text = ''.join(char for char in text if char.isprintable() or char in '\n\t')
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n+', '\n', text)
        
        return text.strip()
    
    def extract_text_from_pdf(self, pdf_path: str) -> Dict[int, str]:
        """Optimized PDF text extraction with better error handling"""
        try:
            pages_text = {}
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    try:
                        text = page.extract_text()
                        if text and text.strip():
                            cleaned_text = self.clean_text(text)
                            if len(cleaned_text) > 30:  # Only keep substantial content
                                pages_text[page_num] = cleaned_text
                    except Exception as e:
                        logger.warning(f"Error extracting page {page_num} from {pdf_path}: {e}")
                        continue
                        
            return pages_text
        except Exception as e:
            logger.error(f"Error reading PDF {pdf_path}: {e}")
            return {}
    
    def detect_content_type(self, text: str) -> str:
        """Detect the type of content to adapt processing"""
        text_lower = text.lower()
        type_scores = {}
        
        for content_type, keywords in self.content_patterns.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            type_scores[content_type] = score / len(keywords)
        
        return max(type_scores, key=type_scores.get) if type_scores else 'general'
    
    def extract_meaningful_title(self, text: str, content_type: str) -> str:
        """Extract meaningful titles based on content type"""
        lines = text.split('\n')
        
        # For recipe content
        if content_type == 'recipe':
            # Look for dish names
            for line in lines[:5]:
                line = line.strip()
                if (len(line) > 3 and len(line) < 60 and 
                    not any(word in line.lower() for word in ['ingredients', 'instructions', 'page', '•', 'preparation'])):
                    # Clean up common prefixes
                    line = re.sub(r'^[•\-\*\s]+', '', line)
                    if len(line) > 2:
                        return line
        
        # For travel content
        elif content_type == 'travel':
            # Look for section headers or place names
            for line in lines[:3]:
                line = line.strip()
                if (len(line) > 5 and len(line) < 80 and 
                    any(word in line.lower() for word in ['guide', 'adventures', 'experiences', 'tips', 'nightlife', 'coastal'])):
                    return line
        
        # For technical content
        elif content_type == 'technical':
            # Look for procedure or feature titles
            for line in lines[:3]:
                line = line.strip()
                if (len(line) > 10 and len(line) < 100 and 
                    any(word in line.lower() for word in ['create', 'convert', 'fill', 'sign', 'form', 'pdf'])):
                    return line
        
        # Fallback: use first substantial line
        for line in lines[:3]:
            line = line.strip()
            if len(line) > 10 and len(line) < 120:
                # Remove excessive punctuation and clean up
                line = re.sub(r'^[•\-\*\s]+', '', line)
                if len(line) > 5:
                    return line[:100] + "..." if len(line) > 100 else line
        
        # Last resort: use first few words
        words = text.split()[:8]
        return ' '.join(words) if words else "Content Section"
    
    def segment_text_intelligently(self, text: str, page_num: int, content_type: str) -> List[Dict]:
        """Intelligent text segmentation based on content type"""
        sections = []
        
        if content_type == 'recipe':
            # Enhanced recipe pattern matching
            recipe_patterns = [
                r'([A-Za-z\s&]+?)\s*•?\s*Ingredients?\s*:([^•]*?)(?=Instructions?|•[A-Za-z\s&]+•|$)',
                r'([A-Za-z\s&]+?)\s*Ingredients?\s*:([^I]*?)(?=Instructions?|$)',
                r'([A-Za-z\s&]+?)\s*•\s*([^•]*?)(?=•[A-Za-z\s&]+|$)'
            ]
            
            for pattern in recipe_patterns:
                matches = list(re.finditer(pattern, text, re.IGNORECASE | re.DOTALL))
                for match in matches:
                    title = self.extract_meaningful_title(match.group(1), content_type)
                    content = match.group(0)
                    
                    if len(content) > 80 and len(title) > 2:
                        sections.append({
                            'title': title,
                            'content': self.clean_text(content),
                            'page_number': page_num,
                            'start_pos': match.start(),
                            'content_type': content_type
                        })
        
        elif content_type == 'travel':
            # Look for travel sections with headers
            travel_patterns = [
                r'([A-Za-z\s\-:]+(?:Adventures?|Experiences?|Tips?|Guide|Nightlife|Entertainment))\s*([^A-Z]{100,}?)(?=[A-Z][a-z\s\-:]+(?:Adventures?|Experiences?|Tips?|Guide|Nightlife|Entertainment)|$)',
                r'((?:Bars?|Nightclub|Restaurant|Hotel|Beach|City)[^:]?):\s([^:]{100,}?)(?=(?:Bars?|Nightclub|Restaurant|Hotel|Beach|City)[^:]*?:|$)'
            ]
            
            for pattern in travel_patterns:
                matches = list(re.finditer(pattern, text, re.IGNORECASE | re.DOTALL))
                for match in matches:
                    title = self.extract_meaningful_title(match.group(1), content_type)
                    content = match.group(0)
                    
                    if len(content) > 100:
                        sections.append({
                            'title': title,
                            'content': self.clean_text(content),
                            'page_number': page_num,
                            'start_pos': match.start(),
                            'content_type': content_type
                        })
        
        elif content_type == 'technical':
            # Look for technical procedures and instructions
            tech_patterns = [
                r'([A-Za-z\s\-]+(?:form|PDF|sign|create|convert|fill)[^.]?)\s([^A-Z]{50,}?)(?=[A-Z][a-z\s\-]+(?:form|PDF|sign|create|convert|fill)|$)',
                r'(To\s+[^:]+?):\s*([^T]{100,}?)(?=To\s+[^:]+?:|$)'
            ]
            
            for pattern in tech_patterns:
                matches = list(re.finditer(pattern, text, re.IGNORECASE | re.DOTALL))
                for match in matches:
                    title = self.extract_meaningful_title(match.group(1), content_type)
                    content = match.group(0)
                    
                    if len(content) > 80:
                        sections.append({
                            'title': title,
                            'content': self.clean_text(content),
                            'page_number': page_num,
                            'start_pos': match.start(),
                            'content_type': content_type
                        })
        
        # Fallback to paragraph-based segmentation if no specific patterns found
        if not sections:
            paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 60]
            for i, para in enumerate(paragraphs[:10]):
                title = self.extract_meaningful_title(para, content_type)
                
                sections.append({
                    'title': title,
                    'content': self.clean_text(para),
                    'page_number': page_num,
                    'start_pos': i * 500,
                    'content_type': content_type
                })
        
        return sections
    
    def calculate_advanced_relevance(self, section: Dict, persona: str, task: str) -> float:
        """Advanced relevance scoring with enhanced content-type awareness"""
        content = section['content']
        content_type = section.get('content_type', 'general')
        
        # Create enhanced query
        query = f"{persona} {task}"
        
        # Method 1: Semantic similarity
        try:
            section_embedding = self.sentence_model.encode([content[:1000]])
            query_embedding = self.sentence_model.encode([query])
            semantic_score = cosine_similarity(section_embedding, query_embedding)[0][0]
        except:
            semantic_score = 0.0
        
        # Method 2: TF-IDF similarity
        try:
            corpus = [content, query]
            tfidf_matrix = self.tfidf.fit_transform(corpus)
            tfidf_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        except:
            tfidf_score = 0.0
        
        # Method 3: Enhanced keyword matching
        persona_words = set(persona.lower().split())
        task_words = set(task.lower().split())
        content_words = set(content.lower().split())
        
        persona_overlap = len(persona_words.intersection(content_words))
        task_overlap = len(task_words.intersection(content_words))
        
        persona_score = min(persona_overlap / len(persona_words), 1.0) if persona_words else 0
        task_score = min(task_overlap / len(task_words), 1.0) if task_words else 0
        
        # Method 4: Content-type specific scoring
        content_keywords = self.content_patterns.get(content_type, [])
        content_keyword_matches = sum(1 for kw in content_keywords if kw in content.lower())
        content_type_score = min(content_keyword_matches / len(content_keywords), 1.0) if content_keywords else 0
        
        # Method 5: Task-specific bonuses
        task_bonuses = 0.0
        
        # Vegetarian/buffet bonus
        if 'vegetarian' in task.lower() or 'buffet' in task.lower():
            veggie_keywords = ['vegetarian', 'vegan', 'chickpeas', 'vegetables', 'eggplant', 'tahini', 'hummus', 'falafel', 'ratatouille']
            veggie_matches = sum(1 for kw in veggie_keywords if kw in content.lower())
            task_bonuses += min(veggie_matches / 5, 0.3)
        
        # Travel planning bonus
        if 'travel' in task.lower() or 'trip' in task.lower():
            travel_keywords = ['beach', 'city', 'restaurant', 'hotel', 'tour', 'nightlife', 'adventure', 'culture']
            travel_matches = sum(1 for kw in travel_keywords if kw in content.lower())
            task_bonuses += min(travel_matches / 5, 0.3)
        
        # Form/HR bonus
        if 'form' in task.lower() or 'onboarding' in task.lower() or 'compliance' in task.lower():
            form_keywords = ['form', 'fillable', 'signature', 'pdf', 'acrobat', 'sign', 'fill']
            form_matches = sum(1 for kw in form_keywords if kw in content.lower())
            task_bonuses += min(form_matches / 5, 0.3)
        
        # Method 6: Length and structure bonus
        structure_score = min(len(content) / 800, 1.0)
        
        # Combine scores with adaptive weights
        if content_type == 'recipe':
            weights = [0.15, 0.10, 0.20, 0.30, 0.15, 0.05]
        elif content_type == 'travel':
            weights = [0.25, 0.15, 0.15, 0.25, 0.15, 0.05]
        elif content_type == 'technical':
            weights = [0.20, 0.20, 0.15, 0.25, 0.15, 0.05]
        else:
            weights = [0.30, 0.25, 0.15, 0.15, 0.10, 0.05]
        
        final_score = (
            weights[0] * semantic_score +
            weights[1] * tfidf_score +
            weights[2] * persona_score +
            weights[3] * task_score +
            weights[4] * content_type_score +
            weights[5] * structure_score +
            task_bonuses
        )
        
        return final_score
    
    def refine_content_intelligently(self, content: str, persona: str, task: str, content_type: str) -> str:
        """Enhanced content refinement matching expected output formats"""
        
        if content_type == 'recipe':
            return self._refine_recipe_content(content)
        elif content_type == 'travel':
            return self._refine_travel_content(content, task)
        elif content_type == 'technical':
            return self._refine_technical_content(content, task)
        else:
            return self._refine_general_content(content, persona, task)
    
    def _refine_recipe_content(self, content: str) -> str:
        """Refine recipe content to match expected format"""
        content_clean = self.clean_text(content)
        
        # Extract recipe title
        title_match = re.search(r'^([A-Za-z\s&]+?)(?:\s*•|\s*Ingredients)', content_clean, re.IGNORECASE)
        title = title_match.group(1).strip() if title_match else ""
        
        # Extract ingredients
        ingredients_match = re.search(r'Ingredients?\s*:?\s*([^I]*?)(?=Instructions?|$)', content_clean, re.IGNORECASE | re.DOTALL)
        ingredients_text = ingredients_match.group(1).strip() if ingredients_match else ""
        
        # Extract instructions  
        instructions_match = re.search(r'Instructions?\s*:?\s*([^.](?:\.[^.]){0,8})', content_clean, re.IGNORECASE | re.DOTALL)
        instructions_text = instructions_match.group(1).strip() if instructions_match else ""
        
        # Format the output
        result_parts = []
        
        if title:
            result_parts.append(title)
        
        if ingredients_text:
            ingredients_cleaned = re.sub(r'^[•\s]*', '', ingredients_text)
            ingredients_cleaned = re.sub(r'\s*•\s*', ', ', ingredients_cleaned)
            ingredients_cleaned = re.sub(r'\s*o\s*', '', ingredients_cleaned)
            result_parts.append(f"Ingredients: {ingredients_cleaned}")
        
        if instructions_text:
            instructions_cleaned = re.sub(r'^[•\s]*', '', instructions_text)
            instructions_cleaned = re.sub(r'\s*o\s*', '', instructions_cleaned)
            result_parts.append(f"Instructions: {instructions_cleaned}")
        
        final_text = ' '.join(result_parts)
        
        # Ensure reasonable length
        if len(final_text) > 500:
            sentences = re.split(r'[.!?]+', final_text)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
            final_text = '. '.join(sentences[:5])
            if not final_text.endswith('.'):
                final_text += '.'
        
        return final_text
    
    def _refine_travel_content(self, content: str, task: str) -> str:
        """Refine travel content to match expected detailed format"""
        content_clean = self.clean_text(content)
        
        # For travel content, preserve more detail and structure
        sentences = re.split(r'[.!?]+', content_clean)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 15]
        
        # Score sentences based on travel relevance
        task_words = set(task.lower().split())
        travel_keywords = ['beach', 'city', 'restaurant', 'hotel', 'bar', 'club', 'tour', 'visit', 'explore', 'discover']
        
        scored_sentences = []
        for sentence in sentences:
            sentence_words = set(sentence.lower().split())
            task_overlap = len(task_words.intersection(sentence_words))
            travel_overlap = len(set(travel_keywords).intersection(sentence_words))
            
            score = (task_overlap * 2) + travel_overlap + (len(sentence) / 100)
            scored_sentences.append((sentence, score))
        
        # Select top sentences but maintain reasonable length
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        selected_sentences = []
        total_length = 0
        
        for sentence, score in scored_sentences:
            if total_length + len(sentence) < 800 and len(selected_sentences) < 8:
                selected_sentences.append(sentence)
                total_length += len(sentence)
            else:
                break
        
        refined = '. '.join(selected_sentences)
        if not refined.endswith('.'):
            refined += '.'
        
        return refined
    
    def _refine_technical_content(self, content: str, task: str) -> str:
        """Refine technical content for procedural information"""
        content_clean = self.clean_text(content)
        
        # For technical content, focus on procedures and instructions
        sentences = re.split(r'[.!?]+', content_clean)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        # Score based on technical relevance
        task_words = set(task.lower().split())
        tech_keywords = ['create', 'form', 'fillable', 'sign', 'pdf', 'acrobat', 'tool', 'field', 'document']
        
        scored_sentences = []
        for sentence in sentences:
            sentence_words = set(sentence.lower().split())
            task_overlap = len(task_words.intersection(sentence_words))
            tech_overlap = len(set(tech_keywords).intersection(sentence_words))
            
            # Bonus for procedural language
            procedural_bonus = 1 if any(word in sentence.lower() for word in ['to', 'select', 'click', 'choose', 'open']) else 0
            
            score = (task_overlap * 2) + tech_overlap + procedural_bonus
            scored_sentences.append((sentence, score))
        
        # Select top sentences
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        top_sentences = [s[0] for s in scored_sentences[:4]]
        
        refined = '. '.join(top_sentences)
        if not refined.endswith('.'):
            refined += '.'
        
        return refined
    
    def _refine_general_content(self, content: str, persona: str, task: str) -> str:
        """Refine general content"""
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 25]
        
        if len(sentences) <= 3:
            return self.clean_text(content[:400])
        
        # Score sentences for relevance
        query_words = set((persona + " " + task).lower().split())
        
        scored_sentences = []
        for sentence in sentences:
            sentence_words = set(sentence.lower().split())
            query_overlap = len(query_words.intersection(sentence_words))
            score = query_overlap / len(query_words) if query_words else 0
            scored_sentences.append((sentence, score))
        
        # Select top sentences
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        top_sentences = [s[0] for s in scored_sentences[:3]]
        
        refined = '. '.join(top_sentences)
        final_text = (refined[:400] + "...") if len(refined) > 400 else refined
        
        return self.clean_text(final_text)
    
    def parse_input_config(self, config: Dict) -> Tuple[str, str, List[Dict]]:
        """Parse different input configuration formats"""
        # Extract persona
        if 'persona' in config:
            if isinstance(config['persona'], dict):
                persona = config['persona'].get('role', 'Professional')
            else:
                persona = str(config['persona'])
        else:
            persona = 'Professional'
        
        # Extract task
        if 'job_to_be_done' in config:
            if isinstance(config['job_to_be_done'], dict):
                task = config['job_to_be_done'].get('task', 'Analyze documents')
            else:
                task = str(config['job_to_be_done'])
        elif 'task' in config:
            task = str(config['task'])
        else:
            task = 'Analyze documents'
        
        # Extract documents
        documents = config.get('documents', [])
        
        return persona, task, documents
    
    def process_documents(self, input_config: Dict) -> Dict:
        """Main processing function with enhanced capabilities"""
        start_time = time.time()
        
        persona, task, documents = self.parse_input_config(input_config)
        
        logger.info(f"Processing {len(documents)} documents")
        logger.info(f"Persona: {persona}")
        logger.info(f"Task: {task}")
        
        all_sections = []
        processed_docs = 0
        
        # Process each document
        for doc_info in documents:
            if isinstance(doc_info, dict):
                filename = doc_info.get('filename', doc_info.get('name', ''))
            else:
                filename = str(doc_info)
            
            if not filename:
                continue
            
            # Try different possible paths
            possible_paths = [
                filename,
                os.path.join('PDFs', filename),
                os.path.join('documents', filename),
                os.path.join('data', filename)
            ]
            
            pdf_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    pdf_path = path
                    break
            
            if not pdf_path:
                logger.warning(f"File not found: {filename}")
                continue
            
            logger.info(f"Processing: {filename}")
            
            # Extract text
            pages_text = self.extract_text_from_pdf(pdf_path)
            if not pages_text:
                logger.warning(f"No text extracted from {filename}")
                continue
            
            # Detect content type from first page
            first_page_text = list(pages_text.values())[0]
            content_type = self.detect_content_type(first_page_text)
            
            # Process each page
            for page_num, page_text in pages_text.items():
                sections = self.segment_text_intelligently(page_text, page_num, content_type)
                
                for section in sections:
                    relevance_score = self.calculate_advanced_relevance(section, persona, task)
                    section['document'] = filename
                    section['relevance_score'] = relevance_score
                    all_sections.append(section)
            
            processed_docs += 1
        
        if not all_sections:
            logger.warning("No sections extracted from any documents")
            return self._create_empty_result(persona, task, [doc.get('filename', str(doc)) for doc in documents])
        
        # Sort and select top sections
        all_sections.sort(key=lambda x: x['relevance_score'], reverse=True)
        top_sections = all_sections[:5]
        
        # Prepare results in exact expected format
        extracted_sections = []
        for i, section in enumerate(top_sections, 1):
            extracted_sections.append({
                "document": section['document'],
                "section_title": section['title'],
                "importance_rank": i,
                "page_number": section['page_number']
            })
        
        subsection_analysis = []
        for section in top_sections:
            refined_text = self.refine_content_intelligently(
                section['content'], 
                persona, 
                task, 
                section.get('content_type', 'general')
            )
            
            subsection_analysis.append({
                "document": section['document'],
                "refined_text": refined_text,
                "page_number": section['page_number']
            })
        
        # Create final output matching expected format exactly
        result = {
            "metadata": {
                "input_documents": [doc.get('filename', str(doc)) if isinstance(doc, dict) else str(doc) for doc in documents],
                "persona": persona,
                "job_to_be_done": task,
                "processing_timestamp": datetime.now().isoformat()
            },
            "extracted_sections": extracted_sections,
            "subsection_analysis": subsection_analysis
        }
        
        processing_time = round(time.time() - start_time, 2)
        logger.info(f"Processing completed in {processing_time} seconds")
        logger.info(f"Found {len(extracted_sections)} relevant sections from {processed_docs} documents")
        
        return result
    
    def _create_empty_result(self, persona: str, task: str, document_names: List[str]) -> Dict:
        """Create empty result structure when no content is found"""
        return {
            "metadata": {
                "input_documents": document_names,
                "persona": persona,
                "job_to_be_done": task,
                "processing_timestamp": datetime.now().isoformat()
            },
            "extracted_sections": [],
            "subsection_analysis": []
        }

def process_collection(collection_name: str, system: UniversalDocumentIntelligence) -> bool:
    """Process a single collection"""
    logger.info(f"Processing {collection_name}")
    
    collection_path = Path(collection_name)
    input_file = collection_path / 'challenge1b_input.json'
    output_file = collection_path / 'challenge1b_output.json'
    
    if not input_file.exists():
        logger.warning(f"Input file {input_file} not found")
        return False
    
    original_dir = os.getcwd()
    os.chdir(collection_path)
    
    try:
        # Load input configuration
        with open('challenge1b_input.json', 'r', encoding='utf-8') as f:
            input_config = json.load(f)
        
        # Process documents
        result = system.process_documents(input_config)
        
        # Save output
        with open('challenge1b_output.json', 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
        
        logger.info(f"Results saved to {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"Error processing {collection_name}: {e}")
        return False
    finally:
        os.chdir(original_dir)

def main():
    """Main execution function"""
    # Initialize the system
    system = UniversalDocumentIntelligence()
    
    # Auto-detect collections
    collections = []
    
    # Look for numbered collections
    for i in range(1, 10):
        collection_name = f'Collection {i}'
        if os.path.exists(collection_name):
            collections.append(collection_name)
    
    # Look for other common collection patterns
    common_patterns = ['Collection1', 'Collection2', 'Collection3', 'Test1', 'Test2', 'Test3']
    for pattern in common_patterns:
        if os.path.exists(pattern) and pattern not in collections:
            collections.append(pattern)
    
    # If no collections found, look for individual input files
    if not collections:
        if os.path.exists('challenge1b_input.json'):
            collections = ['.']
    
    if not collections:
        logger.error("No collections or input files found!")
        return
    
    logger.info(f"Found {len(collections)} collections to process")
    
    successful = 0
    for collection in collections:
        if process_collection(collection, system):
            successful += 1
    
    logger.info(f"Successfully processed {successful}/{len(collections)} collections")

if __name__ == "_main_":
    main()