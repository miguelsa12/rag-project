import fitz  # PyMuPDF
import torch
import numpy as np
import re
import sys
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# Supress Temporary Warnings
class SuppressMuPDFWarnings:
    def __enter__(self):
        self._original_stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stderr.close()
        sys.stderr = self._original_stderr


class RAGPipeline:
    def __init__(self, pdf_folder="data/", embedding_model_path="models/paraphrase-MiniLM-L6-v2",
                 llm_model_path="models/falcon-rw-1b", top_k=3):
        self.pdf_folder = pdf_folder
        self.top_k = top_k
        self.embedder = SentenceTransformer(embedding_model_path, local_files_only=True)
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_path, local_files_only=True)
        self.llm_model = AutoModelForCausalLM.from_pretrained(llm_model_path, local_files_only=True)
        device = 0 if torch.cuda.is_available() else -1
        self.generator = pipeline("text-generation", model=self.llm_model, tokenizer=self.tokenizer, device=device)

    def load_pdfs(self):
        documents = []
        for pdf_path in Path(self.pdf_folder).glob("*.pdf"):
            try:
                with SuppressMuPDFWarnings():
                    with fitz.open(pdf_path) as doc:
                        text = "".join([page.get_text() for page in doc])
                        if len(text.strip()) > 10:  # filtra PDFs vacíos o corruptos
                            documents.append({"filename": pdf_path.name, "content": text})
            except Exception as e:
                print(f"Error loading {pdf_path.name}: {e}")
        return documents

    def chunk_text(self, text, max_tokens=200):
        sentences = re.split(r'(?<=[.!?]) +', text)
        chunks, current_chunk = [], ""
        for sentence in sentences:
            if len(current_chunk.split()) + len(sentence.split()) < max_tokens:
                current_chunk += " " + sentence
            else:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks

    def embed_chunks(self, documents):
        embedded_chunks = []
        for doc in documents:
            chunks = self.chunk_text(doc["content"])
            for chunk in chunks:
                embedding = self.embedder.encode(chunk)
                embedded_chunks.append({
                    "filename": doc["filename"],
                    "text": chunk,
                    "embedding": embedding
                })
        return embedded_chunks

    def retrieve_relevant_chunks(self, question, embedded_chunks):
        question_embedding = self.embedder.encode(question)
        similarities = [
            (cosine_similarity([question_embedding], [chunk["embedding"]])[0][0], chunk)
            for chunk in embedded_chunks
        ]
        sorted_chunks = sorted(similarities, key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in sorted_chunks[:self.top_k]]

    def generate_answer(self, question, retrieved_chunks):
        context = "\n".join([chunk["text"] for chunk in retrieved_chunks])
        prompt = (
            f"Use the following information to answer the instruction briefly and precisely.\n"
            f"Context:\n{context}\n\nInstruction: {question}\nAnswer:"
        )
        response = self.generator(
            prompt,
            max_new_tokens=100,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )[0]["generated_text"]

        # Extract answer 
        answer_start = response.find("Answer:")
        answer = response[answer_start + len("Answer:"):].strip() if answer_start != -1 else response.strip()

        # Delete duplicated exact lines
        lines = answer.splitlines()
        seen, unique_lines = set(), []
        for line in lines:
            clean_line = line.strip()
            if clean_line and clean_line not in seen:
                unique_lines.append(clean_line)
                seen.add(clean_line)

        cleaned_answer = " ".join(unique_lines)

        print(f"\nQuestion:\n{question}\n\nAnswer:\n{cleaned_answer}")
        return cleaned_answer

    def evaluate_answer(self, question, answer):
        question_embedding = self.embedder.encode(question)
        answer_embedding = self.embedder.encode(answer)
        similarity = cosine_similarity([question_embedding], [answer_embedding])[0][0]
        return similarity


if __name__ == "__main__":
    rag = RAGPipeline()
    documents = rag.load_pdfs()
    embedded_chunks = rag.embed_chunks(documents)
    question = "What is the edge?"
    relevant_chunks = rag.retrieve_relevant_chunks(question, embedded_chunks)
    answer = rag.generate_answer(question, relevant_chunks)
    score = rag.evaluate_answer(question, answer)
    print(f"Similarity Score: {score:.2f}")




