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


# Suppress MuPDF warnings
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
                        if len(text.strip()) > 10:
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

    def clean_context_text(self, text):
        text = re.sub(r"\bAnswer:\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\bQuestion:\s*.+?[\.\?\!]", "", text, flags=re.IGNORECASE)
        return text.strip()

    def generate_answer(self, question, retrieved_chunks):
        # Clean each retrieved chunk
        cleaned_chunks = [self.clean_context_text(chunk["text"]) for chunk in retrieved_chunks]

        # Join and truncate
        combined_context = "\n".join(cleaned_chunks)
        max_context_tokens = 850
        encoded_context = self.tokenizer.encode(combined_context, truncation=True, max_length=max_context_tokens)
        truncated_context = self.tokenizer.decode(encoded_context, skip_special_tokens=True)

        # Prompt
        prompt = (
            "You are a helpful and concise assistant. Using the provided document context, "
            "answer the question in your own words using complete sentences. "
            "Do not repeat any questions or copy directly from the context. "
            "Provide one clear and informative answer.\n\n"
            f"Context:\n{truncated_context}\n\n"
            f"Question: {question}\n"
        )

        # Generate response
        response = self.generator(
            prompt,
            max_new_tokens=150,
            top_k=50,
            #temperature=0.5,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )[0]["generated_text"]

        answer_start = response.find("Answer:")
        if answer_start != -1:
            answer = response[answer_start + len("Answer:"):].strip()
            stop_match = re.search(r"[.?!]\s", answer)
            if stop_match:
                answer = answer[:stop_match.end()].strip()
        else:
            answer = response.strip()

        print(f"\nQuestion:\n{question}\n\nAnswer:\n{answer}")
        return answer

    def evaluate_answer(self, question, answer):
        question_embedding = self.embedder.encode(question)
        answer_embedding = self.embedder.encode(answer)
        similarity = cosine_similarity([question_embedding], [answer_embedding])[0][0]
        return similarity


if __name__ == "__main__":
    rag = RAGPipeline()
    documents = rag.load_pdfs()
    embedded_chunks = rag.embed_chunks(documents)
    question = "What Swarm Learning is?"
    relevant_chunks = rag.retrieve_relevant_chunks(question, embedded_chunks)
    answer = rag.generate_answer(question, relevant_chunks)
    score = rag.evaluate_answer(question, answer)
    print(f"Similarity Score: {score:.2f}")





