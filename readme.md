1. RAG Agent for SEC 10-K Filings

This project implements a RAG agent that can automatically download 10-K filings, extract financial text, and answer questions based on the queries.


2. Features

Automatic 10-K download from SEC archives (for Microsoft, Google, NVIDIA).

Extracts text from downloaded filings (HTML/PDF).

Splits documents into smaller chunks for efficient retrieval.

Breaks a query into sub-queries, retrieves evidence, and provides a structured JSON answer.

Supports multiple question types/ queries.

3. Project Structure
RAG Agent_Uniqus/
│
├── main.py        # Main script (RAG agent pipeline)
├── data/          # Stores downloaded 10-K filings
├── requirements.txt # Download dependencies
└── readme.md

4. Installation

Clone the repo:

git clone https://github.com/<codezJhonny>/RAG-Agent_Uniqus.git
cd RAG-Agent_Uniqus


To install dependencies:

pip install -r requirements.txt

5. Usage

Ask a query directly from the terminal:

python main.py "Which company had the highest operating margin in 2023?"

Example Output
{
  "query": "Which company had the highest operating margin in 2023?",
  "sub_queries": [
    "Microsoft operating margin 2023",
    "Google operating margin 2023",
    "NVIDIA operating margin 2023"
  ],
  "reasoning": "Retrieved most relevant text from filings based on sub-queries.",
  "sources": [
    {"file": "data/MSFT_2023.htm", "page": 33, "excerpt": "Operating margin was 42%..."},
    {"file": "data/GOOGL_2023.htm", "page": 41, "excerpt": "Operating margin was 28%..."}
  ],
  "answer": "Microsoft had the highest operating margin in 2023."
}


Author: V Bhanuprakash