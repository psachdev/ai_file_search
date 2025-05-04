# ai_file_search

export OPENAI_API_KEY="<OPENAI_API_KEY>"
python3 pdfToVectorStore_openai.py

pip3 install chromadb
python3 pdfToVectorStore_chroma_local.py

export DEEPSEEK_API_KEY="<DEEPSEEK_API_KEY>"
python3 basic_api_check.py
python3 pdfToVectorStore_chroma_deepseek_reasoner.py
