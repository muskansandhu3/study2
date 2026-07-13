Quick start (local)
===================

1. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

2. Prepare OpenAI key:

```bash
export OPENAI_API_KEY="your-api-key"
```

3. Build embeddings (example):

```bash
python embedding_pipeline.py --openai-key "$OPENAI_API_KEY" --data-path ./data --chroma-dir ./chroma_db_openai --collection-name nasa_space_missions_text
```

4. Run Streamlit chat:

```bash
streamlit run chat.py
```

Notes:
- If you don't have real data yet, put a few small `.txt` files in `./data/apollo11`, `./data/apollo13`, or `./data/challenger` for testing.
- Pushing to GitHub requires your local Git to be configured with credentials or SSH keys.
