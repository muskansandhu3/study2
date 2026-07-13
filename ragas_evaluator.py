from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from typing import Dict, List, Optional

# RAGAS imports (use ragas.evaluation API)
try:
    from ragas.evaluation import LangchainLLMWrapper, LangchainEmbeddingsWrapper, Dataset, evaluate
    from ragas.metrics import ResponseRelevancy, Faithfulness
    RAGAS_AVAILABLE = True
except Exception:
    RAGAS_AVAILABLE = False

def evaluate_response_quality(question: str, answer: str, contexts: List[str]) -> Dict[str, float]:
    """Evaluate response quality using RAGAS metrics"""
    if not RAGAS_AVAILABLE:
        return {"error": "RAGAS not available"}
    
    try:
        evaluator_llm = LangchainLLMWrapper(llm=ChatOpenAI(temperature=0.0, model_name="gpt-3.5-turbo"))
        evaluator_embeddings = LangchainEmbeddingsWrapper(embeddings=OpenAIEmbeddings(model="text-embedding-3-small"))

        metrics = [ResponseRelevancy(), Faithfulness()]

        # Build ragas Dataset
        ds = Dataset({
            'question': [question],
            'contexts': [[c for c in contexts]],
            'answer': [answer],
            'ground_truth': [[]],
        })

        res = evaluate(dataset=ds, metrics=metrics, llm=evaluator_llm, embeddings=evaluator_embeddings)

        # extract numeric metrics
        out = {}
        try:
            # res is a Result object with .to_dict() or item access
            d = res.to_dict() if hasattr(res, 'to_dict') else dict(res)
            # d should have aggregated metrics
            # also per-row metrics may be in d['metrics'] or similar
            for k, v in d.items():
                try:
                    out[k] = float(v)
                except Exception:
                    out[k] = v
        except Exception:
            out = {"result": str(res)}

        return out
    except Exception as e:
        return {"error": str(e)}


def batch_evaluate(questions_file: str, openai_key: str, chroma_dir: str, collection_name: str, n_docs: int = 3, output_file: str = "evaluation_results.json") -> Dict:
    """Run batch end-to-end evaluation using the evaluation questions file.

    For each question:
      - retrieve top-k docs from Chroma
      - format context
      - generate answer with the LLM
      - evaluate with RAGAS metrics

    Returns a dict with per-question metrics and aggregate averages.
    """
    import json
    from pathlib import Path
    import rag_client
    import llm_client

    if not RAGAS_AVAILABLE:
        return {"error": "RAGAS not available"}

    # initialize collection
    collection, success, error = rag_client.initialize_rag_system(chroma_dir, collection_name)
    if not success:
        return {"error": f"Failed to initialize collection: {error}"}

    # prepare RAGAS wrappers
    evaluator_llm = LangchainLLMWrapper(llm=ChatOpenAI(temperature=0.0, model_name="gpt-3.5-turbo"))
    evaluator_embeddings = LangchainEmbeddingsWrapper(embeddings=OpenAIEmbeddings(model="text-embedding-3-small"))
    metrics = [ResponseRelevancy(), Faithfulness()]

    results_per_q = []
    aggregates = {}

    # read questions file (one question per line)
    p = Path(questions_file)
    if not p.exists():
        return {"error": f"Questions file not found: {questions_file}"}

    questions = [l.strip() for l in p.read_text(encoding='utf-8').splitlines() if l.strip()]

    for q in questions:
        # retrieve
        docs_result = rag_client.retrieve_documents(collection, q, n_results=n_docs)
        contexts = []
        ctx_string = ""
        if docs_result and docs_result.get('documents'):
            contexts = docs_result['documents'][0]
            ctx_string = rag_client.format_context(contexts, docs_result.get('metadatas', [[]])[0])

        # generate answer
        answer = llm_client.generate_response(openai_key, q, ctx_string, [], model="gpt-3.5-turbo")

        # evaluate
        sample = SingleTurnSample(input=q, output=answer, contexts=contexts)
        try:
            res = evaluate(samples=[sample], metrics=metrics, llm=evaluator_llm, embeddings=evaluator_embeddings)
        except Exception as e:
            res = {"error": str(e)}

        results_per_q.append({"question": q, "answer": answer, "ragas_result": res})

    # compute simple aggregates (mean) for numeric metrics if possible
    sums = {}
    counts = {}
    for item in results_per_q:
        r = item['ragas_result']
        if isinstance(r, list) and r:
            metrics_dict = r[0].get('metrics', {})
        elif isinstance(r, dict) and 'metrics' in r:
            metrics_dict = r.get('metrics', {})
        else:
            metrics_dict = {}

        for k, v in metrics_dict.items():
            try:
                val = float(v)
            except Exception:
                continue
            sums[k] = sums.get(k, 0.0) + val
            counts[k] = counts.get(k, 0) + 1

    aggregates = {k: (sums[k] / counts[k]) if counts.get(k) else None for k in sums.keys()}

    out = {"results": results_per_q, "aggregates": aggregates}
    try:
        with open(output_file, 'w', encoding='utf-8') as fh:
            json.dump(out, fh, indent=2)
    except Exception:
        pass

    return out
