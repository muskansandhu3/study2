from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from typing import Dict, List, Optional

# RAGAS imports
try:
    from ragas import SingleTurnSample
    from ragas.metrics import BleuScore, NonLLMContextPrecisionWithReference, ResponseRelevancy, Faithfulness, RougeScore
    from ragas import evaluate
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False

def evaluate_response_quality(question: str, answer: str, contexts: List[str]) -> Dict[str, float]:
    """Evaluate response quality using RAGAS metrics"""
    if not RAGAS_AVAILABLE:
        return {"error": "RAGAS not available"}
    
    try:
        # Create wrappers for RAGAS
        evaluator_llm = LangchainLLMWrapper(llm=ChatOpenAI(temperature=0.0, model_name="gpt-3.5-turbo"))
        evaluator_embeddings = LangchainEmbeddingsWrapper(embeddings=OpenAIEmbeddings(model="text-embedding-3-small"))

        # Define metrics
        metrics = [ResponseRelevancy(), Faithfulness()]

        sample = SingleTurnSample(input=question, output=answer, contexts=contexts)

        results = evaluate(samples=[sample], metrics=metrics, llm=evaluator_llm, embeddings=evaluator_embeddings)

        # results is expected to be a list/dict; normalize to metric->value
        out = {}
        if isinstance(results, list) and results:
            # take first sample's metrics
            sample_res = results[0]
            for k, v in sample_res.get('metrics', {}).items():
                out[k] = v
        elif isinstance(results, dict):
            out = results

        return out
    except Exception as e:
        return {"error": str(e)}
