"""Main entry point for Xinfluencer AI."""

import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Data modules
from data.ingest import fetch_tweets
from data.filter import QualityGate
from data.chunk import chunk_text

# Vector modules
from vector.embed import TextEmbedder
from vector.db import VectorDB
from vector.search import VectorSearcher

# Model modules
from model.generate import TextGenerator
from model.selfrag import SelfRAGGenerator
from model.lora import LoRAFineTuner

# Review modules
from review.ai import AIReviewer

# Utility modules
from utils.logger import setup_logger

def main():
    """Run the complete Xinfluencer AI pipeline."""
    # Setup logging
    logger = setup_logger(
        name="xinfluencer_ai",
        level="INFO",
        log_file="logs/xinfluencer.log"
    )
    
    logger.info("🚀 Starting Xinfluencer AI pipeline...")
    
    try:
        # Step 1: Data ingestion
        logger.info("📥 Fetching tweets from KOL accounts...")
        tweets = fetch_tweets()
        logger.info(f"📊 Retrieved {len(tweets)} tweets")
        
        # Step 2: Quality filtering
        logger.info("🔍 Running quality gate filters...")
        gate = QualityGate()
        filtered = gate.filter(tweets)
        logger.info(f"✅ {len(filtered)} tweets passed quality gate")
        
        # Step 3: Text chunking
        logger.info("✂️ Chunking tweets for embedding...")
        all_chunks = []
        for tweet in filtered:
            chunks = chunk_text(tweet["text"])
            for chunk in chunks:
                chunk["tweet_id"] = tweet["id"]
                chunk["timestamp"] = tweet["created_at"]
                all_chunks.append(chunk)
        
        logger.info(f"📝 Generated {len(all_chunks)} text chunks")
        
        # Step 4: Vector embedding
        logger.info("🔗 Generating embeddings...")
        embedder = TextEmbedder()
        embedded_chunks = embedder.embed_chunks(all_chunks)
        logger.info(f"✨ Generated embeddings for {len(embedded_chunks)} chunks")
        
        # Step 5: Vector database storage
        logger.info("💾 Storing chunks in vector database...")
        vector_db = VectorDB()
        vector_db.create_collection()
        vector_db.upsert_chunks(embedded_chunks)
        logger.info("✅ Chunks stored in vector database")
        
        # Step 6: Initialize generation components
        logger.info("🧠 Initializing AI components...")
        generator = TextGenerator()
        searcher = VectorSearcher(vector_db, embedder)
        selfrag_generator = SelfRAGGenerator(generator, searcher)
        ai_reviewer = AIReviewer(generator)
        
        # Step 7: Demo generation and review
        logger.info("🎯 Running demo generation...")
        demo_queries = [
            "What's the latest trend in crypto?",
            "How should I invest in Bitcoin?",
            "What are the best DeFi protocols?"
        ]
        
        demo_results = []
        for query in demo_queries:
            logger.info(f"Processing query: {query}")
            
            # Generate with Self-RAG
            selfrag_result = selfrag_generator.generate_with_self_rag(query)
            
            # AI Review
            review_result = ai_reviewer.review_response(
                query,
                selfrag_result["best_response"],
                selfrag_result["iterations"][-1].get("context", "")
            )
            
            demo_results.append({
                "query": query,
                "selfrag_result": selfrag_result,
                "review_result": review_result
            })
            
            logger.info(f"Query processed. Self-RAG score: {selfrag_result['best_score']:.1f}, "
                       f"Review score: {review_result['overall_score']:.1f}")
        
        # Step 8: LoRA fine-tuning preparation (demo)
        logger.info("🔧 Preparing LoRA fine-tuning...")
        lora_tuner = LoRAFineTuner()
        
        # Prepare training data from approved responses
        training_data = []
        for result in demo_results:
            if result["review_result"]["approved"]:
                training_data.append({
                    "query": result["query"],
                    "response": result["selfrag_result"]["best_response"],
                    "approved": True
                })
        
        if training_data:
            logger.info(f"Prepared {len(training_data)} samples for LoRA training")
            # Note: Actual fine-tuning would be resource-intensive
            # lora_adapter_path = lora_tuner.fine_tune(training_data)
            logger.info("LoRA fine-tuning preparation completed (actual training skipped for demo)")
        
        # Step 9: Results summary
        logger.info("📋 Pipeline Results Summary:")
        logger.info(f"  • Tweets processed: {len(tweets)}")
        logger.info(f"  • Tweets after filtering: {len(filtered)}")
        logger.info(f"  • Text chunks generated: {len(all_chunks)}")
        logger.info(f"  • Embeddings created: {len(embedded_chunks)}")
        logger.info(f"  • Demo queries processed: {len(demo_queries)}")
        
        approved_count = sum(1 for r in demo_results if r["review_result"]["approved"])
        logger.info(f"  • Approved responses: {approved_count}/{len(demo_results)}")
        
        avg_selfrag_score = sum(r["selfrag_result"]["best_score"] for r in demo_results) / len(demo_results)
        avg_review_score = sum(r["review_result"]["overall_score"] for r in demo_results) / len(demo_results)
        
        logger.info(f"  • Average Self-RAG score: {avg_selfrag_score:.1f}")
        logger.info(f"  • Average review score: {avg_review_score:.1f}")
        
        logger.info("✨ Xinfluencer AI pipeline completed successfully!")
        
        return {
            "status": "success",
            "tweets_processed": len(tweets),
            "tweets_filtered": len(filtered),
            "chunks_generated": len(all_chunks),
            "demo_results": demo_results,
            "avg_selfrag_score": avg_selfrag_score,
            "avg_review_score": avg_review_score
        }
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed with error: {e}")
        logger.exception("Full traceback:")
        return {
            "status": "error",
            "error": str(e)
        }

if __name__ == "__main__":
    result = main()
    if result["status"] == "error":
        exit(1) 