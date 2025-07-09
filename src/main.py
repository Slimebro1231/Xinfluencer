"""Main entry point for Xinfluencer AI."""

import logging
import sys
import time
from pathlib import Path

# Add the project's 'src' directory to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

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

# Database module
from database.manager import initialize_db, log_result


class PipelineService:
    """A long-running service for the Xinfluencer AI pipeline."""

    def __init__(self, run_interval_seconds=60):
        """Initialize the pipeline service and its components."""
        self.run_interval_seconds = run_interval_seconds
        self.logger = setup_logger(
            name="xinfluencer_ai",
            level="INFO",
            log_file="logs/xinfluencer.log"
        )

        self.logger.info("🛠️ Initializing pipeline components...")
        self.quality_gate = QualityGate()
        self.text_embedder = TextEmbedder()
        self.vector_db = VectorDB()
        self.text_generator = TextGenerator()
        self.vector_searcher = VectorSearcher(self.vector_db, self.text_embedder)
        self.ai_reviewer = AIReviewer(self.text_generator)
        self.selfrag_generator = SelfRAGGenerator(
            self.text_generator, self.vector_searcher, self.ai_reviewer
        )
        self.lora_tuner = LoRAFineTuner()
        
        self.vector_db.create_collection()
        initialize_db()  # Initialize the results database
        self.logger.info("✅ All pipeline components initialized successfully.")

    def run_cycle(self):
        """Run one complete cycle of the Xinfluencer AI pipeline."""
        self.logger.info("🚀 Starting new pipeline cycle...")

        try:
            # Step 1: Data ingestion (MODIFIED FOR TESTING)
            self.logger.info("📥 SKIPPING tweet fetching for testing and using mock data.")
            # tweets = fetch_tweets()
            tweets = [
                {"id": "1", "text": "This is a test tweet about crypto and blockchain.", "created_at": "2025-07-09T12:00:00Z"},
                {"id": "2", "text": "What are the latest advancements in DeFi?", "created_at": "2025-07-09T12:05:00Z"},
                {"id": "3", "text": "Elon Musk just tweeted about Dogecoin again!", "created_at": "2025-07-09T12:10:00Z"},
            ]
            self.logger.info(f"📊 Using {len(tweets)} mock tweets for pipeline run.")

            # Step 2: Quality filtering
            self.logger.info("🔍 Running quality gate filters...")
            filtered = self.quality_gate.filter(tweets)
            self.logger.info(f"✅ {len(filtered)} tweets passed quality gate")

            # Step 3: Text chunking
            self.logger.info("✂️ Chunking tweets for embedding...")
            all_chunks = []
            for tweet in filtered:
                chunks = chunk_text(tweet["text"])
                for chunk in chunks:
                    chunk["tweet_id"] = tweet["id"]
                    chunk["timestamp"] = tweet["created_at"]
                    all_chunks.append(chunk)

            self.logger.info(f"📝 Generated {len(all_chunks)} text chunks")

            # Step 4: Vector embedding
            self.logger.info("🔗 Generating embeddings...")
            embedded_chunks = self.text_embedder.embed_chunks(all_chunks)
            self.logger.info(f"✨ Generated embeddings for {len(embedded_chunks)} chunks")

            # Step 5: Vector database storage
            self.logger.info("💾 Storing chunks in vector database...")
            self.vector_db.upsert_chunks(embedded_chunks)
            self.logger.info("✅ Chunks stored in vector database")

            # Step 6: Demo generation and review
            self.logger.info("🎯 Running demo generation...")
            demo_queries = [
                "What's the latest trend in crypto?",
                "How should I invest in Bitcoin?",
                "What are the best DeFi protocols?"
            ]

            demo_results = []
            for query in demo_queries:
                self.logger.info(f"Processing query: {query}")

                # Generate with Self-RAG
                selfrag_result = self.selfrag_generator.generate_with_self_rag(query)

                # AI Review
                review_result = self.ai_reviewer.review_response(
                    query,
                    selfrag_result["best_response"],
                    selfrag_result["iterations"][-1].get("context", "")
                )

                # Log the result to the database
                log_result(
                    query=query,
                    response=selfrag_result["best_response"],
                    selfrag_score=selfrag_result["best_score"],
                    review_score=review_result["overall_score"],
                    approved=review_result["approved"],
                    context=selfrag_result["iterations"][-1].get("context", "")
                )
                
                demo_results.append({
                    "query": query,
                    "selfrag_result": selfrag_result,
                    "review_result": review_result
                })

                self.logger.info(f"Query processed. Self-RAG score: {selfrag_result['best_score']:.1f}, "
                                f"Review score: {review_result['overall_score']:.1f}")

            # Step 7: LoRA fine-tuning preparation (demo)
            self.logger.info("🔧 Preparing LoRA fine-tuning...")
            
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
                self.logger.info(f"Prepared {len(training_data)} samples for LoRA training")
                # Note: Actual fine-tuning would be resource-intensive
                # lora_adapter_path = self.lora_tuner.fine_tune(training_data)
                self.logger.info("LoRA fine-tuning preparation completed (actual training skipped for demo)")

            # Step 8: Results summary
            self.logger.info("📋 Pipeline Cycle Results Summary:")
            self.logger.info(f"  • Tweets processed: {len(tweets)}")
            self.logger.info(f"  • Tweets after filtering: {len(filtered)}")
            self.logger.info(f"  • Text chunks generated: {len(all_chunks)}")
            self.logger.info(f"  • Embeddings created: {len(embedded_chunks)}")
            self.logger.info(f"  • Demo queries processed: {len(demo_queries)}")

            approved_count = sum(1 for r in demo_results if r["review_result"]["approved"])
            self.logger.info(f"  • Approved responses: {approved_count}/{len(demo_results)}")

            avg_selfrag_score = sum(r["selfrag_result"]["best_score"] for r in demo_results) / len(demo_results)
            avg_review_score = sum(r["review_result"]["overall_score"] for r in demo_results) / len(demo_results)

            self.logger.info(f"  • Average Self-RAG score: {avg_selfrag_score:.1f}")
            self.logger.info(f"  • Average review score: {avg_review_score:.1f}")

        except Exception as e:
            self.logger.error(f"❌ Pipeline cycle failed with error: {e}")
            self.logger.exception("Full traceback:")

        self.logger.info("✅ Pipeline cycle finished.")

    def start(self):
        """Start the long-running pipeline service."""
        self.logger.info("🔥 Xinfluencer AI Service starting...")
        try:
            while True:
                self.run_cycle()
                self.logger.info(f"😴 Service sleeping for {self.run_interval_seconds} seconds...")
                time.sleep(self.run_interval_seconds)
        except KeyboardInterrupt:
            self.logger.info("🛑 Service received shutdown signal.")
        finally:
            self.logger.info("👋 Xinfluencer AI Service shutting down.")
            # Add any cleanup logic here, e.g., closing DB connections
            pass

def main():
    """Create and start the Xinfluencer AI service."""
    service = PipelineService()
    service.start()


if __name__ == "__main__":
    main()