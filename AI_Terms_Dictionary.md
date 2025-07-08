# AI Terms Dictionary
*Quick reference for explaining technical terms to non-technical audiences*

---

## **Terms in Order of Appearance (flow.md)**

### **KOL (Key Opinion Leader)**
- What it is: Influential experts in a field
- In our system: ~100 trusted crypto/finance experts we learn from
- Simple analogy: Like the most respected professors in a university

### **Twitter API**
- What it is: Twitter's interface for programmatically accessing data
- In our system: Tweepy library collects tweets from KOL accounts
- Simple analogy: Like a special key that lets us automatically read Twitter

### **Raw Tweets (JSON)**
- What it is: Unprocessed Twitter data in structured format
- In our system: Initial data before any filtering or processing
- Simple analogy: Like raw ingredients before cooking

### **Automated Filters**
- What it is: Software that automatically removes bad content
- In our system: Checks language, toxicity, bot scores, and perplexity
- Simple analogy: Like airport security screening passengers

### **Language Check**
- What it is: Verifying content is in the target language
- In our system: fastText identifies non-English tweets
- Simple analogy: Like checking if a book is in the right language

### **Toxicity Filter**
- What it is: Identifying harmful or inappropriate content
- In our system: Google Perspective API scores content toxicity
- Simple analogy: Like a content filter that blocks inappropriate websites

### **Bot Detection**
- What it is: Identifying fake or automated accounts
- In our system: Botometer-Lite filters out bot-generated content
- Simple analogy: Like a bouncer checking IDs to spot fake ones

### **Perplexity Check**
- What it is: Measure of how predictable or complex text is
- In our system: GPT-2 PPL filters overly simple or complex content
- Simple analogy: Like checking if a sentence makes sense or is gibberish

### **Bootstrap Phase**
- What it is: Initial learning period with heavy human oversight
- In our system: Humans review all early outputs to ensure quality
- Simple analogy: Like teaching a child to ride a bike with training wheels

### **Human Review**
- What it is: People checking AI outputs for quality
- In our system: Required during bootstrap and early phases
- Simple analogy: Like having an editor review articles before publication

### **Token Chunks**
- What it is: Breaking text into smaller, manageable pieces
- In our system: 256-token chunks for processing
- Simple analogy: Like breaking a long sentence into individual words

### **Embeddings**
- What it is: Converting text into numbers that capture meaning
- In our system: bge-large-en model converts tweets to vectors
- Simple analogy: Like translating words into a universal language computers understand

### **Vector Database**
- What it is: Storage system optimized for finding similar vectors
- In our system: Qdrant + cuVS for fast similarity search
- Simple analogy: Like a smart filing cabinet that groups similar documents together

### **RAG (Retrieval-Augmented Generation)**
- What it is: AI system that looks up information before generating responses
- In our system: Finds relevant tweets before writing new ones
- Simple analogy: Like doing research before writing an essay

### **Query**
- What it is: The question or task given to the AI
- In our system: The prompt that triggers content generation
- Simple analogy: Like asking a librarian to find books on a topic

### **Cross-Encoder**
- What it is: A model that ranks search results by relevance
- In our system: ColBERT-v2 improves search accuracy by 10-15%
- Simple analogy: Like a smart librarian who knows exactly which books are most relevant

### **Top-k Context**
- What it is: The most relevant pieces of information retrieved
- In our system: Best matching tweets used to inform generation
- Simple analogy: Like the most relevant pages from a library search

### **LLM (Large Language Model)**
- What it is: AI model that understands and generates human language
- In our system: Mistral-7B base model for tweet generation
- Simple analogy: Like a very sophisticated autocomplete that understands context

### **LoRA (Low-Rank Adaptation)**
- What it is: Efficient way to update AI models with new knowledge
- In our system: Daily micro-updates using only 2% of parameters
- Simple analogy: Like adding sticky notes to a textbook instead of rewriting it

### **Self-RAG**
- What it is: AI technique where the model retrieves evidence and critiques itself
- In our system: Reduces hallucinations by ~30%
- Simple analogy: Like writing a draft, then fact-checking it before publishing

### **Draft Tweet**
- What it is: First version of content before review
- In our system: Initial output from the AI before human/AI review
- Simple analogy: Like a rough draft of an essay before editing

### **AI Peer Review**
- What it is: Another AI system critiquing the output
- In our system: Raw GPT-4o provides automatic feedback
- Simple analogy: Like having a very knowledgeable friend review your work

### **Manual Edit**
- What it is: Human modification of AI-generated content
- In our system: When humans need to fix or improve drafts
- Simple analogy: Like a teacher correcting a student's essay

### **Twitter Metrics**
- What it is: Engagement data from posted content
- In our system: Views, likes, reposts used as feedback signals
- Simple analogy: Like applause at a performance - tells you what works

### **Reward Model**
- What it is: System that scores AI outputs based on desired outcomes
- In our system: Combines human feedback, AI review, and Twitter metrics
- Simple analogy: Like a judge scoring performances in a competition

### **PPO (Proximal Policy Optimization)**
- What it is: Algorithm that improves AI behavior based on feedback
- In our system: Weekly updates using TRL library
- Simple analogy: Like a coach adjusting strategy based on game results

### **TRL (Transformer Reinforcement Learning)**
- What it is: Hugging Face library for training AI with reinforcement learning
- In our system: Implements PPO algorithm
- Simple analogy: Like a specialized toolkit for training AI models

### **Fine-tuning**
- What it is: Teaching an AI model new skills without starting over
- In our system: Daily LoRA updates improve performance
- Simple analogy: Like taking advanced driving lessons after getting your license

### **Dashboard**
- What it is: Visual display of system performance metrics
- In our system: Prometheus + Grafana shows real-time system health
- Simple analogy: Like a car's dashboard showing speed, fuel, and engine status

### **Prometheus**
- What it is: Open-source monitoring and alerting system
- In our system: Collects and stores performance metrics
- Simple analogy: Like a security camera system that records everything

### **RAGAS**
- What it is: Evaluation framework for RAG systems
- In our system: Measures accuracy, faithfulness, and relevance
- Simple analogy: Like a quality control inspector for AI outputs

### **Faithfulness**
- What it is: How well the AI sticks to factual information
- In our system: Measured by RAGAS to prevent hallucinations
- Simple analogy: Like checking if a student's essay accurately reflects the source material

### **Brand Drift**
- What it is: When AI output becomes inconsistent with intended personality
- In our system: Monitored to maintain consistent voice
- Simple analogy: Like a person's personality changing unexpectedly

---

## **Additional Important Terms**

### **HNSW (Hierarchical Navigable Small World)**
- What it is: Algorithm for fast similarity search in vector databases
- In our system: Used for efficient retrieval when database is under 10M vectors
- Simple analogy: Like a smart GPS that finds the fastest route between points

### **IVF-PQ (Inverted File with Product Quantization)**
- What it is: Advanced indexing method for large vector databases
- In our system: Used when database grows beyond 10M vectors
- Simple analogy: Like upgrading from a small filing cabinet to a large warehouse system

### **GPU Search**
- What it is: Using graphics processors for faster computation
- In our system: Accelerates vector similarity searches
- Simple analogy: Like using a sports car instead of a regular car for deliveries

### **Precision@5**
- What it is: Measure of how accurate the top 5 search results are
- In our system: Key metric for retrieval quality, target >95%
- Simple analogy: Like checking if the first 5 books in a search are actually relevant

### **Recall@10**
- What it is: Measure of how much relevant information is found in top 10 results
- In our system: Complementary metric to precision
- Simple analogy: Like checking if you found most of the relevant books in the first 10 results

### **Context-Precision**
- What it is: How well retrieved information matches the query context
- In our system: RAGAS metric for measuring retrieval relevance
- Simple analogy: Like checking if the information found actually answers the question

### **Open-weights**
- What it is: AI models whose internal structure is publicly available
- In our system: Mistral-7B and Llama-3 8B are open-weights models
- Simple analogy: Like open-source software where you can see how it works

### **Rank (r)**
- What it is: Number that determines LoRA's learning capacity
- In our system: Rank 16, meaning 16 new parameters per layer
- Simple analogy: Like the number of channels on a TV - more channels, more information

### **Alpha (α)**
- What it is: Scaling factor that controls LoRA's influence
- In our system: α = 2 × r (32), balances learning vs stability
- Simple analogy: Like the volume knob - controls how strongly the new learning affects the model

### **AdamW**
- What it is: Optimization algorithm for training neural networks
- In our system: Learning rate optimizer for LoRA training
- Simple analogy: Like a smart thermostat that adjusts temperature efficiently

### **Learning Rate**
- What it is: How big steps the AI takes when learning
- In our system: 1 × 10⁻⁴ (very small steps to avoid overfitting)
- Simple analogy: Like how big steps you take when learning to walk - too big and you fall

### **Epochs**
- What it is: Number of times the AI sees the same training data
- In our system: 1 epoch to avoid overfitting
- Simple analogy: Like reading a textbook once vs multiple times

### **Adapter Sprawl**
- What it is: Problem of having too many LoRA adapters
- In our system: Prevented by merging adapters every 4-6 weeks
- Simple analogy: Like having too many sticky notes on a book - it becomes unmanageable

### **Reflexion**
- What it is: AI technique where the model critiques its own work
- In our system: Part of Self-RAG process
- Simple analogy: Like proofreading your own writing before submitting

### **Hallucination**
- What it is: When AI makes up false information
- In our system: Self-RAG reduces this by ~30%
- Simple analogy: Like a student confidently writing an essay about facts they imagined

### **CTR (Click-Through Rate)**
- What it is: Percentage of people who click on content
- In our system: Part of engagement metrics for reward model
- Simple analogy: Like measuring how many people open a door when you put up a sign

### **Follower Delta**
- What it is: Change in follower count over time
- In our system: Positive signal for reward model
- Simple analogy: Like measuring if your audience is growing or shrinking

### **Off-topic**
- What it is: Content that doesn't match the intended subject
- In our system: Negative signal in reward model
- Simple analogy: Like talking about cooking in a car repair class

### **Safety Valve**
- What it is: Automatic shutdown when quality drops below threshold
- In our system: Blocks posting if faithfulness < 0.9 or toxicity > 0.6
- Simple analogy: Like a circuit breaker that shuts off power when there's a problem

### **Importance Sampling**
- What it is: Technique for learning from past experiences
- In our system: 1-step importance sampling in PPO updates
- Simple analogy: Like learning from your mistakes by focusing on what went wrong

### **Neo4j**
- What it is: Graph database for storing relationships
- In our system: Stores immutable knowledge (like LBMA rules)
- Simple analogy: Like a family tree that shows how everything is connected

### **Knowledge Graph**
- What it is: Database that stores facts and their relationships
- In our system: Contains critical, unchanging information
- Simple analogy: Like an encyclopedia that shows how facts relate to each other

### **LBMA Rules**
- What it is: London Bullion Market Association standards
- In our system: Example of immutable knowledge stored in knowledge graph
- Simple analogy: Like the rules of chess that never change

### **Synthetic User Simulator**
- What it is: AI system that generates fake user interactions
- In our system: Future enhancement for pre-training behavior policy
- Simple analogy: Like a flight simulator that lets pilots practice before flying real planes

### **Multi-lingual Switch**
- What it is: Ability to work in multiple languages
- In our system: Future enhancement with language-specific adapters
- Simple analogy: Like a universal translator that can switch between languages

### **On-device Summarizer**
- What it is: Local system that creates summaries
- In our system: Future enhancement for trend reports
- Simple analogy: Like having a personal assistant who reads everything and gives you the highlights

---

*This dictionary covers all technical terms mentioned in the flow.md document, organized in order of appearance, plus additional important concepts for comprehensive coverage.* 