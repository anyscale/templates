# Enhanced RAG Deep Dive

In this tutorial, we walk through end-to-end Retrieval-Augmented Generation (RAG) pipelines using [Ray](https://docs.ray.io/), from data ingestion and LLM deployment to prompt engineering, evaluation and scaling out all workloads in the application.

## Notebooks

1. **01 (Optional) Regular_Document_Processing_Pipeline.ipynb**  
   Demonstrates a baseline document processing workflow for extracting, cleaning, and indexing text prior to RAG.

2. **02 Scalable_RAG_Data_Ingestion_with_Ray_Data.ipynb**  
   Shows how to build a high-throughput data ingestion pipeline for RAG using Ray Data.

3. **03 Deploy_LLM_with_Ray_Serve.ipynb**  
   Guides you through containerizing and serving a large language model at scale with Ray Serve.

4. **04 Build_Basic_RAG_Chatbot.ipynb**  
   Combines your indexed documents and served LLM to create a simple, interactive RAG chatbot.

5. **05 Improve_RAG_with_Prompt_Engineering.ipynb**  
   Explores prompt-engineering techniques to boost relevance and accuracy in RAG responses.

6. **06 (Optional) Evaluate_RAG_with_Online_Inference.ipynb**  
   Provides methods to assess RAG quality in real time via live queries and metrics tracking.

7. **07 Evaluate_RAG_with_Ray_Data_LLM_Batch_inference.ipynb**  
   Implements large-scale batch evaluation of RAG outputs using Ray Data + LLM batch inference.

---

> **Note:** Notebooks marked “(Optional)” cover complementary topics and can be skipped if you prefer to focus on the core RAG flow.
