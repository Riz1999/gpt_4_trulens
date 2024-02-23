import os
import pandas as pd
import streamlit as st
from openai import OpenAI
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from trulens_eval import Tru, TruCustomApp, Feedback, Select
from trulens_eval.feedback import Groundedness
from trulens_eval.feedback.provider.openai import OpenAI as fOpenAI
import numpy as np
class TruChain:
    def __init__(self, rag, pp_id, feedbacks):
        self.rag = rag
        self.pp_id = pp_id
        self.feedbacks = feedbacks

    def some_method(self):
        # Define some method here
        pass

def run_dashboard():
    # Here, you would run your Trulens dashboard on the specified port
    # For demonstration purposes, let's assume the dashboard URL is http://localhost:8502
    dashboard_url = f"https://smart-pugs-wonder.loca.lt/"
    return dashboard_url

# Define instrument decorator
def instrument(func):
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__}")
        result = func(*args, **kwargs)
        return result
    return wrapper

# Load your dataset containing lung cancer questions and answers
lung_cancer_data = pd.read_csv('CancerQA.csv')

# Set page title
st.set_page_config(page_title="RAG App")

# Define main function
def main():
    # Set title
    st.title("RAG App")

    # Display random lung cancer question and answer
    st.subheader("Lung Cancer Awareness BOT")
    
    with tru_rag as recording:
            query = st.text_input("Enter your query:", "What is AI")
        # Perform a query and display generated output
            generated_output = rag.query(query)
    # Take input query from the user
    

    # Execute further processes when the button is clicked
    if st.button("Execute"):
        
        st.subheader("Generated Output")
        st.write(generated_output)
            # Display evaluation results
        st.subheader("Evaluation Results")
        tru.get_leaderboard(app_ids=["RAG_v1"])
        
        # Run Trulens dashboard and get the URL to display
        dashboard_url = run_dashboard()
        
        # Display the URL on Streamlit frontend as a clickable link
        st.subheader("Trulens Dashboard URL")
        st.markdown(f"[Trulens Dashboard]({dashboard_url})")
    
# Select a random row from the dataset
random_entry = lung_cancer_data.sample(n=1, random_state=42)

# Extract information about the selected question and answer
lung_cancer_info = f"""
Question: {random_entry['Question'].values[0]}
Answer: {random_entry['Answer'].values[0]}
"""

# Set your OpenAI API key
# Extract information about the selected question and answer
os.environ["OPENAI_API_KEY"] = st.text_input("Enter your OpenAI API key:")

# Initialize OpenAI client
oai_client = OpenAI()

# Create embeddings for the lung cancer information
oai_client.embeddings.create(
    model="text-embedding-ada-002",
    input=lung_cancer_info
)

# Initialize ChromaDB client
chroma_client = chromadb.Client()

# Define embedding function using OpenAI
embedding_function = OpenAIEmbeddingFunction(api_key=os.environ.get('OPENAI_API_KEY'),
                                             model_name="text-embedding-ada-002")

# Get or create a collection in ChromaDB
vector_store = chroma_client.get_or_create_collection(name="LungCancer",
                                                      embedding_function=embedding_function)

# Add lung cancer information to the collection
vector_store.add("lung_cancer_info", documents=lung_cancer_info)

# Define RAG (Retrieval-Augmented Generation) class
class RAG_from_scratch:
    @instrument
    def retrieve(self, query: str) -> list:
        """
        Retrieve relevant text from vector store.
        """
        results = vector_store.query(
            query_texts=query,
            n_results=1
        )
        return results['documents'][0]

    @instrument
    def generate_completion(self, query: str, context_str: list) -> str:
        """
        Generate answer from context.
        """
        completion = oai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content":
                    f"We have provided context information below. \n"
                    f"---------------------\n"
                    f"{context_str}"
                    f"\n---------------------\n"
                    f"Given this information, please answer the question: {query}"
                }
            ]
        ).choices[0].message.content
        return completion

    @instrument
    def query(self, query: str) -> str:
        context_str = self.retrieve(query)
        completion = self.generate_completion(query, context_str)
        return completion

# Initialize Tru for evaluation
tru = Tru()

# Define feedback providers
fopenai = fOpenAI()
grounded = Groundedness(groundedness_provider=fopenai)

# Define feedback functions
f_groundedness = (
    Feedback(grounded.groundedness_measure_with_cot_reasons, name="Groundedness")
    .on(Select.RecordCalls.retrieve.rets.collect())
    .on_output()
    .aggregate(grounded.grounded_statements_aggregator)
)

f_qa_relevance = (
    Feedback(fopenai.relevance_with_cot_reasons, name="Answer Relevance")
    .on(Select.RecordCalls.retrieve.args.query)
    .on_output()
)

f_context_relevance = (
    Feedback(fopenai.qs_relevance_with_cot_reasons, name="Context Relevance")
    .on(Select.RecordCalls.retrieve.args.query)
    .on(Select.RecordCalls.retrieve.rets.collect())
    .aggregate(np.mean)
)

# Define RAG object
rag = RAG_from_scratch()

# Define TruCustomApp
tru_rag = TruCustomApp(rag,
                       app_id='RAG_v1',
                       feedbacks=[f_groundedness, f_qa_relevance, f_context_relevance])
truchain=TruChain(rag,pp_id='RAG_v1',
                       feedbacks=[f_groundedness, f_qa_relevance, f_context_relevance])
tru.run_dashboard()
if __name__ == "__main__":
    main()
