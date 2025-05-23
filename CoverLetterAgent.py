import streamlit as st
import requests
import tempfile
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from langchain.prompts import ChatPromptTemplate
from langchain.agents import initialize_agent, Tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_tavily import TavilySearch
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnableLambda, RunnableSequence


# Load environment vars (e.g. OPENAI_API_KEY)
load_dotenv()

# Init LLM
model = ChatOpenAI(model="gpt-4o-mini")

# Extract job description from posting text on URL
def extract_job_description(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    text = soup.get_text(separator=' ')
    return text[:3000] # Truncate so we don't hit token limits

# Parse uploaded resume PDF
def parse_resume(file):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(file.read())
        loader = PyPDFLoader(tmp.name)
        docs = loader.load_and_split()
    return docs

# Extract company name and website domain
def extract_company_info(inputs: dict) -> dict: #ChatPromptTemplate:

    job_description = inputs["job_description"]

    template = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a helpful assistant who only outputs your response in the given JSON format."
        ),
        (
            "human",
            """
            Given the following job description, extract the name of the company and its website domain.

            Job description:
            {job_description}

            Return your answer in JSON format:
            {{"company_name": "...", "website_domain": "..."}}
            """
        )
    ])

    chain = template | model | JsonOutputParser()
    company_info = chain.invoke({
        "job_description": job_description
    })

    return {
        **inputs,
        "company_info": company_info,
    }

# Search for company's values page
def find_company_values(inputs: dict) -> dict:
    company_info = inputs["company_info"]
    website_domain = company_info["website_domain"]
    company_name = company_info["company_name"]

    search = TavilySearch(
        max_results=5,
        include_domains=[website_domain]
    )

    tools = [
        Tool(
            name="Search Company Values",
            func=search.invoke,
            description="Useful for finding the company's values from its website"
        )
    ]

    agent = initialize_agent(tools, model, agent_type="zero-shot-react-description")
    company_values = agent.invoke(f"What are the core values of {company_name}? Give a brief, concise description of each.")

    return {
        **inputs,
        "company_values": company_values
    }

# Make template for prompt
def write_cover_letter(inputs: dict) -> dict:
    job_description = inputs["job_description"]
    relevant_experience = inputs["relevant_experience"]
    company_values = inputs["company_values"]

    template = ChatPromptTemplate.from_messages([
        (
            "system",
            """
            You are the applicant for this position, and can write cover letters to a professional degree.
            You are ambitious, driven and passionate to do this job, exuding that warmth while keeping a professional tone.
            """
        ),
        (
            "human",
            """
            Given the job description, the applicant's relevant experiences and the company values, write a personalized cover letter.

            Job Description:
            {job_description}

            Relevant Resume Excerpts:
            {relevant_experience}

            Company Values:
            {company_values}

            The letter should be structured professionally, reference key responsibilities or values from the job, and fit on one page.
            Make sure the experiences are relevant, and where possible link them to key customer values, although avoid this if the connection is tenuous.
            Regarding relevant experience and company values, paraphrase instead of using the vocabulary from the resume.
            """
        )
    ])

    chain = template | model | StrOutputParser()
    cover_letter = chain.invoke({
        "job_description": job_description,
        "relevant_experience": relevant_experience,
        "company_values": company_values
    })

    return {
        **inputs,
        "cover_letter": cover_letter
    }

# Streamlit UI
st.set_page_config(page_title="Cover Letter Writer", layout="centered")
st.title("Cover Letter Writer")
st.write("Punch in a job posting URL and upload your resume, and the agent will suggest a personalized cover letter!")
st.write("Note: this was built simply as a proof-of-concept app. We do not condone using this for actual job applications, and believe the best cover letters are handcrafted with love.")

job_url = st.text_input("Job Posting URL:")
uploaded_file = st.file_uploader("Upload your resume here (PDFs only):", type=["pdf"])

if st.button("Generate Cover Letter") and job_url and uploaded_file:
    if not job_url:
        st.error("Please enter a job posting url to generate a reply.")
    elif not uploaded_file:
        st.error("Please upload a resume.")
    else:
        with st.spinner("Processing..."):
            # 1. Extract job description
            job_text = extract_job_description(job_url)

            # 2. Parse Resume
            resume_docs = parse_resume(uploaded_file)

            # 3. Vector store to find relevant resume information
            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.from_documents(resume_docs, embeddings)
            relevant_experiences = vectorstore.similarity_search(job_text, k=5)
            relevant_exp_text = "\n\n".join([doc.page_content for doc in relevant_experiences])

            # 4. Define pipeline
            pipeline = RunnableSequence(
                RunnableLambda(extract_company_info)
                | RunnableLambda(find_company_values)
                | RunnableLambda(write_cover_letter)
            )

            # 5. Create prompt and generate cover letter
            try:
                inputs = {
                    "job_description": job_text,
                    "relevant_experience": relevant_exp_text
                }

                output = pipeline.invoke(inputs)
                st.subheader("Suggested Cover Letter:")
                st.write(output["cover_letter"])

            except Exception as e:
                st.error(f"Error: {e}")

# Footer
st.markdown("---")
st.markdown("Powered by Keenan's job anxiety. Don't use this for job applications, kids!")