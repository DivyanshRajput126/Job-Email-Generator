from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv
import os

load_dotenv()

class Chains:
    def __init__(self):
        self.llm = ChatGroq(model="llama-3.3-70b-versatile",groq_api_key = os.getenv("GROQ_API_KEY"),temperature=0.6)
    
    def extract_job(self,cleaned_text):
        prompt_extract = PromptTemplate.from_template(
            '''
            #SCrapped Text from Website:
            # {page_data}
            #Istructions:
            # The scrapped page is from carrier page of website 
            # your job is to extract job posting and return them in json format containing following keys: role, experience, skills, description.
            # only return valid json
            # valid json(no preamble)
            '''
        )   
        chain_extract = prompt_extract | self.llm
        res = chain_extract.invoke(input={"page_data":cleaned_text})
        try:
            json_parser = JsonOutputParser()
            res = json_parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException("Context too big. unable to parse job")
        return res if isinstance(res,list) else [res]
    
    def write_mail(self, job ,links):
        prompt_email = PromptTemplate.from_template(
            '''
            ### Job Description:
            {job_description}
            ### INSTRUCTION:
            You are Divyansh , a Student at Gujarat Technological University.
            Your job is to write a cold email to the client regarding the job mentioned above describing the capability of Name  
            in fulfilling their needs.
            Also add the most relevant ones from the following links to showcase Divyansh's portfolio: {link_list}
            Remember you are Divyansh, Student at Gujarat Technological University. 
            Do not provide a preamble.
            ### EMAIL (NO PREAMBLE):
            '''
        )
        chain_email = prompt_email | self.llm
        res = chain_email.invoke({"job_description":str(job),"link_list":links})
        return res.content

if __name__ == "__main__":
    print(os.getenv("GROQ_API_KEY"))