
import os
from http import cookies
import scrapy
from scrapy import Spider
from scrapy.http import FormRequest
from scrapy_selenium import SeleniumRequest
from scrapy_playwright.page import PageMethod
from basic_scrapy_spider.items import CompanyListing
from bs4 import BeautifulSoup
import openai
from openai import OpenAI
from chunkipy import TextChunker, TokenEstimator

import callOpenaiEmbeddings as callOpenAi

import textChunker

client = OpenAI()

def read_file(filename):
    with open (filename, 'r') as f:
        file_content = f.read()
    return file_content

filename = 'fully_rendered_html.html'
myResume = '/Users/omkarpatil/Resume_Automator_Repo/Resume-Automator-Curate-your-resume-wrt-the-job-description/Resume.txt'
outputFile = "CateredResume_v1(openai_embeddings).txt"
scrapped_text1 = 'scraped_text1.txt'

def write_to_file(content,filename):
    with open (filename, "w" ) as f:
        f.write(content)

robert_token_estimator = textChunker.RobertaTokenEstimator()

#print(type(chunks))

scrolling_script = """
    const scrolls = 12
    let scrollCount = 0
    
    // scroll down and then wait for 0.5s
    const scrollInterval = setInterval(() => {
      window.scrollTo(0, document.body.scrollHeight)
      scrollCount++
    
      if (scrollCount === numScrolls) {
        clearInterval(scrollInterval)
      }
    }, 500)
    """


text = read_file('/Users/omkarpatil/Resume_Automator_Repo/Resume-Automator-Curate-your-resume-wrt-the-job-description/basic_scrapy_spider/spiders/scraped_text1.txt')
#text = read_file('/content/sample_data/text1.txt')

class JobListingSpider(Spider):
    name = 'jobListing_spider'

    def start_requests(self):
        job_url = 'https://jobs.citi.com/job/-/-/287/57915229664?utm_term=326267118&ss=paid&utm_campaign=NAM_lateral&utm_medium=job_posting&source=linkedinJB&utm_source=linkedin.com&utm_content=social_media&dclid=CO3tmqXcvoQDFdJTRwEd99oF-A'
        #job_url = 'https://careers.doordash.com/jobs/5698581'
        yield scrapy.Request(job_url, callback=self.start_scraping)

    
    async def start_scraping(self, response):
        write_to_file(response.text,filename)
        soup = BeautifulSoup(response.text, 'html.parser')
        part_1 = soup.get_text(separator='\n')
        print(f"Num of chars: {len(part_1)}")
        print(f"Num of tokens (using RoBertTokenEstimator): {robert_token_estimator.estimate_tokens(part_1)}")

        # This creates an instance of the TextChunker class with a chunk size of 512,
        # using token-based segmentation and a custom tokenizer function bert_tokenizer.encode to count tokens.

        text_chunker = TextChunker(512, tokens=True, token_estimator=robert_token_estimator)
        chunks = text_chunker.chunk(part_1)

        contextEmbeddings = callOpenAi.create_context_embeddings(chunks)
        queryEmbeddings = callOpenAi.create_query_embeddings()

        similarity_distances = callOpenAi.similarity_search(contextEmbeddings,queryEmbeddings)
        
        # Print out the similar chunks 
        #for i in range(len(similarity_distances)) :
            #print(f'most_similar_chunk{i} -->',chunks[similarity_distances[i][1]]) 

        write_to_file(part_1,scrapped_text1)

        part_2 = " \n The above text is the job that I want to apply for with all its job description and down below is my resume \n"

        ########################################### the below code sends the all the above parts to LLM ###########################################
        part_3 = read_file(myResume)
        userPrompt = chunks[similarity_distances[0][1]] + part_2 + part_3
        completion = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a Resume builder, skilled in updating current resume considering the job requirements."},
                {"role": "user", "content": f'{userPrompt} \n Now with all the context provided updated resume. 2.Add some resume points using the "key-words" in the job description '}
            ]
            )
        #message = str (completion.choices[0].message)
        write_to_file(completion.choices[0].message.content,outputFile)


    async def errback(self , failure):
        print("----------------------- FAILURE PAGE CLOSED --------------------------")
        page = failure.request.meta["playwright_page"]
        await page.close()
        

               




