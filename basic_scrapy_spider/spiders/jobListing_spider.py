
import os
from http import cookies
import scrapy
from scrapy import Spider
from scrapy.http import FormRequest
from scrapy_selenium import SeleniumRequest
from scrapy_playwright.page import PageMethod
from basic_scrapy_spider.items import CompanyListing
from bs4 import BeautifulSoup
from openai import OpenAI
client = OpenAI()

# request = scrapy.Request( login_url, method='POST', 
#                           body=json.dumps(my_data), 
#                           headers={'Content-Type':'application/json'} )

filename = 'fully_rendered_html.html'
myResume = 'Resume.txt'
outputFile = "CateredResume.txt"

def write_to_file(content,filename):
    with open (filename, "w" ) as f:
        f.write(content)

def read_file(filename):
    with open (filename, 'r') as f:
        file_content = f.read()
    return file_content

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

class JobListingSpider(Spider):
    name = 'jobListing_spider'

    def start_requests(self):
        #login_url = 'https://careers.doordash.com/jobs/5698581'
        job_url = 'https://jobs.citi.com/job/-/-/287/57915229664?utm_term=326267118&ss=paid&utm_campaign=NAM_lateral&utm_medium=job_posting&source=linkedinJB&utm_source=linkedin.com&utm_content=social_media&dclid=CO3tmqXcvoQDFdJTRwEd99oF-A'
        yield scrapy.Request(job_url, callback=self.start_scraping)

    
    async def start_scraping(self, response):
        write_to_file(response.text,filename)
        soup = BeautifulSoup(response.text, 'html.parser')
        #element = soup.find(class_='w-richtext')
        part_1 = soup.get_text() 
        part_2 = " \n The above text is the job that I want to apply for with all its job description and down below is my resume \n"
        part_3 = read_file(myResume)
        userPrompt = part_1 + part_2 + part_3
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
        

               




