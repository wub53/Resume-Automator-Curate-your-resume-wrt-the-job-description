# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class CompanyListing(scrapy.Item):
    # define the fields for your item here like:
    name = scrapy.Field()
    location = scrapy.Field()
    industry = scrapy.Field()
    revenue = scrapy.Field()
    ebitda = scrapy.Field()
