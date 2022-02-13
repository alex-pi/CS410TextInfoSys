import scrapy
from Quotes.items import QuotesItem

class QuotesSpider(scrapy.Spider):
    name = 'quotes'
    allowed_domains = ['quotes.toscrape.com']
    start_urls = ['http://quotes.toscrape.com/']

    '''
    Using XPath
    '''
    def parse(self, response):
        print("Response Type >>> ", type(response))
        rows = response.xpath("//div[@class='quote']") #root element

        print("Quotes Count >> ", rows.__len__())
        for row in rows:
            item = QuotesItem()

            item['tags'] = row.xpath('div[@class="tags"]/meta[@itemprop="keywords"]/@content').extract_first().strip()
            item['author'] = row.xpath('//span/small[@itemprop="author"]/text()').extract_first()
            item['quote'] = row.xpath('span[@itemprop="text"]/text()').extract_first()
            item['author_link'] = row.xpath('//a[contains(@href,"/author/")]/@href').extract_first()

            if len(item['author_link'])>0:
                item['author_link'] = 'http://quotes.toscrape.com'+item['author_link']

            yield item
        #using CSS
        nextPage = response.css("ul.pager > li.next > a::attr(href)").extract_first()
        #using XPath
        #nextPage = response.xpath("//ul[@class='pager']//li[@class='next']/a/@href").extract_first()

        if nextPage:
            print("Next Page URL: ", nextPage)
            #nextPage obtained from either XPath or CSS can be used.
            yield scrapy.Request('http://quotes.toscrape.com'+nextPage, callback=self.parse)

    print('Completed')