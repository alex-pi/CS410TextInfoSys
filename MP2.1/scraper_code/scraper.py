from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from scraper_code.scrape_faculty import *
from scraper_code.scrape_util import write_lst


options = Options()
options.headless = True
driver = webdriver.Chrome('./chromedriver', options=options)

dir_url = 'https://polisci.mit.edu/people/faculty'
faculty_links = scrape_dir_page(dir_url, driver)
urls, bios = scrape_faculty_pages(faculty_links, driver)
print(bios)

driver.close()

bio_urls_file = '../bio_urls.txt'
bios_file = '../bios.txt'
write_lst(faculty_links, bio_urls_file)
write_lst(bios, bios_file)