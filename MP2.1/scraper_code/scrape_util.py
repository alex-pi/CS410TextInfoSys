from bs4 import BeautifulSoup
from urllib.parse import urlparse
import re
import urllib


# uses webdriver object to execute javascript code and get dynamically loaded webcontent
def get_js_soup(url, driver):
    driver.get(url)
    res_html = driver.execute_script('return document.body.innerHTML')
    soup = BeautifulSoup(res_html, 'html.parser')  # beautiful soup object to be used for parsing html content
    return soup


# tidies extracted text
def process_bio(bio):
    bio = bio.encode('ascii', errors='ignore').decode('utf-8')       # removes non-ascii characters
    bio = re.sub('\s+', ' ', bio)       # repalces repeated whitespace characters with single space
    return bio


''' More tidying
Sometimes the text extracted HTML webpage may contain javascript code and some style elements. 
This function removes script and style tags from HTML so that extracted text does not contain them.
'''
def remove_script(soup):
    for script in soup(["script", "style"]):
        script.decompose()
    return soup


# Checks if bio_url is a valid faculty homepage
def is_valid_homepage(bio_url, dir_url):
    if bio_url.endswith('.pdf'):  # we're not parsing pdfs
        return False
    try:
        # sometimes the homepage url points to the same page as the faculty profile page
        # which should be treated differently from an actual homepage
        ret_url = urllib.request.urlopen(bio_url).geturl()
    except:
        return False       # unable to access bio_url
    # removes url scheme (https,http or www)
    urls = [re.sub('((https?://)|(www.))', '', url) for url in [ret_url, dir_url]]
    return not(urls[0] == urls[1])


def get_base_url(url):
    o = urlparse(url)
    base_url = '{}://{}'.format(o.scheme, o.netloc)
    return base_url


def write_lst(lst, file_):
    with open(file_, 'w') as f:
        for l in lst:
            f.write(l)
            f.write('\n')

