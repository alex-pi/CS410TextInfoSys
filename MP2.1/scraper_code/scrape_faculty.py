from scraper_code.scrape_util import *


def scrape_dir_page(dir_url, driver):
    print('-' * 20, 'Scraping directory page', '-' * 20)
    faculty_links = []
    faculty_base_url = get_base_url(dir_url)
    # execute js on webpage to load faculty listings on webpage and get ready to parse the loaded HTML
    soup = get_js_soup(dir_url, driver)
    for link_holder in soup.find_all('div', class_='namearea'):  # get list of all <div> of class 'name'
        rel_link = link_holder.find('a')['href']  # get url
        # url returned is relative, so we need to add base url
        faculty_links.append(faculty_base_url + rel_link)
    print('-' * 20, 'Found {} faculty profile urls'.format(len(faculty_links)), '-' * 20)
    return faculty_links


def scrape_faculty_pages(fac_urls, driver):
    bios = []
    urls = []
    for fac_url in fac_urls:
        print('Scraping bio from {}'.format(fac_url))
        try:
            soup = remove_script(get_js_soup(fac_url, driver))
            bio_section = soup.find("h2", string='Biography').find_parent("section")
            bio = ''
            for p in bio_section.find_all("p"):
                bio += process_bio(p.get_text())
            if bio.strip() != '' and fac_url.strip() != '':
                urls.append(fac_url)
                bios.append(bio)
            print('Bio found: {}'.format(bio))
        except:
            print('Could not access {}'.format(fac_url))

    return urls, bios
