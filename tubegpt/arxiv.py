import re

import arxiv


def get_arxiv_info(url):
    pattern = r'arxiv.org\/(?:abs|pdf)\/([\w.-]+)'
    match = re.search(pattern, url)

    if match:
        arxiv_id = match.group(1)
        search = arxiv.Search(id_list=[arxiv_id])
        paper = next(search.results())
        return paper
    else:
        return None

