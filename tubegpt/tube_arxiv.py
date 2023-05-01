import re
from typing import List

import arxiv


def find_paper(url: str) -> arxiv.Result:
    pattern = r'arxiv\.org\/(?:abs|pdf)\/(\d+\.\d+)'
    match = re.search(pattern, url)
    if match:
        arxiv_id = match.group(1)
        search = arxiv.Search(id_list=[arxiv_id])
        paper = next(search.results())
        return paper
    else:
        return None
    
def paper_blurb(url: str) -> str:
    paper: arxiv.Result = find_paper(url)
    if paper:
        title : str = paper.title
        authors : List[str] = [author.name for author in paper.authors]
        published : str = paper.published.strftime("%m/%d/%Y")
        url : str = paper.pdf_url
        blurb: str = f"""
{title}
{", ".join(authors)}
Released on {published}
ArXiV: {url}
        """
        return blurb
    else:
        return None

if __name__ == "__main__":
    url = "https://arxiv.org/abs/2106.04430"
    print(paper_blurb(url))