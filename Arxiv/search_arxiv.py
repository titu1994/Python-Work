import arxiv

search = arxiv.Search(
    query="Librispeech",
    max_results=10,
    sort_by=arxiv.SortCriterion.SubmittedDate,
    sort_order=arxiv.SortOrder.Descending,
)

for result in search.results():
    print(result)
