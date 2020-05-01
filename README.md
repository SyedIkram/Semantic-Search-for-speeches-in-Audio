# 3SA: Semantic Search for Speeches in Audio

Semantic search is the ability to search for documents by understanding the overall meaning of the query rather than using simple keyword matches. Recent breakthroughs in NLP like Bert, Albert, Roberta, etcetera, paved the way for the development of such powerful semantic search engines. But most of these search algorithms are mainly focused on textual information, i.e., both the document and the query are in natural language. In this project, we aim to develop a semantic search algorithm for arbitrary objects (objects which are not in natural language), specifically for speeches in audio, by leveraging advanced NLP techniques. We introduce 3SA, Semantic Search for Speeches in Audio, which can enable the search for audio files, semantically. We perform our experiments on the Librispeech dataset and further evaluate our search results using basic information retrieval metrics.

### Run instructions

Make sure you have Python>=3.6
Setup a virtual environment:

    python3 -m venv venv
    source venv/bin/activate
    pip3 install -r requirements.txt
    jupyter notebook
    
Run the project notebook - project.ipynb file.
