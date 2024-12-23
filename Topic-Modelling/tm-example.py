from gensim.models.ldamodel import LdaModel
from gensim.corpora.dictionary import Dictionary
from nltk.tokenize import word_tokenize

# Sample corpus
documents = [
    "Artificial intelligence is transforming the world.",
    "Machine learning and AI are subsets of data science.",
    "Natural language processing enables machines to understand human language.",
]

# Preprocessing
tokenized_docs = [word_tokenize(doc.lower()) for doc in documents]
dictionary = Dictionary(tokenized_docs)
corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]

# Train LDA Model
lda = LdaModel(corpus, num_topics=2, id2word=dictionary, passes=15)

# Display Topics
for idx, topic in lda.print_topics(-1):
    print(f"Topic {idx}: {topic}")
