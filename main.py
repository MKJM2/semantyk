import spacy
from gensim.models import KeyedVectors

WORD_EMBEDDINGS: str = 'models/word2vec_100_3_polish.bin'
LEMMATIZER_WEIGHTS: str = 'pl_core_news_lg'
TOP_N: int = 1000

def guess(word_vectors, lemmatizer, target: str) -> tuple[str, int]:
    # Cosine similarity between two vectors
    g: str  = ''
    result = None
    while result is None:
        g = input("Guess: ")
        try:
            lem = lemmatizer(g)
            if len(lem) > 1:
                print(f"Please enter only a single word")
                continue
            g = lem[0].lemma_
            result = word_vectors.similarity(target, g)
        except:
            print(f"Word '{g}' not found in our dictionary")
    return g, result

def main():
    print("Loading word embeddings...", end='')
    word_vectors = KeyedVectors.load(WORD_EMBEDDINGS)
    print("Done!")

    print("Loading lemmatizer...", end='')
    lemmatizer = spacy.load(LEMMATIZER_WEIGHTS)
    print("Done!")

    TARGET = input("Target word? ")

    try:
        closest = word_vectors.similar_by_word(TARGET, topn=TOP_N)
    except:
        print("Word not in dictionary!")
        return

    while True:
        g, sim = guess(word_vectors, lemmatizer, TARGET)
        print(f"{g} [{sim:.3f}]", end = '')
        if g == TARGET:
            print(" Congratz!")
            break
        for i, (word, s) in enumerate(closest):
            if g == word:
                print(f" [{TOP_N - (i + 1)} / 1000]", end = '')
        print("")

    print("The closest 100 words were:")
    for i, word in enumerate(closest[:100]):
        print(f"{i}. {word}")

if __name__ == '__main__':
    main()
