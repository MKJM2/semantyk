import spacy
from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from gensim.models import KeyedVectors

# Constants
WORD_EMBEDDINGS: str = 'models/word2vec_100_3_polish.bin'
LEMMATIZER_WEIGHTS: str = 'pl_core_news_lg'
TOP_N: int = 1000

# Globals
app = FastAPI()
lemmatizer = spacy.load(LEMMATIZER_WEIGHTS)
word_vectors = KeyedVectors.load(WORD_EMBEDDINGS)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Target word
TARGET = "jabłko"
try:
    closest = word_vectors.similar_by_word(TARGET, topn=TOP_N)
except:
    print("Word not in dictionary!")

@app.get("/guess/{guess}")
async def guess_word(guess: str):  # Cosine similarity between guess and target words
    similarity: float = 0
    closeness: int = -1

    word = guess.strip().lower()
    try:
        lem = lemmatizer(word)
        if len(lem) > 1:
            print(f"Please enter only a single word")
            return {"error": "More than one word entered"}
        word = lem[0].lemma_.strip().lower()
        if word == TARGET:
            return {"word": word, "similarity": str(1.0), "close": str(0)}
        similarity = word_vectors.similarity(TARGET, word)
        for i, (w, s) in enumerate(closest):
            if w == word:
                closeness = TOP_N - (i + 1)
                break
    except:
        print(f"Word '{word}' not found in our dictionary")
        return {"error": "Not found in dictionary"}

    return {"word": word, "similarity": str(similarity), "close": str(closeness)}
