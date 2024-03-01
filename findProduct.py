import nltk
from nltk.tokenize import word_tokenize
from nltk.metrics import edit_distance

nltk.download('punkt')


def findMatch(main_sentence, sentences):
    main_tokens = word_tokenize(main_sentence.lower())
    matches = []

    for i, sentence in enumerate(sentences):
        sentence_tokens = word_tokenize(sentence.lower())
        distance = edit_distance(main_tokens, sentence_tokens)
        similarity = 1 - \
            (distance / max(len(main_tokens), len(sentence_tokens)))
        matches.append((i, similarity))

    matches.sort(key=lambda x: x[1], reverse=True)
    return matches[0]


main_sentence = "Iphone 15 plus black (128GB storage) "
sentences = [
    "black Iphone 13 128GB 8GB",
    "Black iphone 15 plus (16GB RAM) (128GB storage)",
    "Iphone 15 plus black 256GB 4GB",
    "Iphone 15 plus black (128GB storage) (16GB RAM)",
    "Yet another sentence that is also not a match"
]

index = findMatch(main_sentence, sentences)
print("Matches:")
print("Best Match: ", index[0])