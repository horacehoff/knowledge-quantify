import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from collections import Counter
from heapq import nlargest

nlp = spacy.load("en_core_web_md")

def quantify_knowledge_percentage(eval_text, summary_inverse_aggressiveness):
    original_text = eval_text
    print("Original word count: " + str(len(eval_text.split())))
    original_word_count = len(eval_text.split())

    # NLP PERCENTAGE - heavily inspired by Kamal Khumar -> https://medium.com/analytics-vidhya/text-summarization-using-spacy-ca4867c6b744
    doc = nlp(eval_text)

    keyword = []
    stop_words = list(STOP_WORDS)
    pos_tag = ["PROPN", "ADJ", "NOUN", "VERB"]
    for token in doc:
        if token.text in stop_words or token.text in punctuation:
            continue
        if token.pos_ in pos_tag:
            keyword.append(token.text)

    freq_word = Counter(keyword)
    max_freq = Counter(keyword).most_common(1)[0][1]
    for word in freq_word.keys():
        freq_word[word] = freq_word[word] / max_freq

    sent_strength = {}

    for sent in doc.sents:
        for word in sent:
            if word.text in freq_word.keys():
                if sent in sent_strength.keys():
                    sent_strength[sent] += freq_word[word.text]
                else:
                    sent_strength[sent] = freq_word[word.text]

    summarized_sentences = nlargest(summary_inverse_aggressiveness, sent_strength, key=sent_strength.get)
    final_sentences = [w.text for w in summarized_sentences]
    summary = ' '.join(final_sentences)

    # "USEFUL" WORDS PERCENTAGE
    redundant_words = ["where", "that", "although", "even though", "even if", "in case", "in spite of", "despite",
                       "so that", "whatever", "whereas", "whenever", "wherever", "furthermore", "as a result",
                       "consequently", "besides", "in addition", "more over", "moreover", "however", "nevertheless",
                       "nonetheless", "in the same way", "likewise", "similarly", "thus", "therefore", "as a result",
                       "because", "additionally", "also", "again", "further", "then", "too", "correspondingly",
                       "indeed",
                       "regarding", "whereas", "conversely", "in comparison", "by contrast", "another view is",
                       "alternatively", "although", "otherwise", "instead", "indeed", "resulting from", "consequently",
                       "in the same way", "compared with", "by contrast", "although", "compared with", "by contrast",
                       "yet",
                       "notwithstanding", "in spite of", "after all", "at the same time", "even if", "in contrast",
                       "in other terms", "rather", "or", "in view of this", "additionally", "finally", "also",
                       "subsequently", "eventually", "next", "then", "the", "but", "not only"]

    for word in redundant_words:
        if word in eval_text:
            eval_text = eval_text.replace(word, "")
    eval_text = " ".join(eval_text.split())

    print("Final word count (WORDS REMOVAl): " + str(len(eval_text.split())))
    print("Final word count (NLP): " + str(len(summary.split())))
    print("Knowledge Density (WORDS REMOVAl): ~" + str(
        round(len(eval_text.split()) * 100 / original_word_count, 1)) + "%")
    words_density = len(eval_text.split()) * 100 / original_word_count
    print("Knowledge Density (NLP): ~" + str(round(len(summary.split()) * 100 / len(original_text.split()), 1)) + "%")
    nlp_density = len(summary.split()) * 100 / len(original_text.split())
    print("Knowledge Density (AVG): ~" + str(round((words_density + nlp_density*2) / 3, 1)) + "%")
    return round((words_density + nlp_density) / 2, 1)


quantify_knowledge_percentage("""""", 30)