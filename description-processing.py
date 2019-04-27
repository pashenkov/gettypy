import pandas as pd
from pandas import ExcelFile
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag


def degender(text):
    replacements = [
        (' woman', ' person'),
        (' man', ' person'),
        (' guy', ' person'),
        (' girl', ' person'),
        (' boy' , ' person'),
        (' female', ' person'),
        (' male', ' person'),
        (' lady', ' person'),
        (' women', ' people'),
        (' men', ' people'),
        (' guys', ' people'),
        (' girls', ' people'),
        (' boys', ' people'),
        (' females', ' people'),
        (' males', ' people'),
        (' ladies', ' people'),
        (' he', ' he/she'),
        (' she', ' he/she'),
        (' his', ' his/her'),
        (' her', ' his/her')
    ]

    for s in replacements:
        text = text.replace(*s)
    return text

df = pd.read_excel('TECH 2709 Getty Dataset.xlsx', sheet_name='Generated Descriptions')

# delete empty rows
#df = df.drop([0, 1, 2, 3, 4], axis=0)
#df = df.drop(df.index[0:4], axis=0)

print("df.head():")
print(df.head())

print("df.columns:")
print(df.columns)

df_responses = df['Response']  # [0:50]

print("# responses:")
print(df_responses.count())

responses = []
for i in range(df_responses.count()):
    # print(df_titles[i])
    response = df_responses[df_responses.index[i]]
    if(not pd.isnull(response)):  # ignore empty rows
        response = response.replace('\n', '')  # remove newline characters
        response = response.strip()  # remove whitespace
        print("[", i, "] ", response)
        responses.append(response)

# print("Responses:")
# print(responses)

print("Tokenize sentences..")
#response = responses[0]
# response_sentences = sent_tokenize(response)
#print("# sentences: ", len(response_sentences))

MIN_LENGTH_SENTENCE = 10

chosen_response = []

for response in responses:
    # sepreate a paragraph by sentences
    response_sentences = sent_tokenize(response)
    print("\n### processing response..")
    print("# sentences: ", len(response_sentences))

    # loop through each sentence in the tokenized sentence list
    for sentence in response_sentences:
        sentence_tag = pos_tag(word_tokenize(sentence))
        # Check element if exists in list of list
        hasNoun = 'NN' in (tag for word_tag in sentence_tag for tag in word_tag)
        hasVerbG = 'VBG' in (tag for word_tag in sentence_tag for tag in word_tag)
        hasVerbD = 'VBD' in (tag for word_tag in sentence_tag for tag in word_tag)
        if not (hasNoun and (hasVerbD or hasVerbG)):
            print("- removing Sentences doesn't have noun or verb:")
            print(sentence)
            response_sentences.remove(sentence)
        elif (len(sentence) < MIN_LENGTH_SENTENCE):
            print("- removing too short sentence:")
            print(sentence)
            response_sentences.remove(sentence)
        elif (sentence[len(sentence) - 1] != '.'):
            # sentence does not end with '.'
            print("- removing incomplete sentence:")
            print(sentence)
            response_sentences.remove(sentence)
        elif (not sentence[0].isupper()):  # if not starting as capital Character which means the word is broken
            print("- removing the sentence which the first word maybe broken")
            print(sentence)
            response_sentences.remove(sentence)

    for i in range(len(response_sentences)):
        sentence = response_sentences[i]
        if (not sentence[0].isalnum()):
            print("- removing characters not beginning with letters:")
            print(sentence)
            while (not sentence[0].isalnum()):
                sentence = sentence[1:]
            response_sentences[i] = sentence

    print("\n# sentences after processing: ", len(response_sentences))

    for i in range(len(response_sentences)):
        sentence = response_sentences[i]
        # before add to a new list degender the sentences
        sentence = degender(sentence)
        print("[", i, "] ", sentence)
        chosen_response.append(sentence)

print("\nFinal Result:")
print(chosen_response)

print(len(chosen_response))
with open('clean_description.txt', 'w') as f:
    for res in chosen_response:
        res= res.encode('ascii', 'ignore').decode('unicode_escape')
        f.writelines(res+'\n')