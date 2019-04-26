import pandas as pd
from pandas import ExcelFile
from nltk.tokenize import word_tokenize, sent_tokenize

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

for response in responses:
    response_sentences = sent_tokenize(response)
    print("\n### processing response..")
    print("# sentences: ", len(response_sentences))

    for sentence in response_sentences:
        if(len(sentence) < MIN_LENGTH_SENTENCE):
            print("- removing too short sentence:")
            print(sentence)
            response_sentences.remove(sentence)
        elif(sentence[len(sentence)-1] != '.'):
            # sentence does not end with '.'
            print("- removing incomplete sentence:")
            print(sentence)
            response_sentences.remove(sentence)

    for i in range(len(response_sentences)):
        sentence = response_sentences[i]
        if(not sentence[0].isalnum()):
            print("- removing characters not beginning with letters:")
            print(sentence)
            while(not sentence[0].isalnum()):
                sentence = sentence[1:]
            response_sentences[i] = sentence

    print("\n# sentences after processing: ", len(response_sentences))

    for i in range(len(response_sentences)):
        sentence = response_sentences[i]
        print("[", i, "] ", sentence)