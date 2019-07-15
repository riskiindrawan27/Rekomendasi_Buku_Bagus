import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataBuku=pd.read_csv('books.csv')
dataRating=pd.read_csv('ratings.csv')
# print(dataBuku.head())
# print(dataRating.head())

def mergeCol(i):
    return str(i['authors'])+' '+str(i['original_title'])+' '+str(i['title'])+' '+str(i['language_code'])
dataBuku['Features']=dataBuku.apply(mergeCol,axis='columns')
# print(dataBuku.head())
dataBuku

# count vectorizer
from sklearn.feature_extraction.text import CountVectorizer
model=CountVectorizer(tokenizer=lambda x:x.split(' '))
matrixFeature=model.fit_transform(dataBuku['Features'])

features=model.get_feature_names()
jmlFeatures=len(features)
# print(features)
# print(jmlFeatures)
# print(matrixFeature[0])
# print(matrixFeature.toarray()[0])

# cosine similarity
from sklearn.metrics.pairwise import cosine_similarity
score=cosine_similarity(matrixFeature)
# print(score)
# print(list(enumerate(score[0])))

# Input dengan rating 3
Andi1=dataBuku[dataBuku['original_title']=='The Hunger Games']['book_id'].tolist()[0]-1 
Andi2=dataBuku[dataBuku['original_title']=='Catching Fire']['book_id'].tolist()[0]-1 
Andi3=dataBuku[dataBuku['original_title']=='Mockingjay']['book_id'].tolist()[0]-1 
Andi4=dataBuku[dataBuku['original_title']=='The Hobbit or There and Back Again']['book_id'].tolist()[0]-1 
suka1=[Andi1,Andi2,Andi3,Andi4]

Budi1=dataBuku[dataBuku['original_title']=='Harry Potter and the Philosopher\'s Stone']['book_id'].tolist()[0]-1 
Budi2=dataBuku[dataBuku['original_title']=='Harry Potter and the Chamber of Secrets']['book_id'].tolist()[0]-1 
Budi3=dataBuku[dataBuku['original_title']=='Harry Potter and the Prisoner of Azkaban']['book_id'].tolist()[0]-1 
suka2=[Budi1,Budi2,Budi3]

Ciko1=dataBuku[dataBuku['original_title']=='Robots and Empire']['book_id'].tolist()[0]-1 
suka3=[Ciko1]

Dedi1=dataBuku[dataBuku['original_title']=='Nine Parts of Desire: The Hidden World of Islamic Women']['book_id'].tolist()[0]-1 
Dedi2=dataBuku[dataBuku['original_title']=='A History of God: The 4,000-Year Quest of Judaism, Christianity, and Islam']['book_id'].tolist()[0]-1 
Dedi3=dataBuku[dataBuku['original_title']=='No god but God: The Origins, Evolution, and Future of Islam']['book_id'].tolist()[0]-1 
suka4=[Dedi1,Dedi2,Dedi3]

Ello1=dataBuku[dataBuku['original_title']=='Doctor Sleep']['book_id'].tolist()[0]-1 
Ello2=dataBuku[dataBuku['original_title']=='The Story of Doctor Dolittle']['book_id'].tolist()[0]-1 
Ello3=dataBuku[dataBuku['title']=='Bridget Jones\'s Diary (Bridget Jones, #1)']['book_id'].tolist()[0]-1 
suka5=[Ello1,Ello2,Ello3]

daftarScoreA1=list(enumerate(score[Andi1]))
daftarScoreA2=list(enumerate(score[Andi2]))
daftarScoreA3=list(enumerate(score[Andi3]))
daftarScoreA4=list(enumerate(score[Andi4]))

daftarScoreB1=list(enumerate(score[Budi1]))
daftarScoreB2=list(enumerate(score[Budi2]))
daftarScoreB3=list(enumerate(score[Budi3]))

daftarScoreCiko=list(enumerate(score[Ciko1]))

daftarScoreD1=list(enumerate(score[Dedi1]))
daftarScoreD2=list(enumerate(score[Dedi2]))
daftarScoreD3=list(enumerate(score[Dedi3]))

daftarScoreE1=list(enumerate(score[Ello1]))
daftarScoreE2=list(enumerate(score[Ello2]))
daftarScoreE3=list(enumerate(score[Ello3]))

# print(daftarScore1[0][1])
# print(daftarScore2[17][1])
daftarScoreAndi=[]
for i in daftarScoreA1:
    daftarScoreAndi.append((i[0],0.25*(daftarScoreA1[i[0]][1]+daftarScoreA2[i[0]][1]+daftarScoreA3[i[0]][1]+daftarScoreA4[i[0]][1])))
daftarScoreBudi=[]
for i in daftarScoreA1:
    daftarScoreBudi.append((i[0],(daftarScoreB1[i[0]][1]+daftarScoreB2[i[0]][1]+daftarScoreB3[i[0]][1])/3))
daftarScoreDedi=[]
for i in daftarScoreA1:
    daftarScoreDedi.append((i[0],(daftarScoreD1[i[0]][1]+daftarScoreD2[i[0]][1]+daftarScoreD3[i[0]][1])/3))
daftarScoreEllo=[]
for i in daftarScoreA1:
    daftarScoreEllo.append((i[0],(daftarScoreE1[i[0]][1]+daftarScoreE2[i[0]][1]+daftarScoreE3[i[0]][1])/3))
# print(daftarScoreAndi[0][1])

sortDaftarScoreAndi=sorted(
    daftarScoreAndi,
    key=lambda j:j[1],
    reverse=True
)
sortDaftarScoreBudi=sorted(
    daftarScoreBudi,
    key=lambda j:j[1],
    reverse=True
)
sortDaftarScoreCiko=sorted(
    daftarScoreCiko,
    key=lambda j:j[1],
    reverse=True
)
sortDaftarScoreDedi=sorted(
    daftarScoreDedi,
    key=lambda j:j[1],
    reverse=True
)
sortDaftarScoreEllo=sorted(
    daftarScoreEllo,
    key=lambda j:j[1],
    reverse=True
)
# print(sortDaftarScore)
# print(sortDaftarScore[:5])
# print(sortDaftarScore[:5][0])

# recommending top 5 highest cosine similarity score games
similarGamesAndi=[]
for i in sortDaftarScoreAndi:
    if i[1]>0:
        similarGamesAndi.append(i)
similarGamesBudi=[]
for i in sortDaftarScoreBudi:
    if i[1]>0:
        similarGamesBudi.append(i)
similarGamesCiko=[]
for i in sortDaftarScoreCiko:
    if i[1]>0:
        similarGamesCiko.append(i)
similarGamesDedi=[]
for i in sortDaftarScoreDedi:
    if i[1]>0:
        similarGamesDedi.append(i)
similarGamesEllo=[]
for i in sortDaftarScoreEllo:
    if i[1]>0:
        similarGamesEllo.append(i)

print('1. Buku bagus untuk Andi:')
for i in range(0,5):
    if similarGamesAndi[i][0] not in suka1:
        print('-',dataBuku['original_title'].iloc[similarGamesAndi[i][0]])
    else:
        i+=5
        print('-',dataBuku['original_title'].iloc[similarGamesAndi[i][0]])

print(' ')
print('2. Buku bagus untuk Budi:')
for i in range(0,5):
    if similarGamesBudi[i][0] not in suka2:
        print('-',dataBuku['original_title'].iloc[similarGamesBudi[i][0]])
    else:
        i+=5
        print('-',dataBuku['original_title'].iloc[similarGamesBudi[i][0]])

print(' ')
print('3. Buku bagus untuk Ciko:')
for i in range(0,5):
    if similarGamesCiko[i][0] not in suka3:
        print('-',dataBuku['original_title'].iloc[similarGamesCiko[i][0]])
    else:
        i+=5
        print('-',dataBuku['original_title'].iloc[similarGamesCiko[i][0]])

print(' ')
print('4. Buku bagus untuk Dedi:')
for i in range(0,5):
    if similarGamesDedi[i][0] not in suka4:
        print('-',dataBuku['original_title'].iloc[similarGamesDedi[i][0]])
    else:
        i+=5
        print('-',dataBuku['original_title'].iloc[similarGamesDedi[i][0]])

print(' ')
print('5. Buku bagus untuk Ello:')
for i in range(0,5):
    if similarGamesEllo[i][0] not in suka5:
        if str(dataBuku['original_title'].iloc[similarGamesEllo[i][0]])=='nan':
            print('-',dataBuku['title'].iloc[similarGamesEllo[i][0]])
        else:
            print('-',dataBuku['original_title'].iloc[similarGamesEllo[i][0]])  
    else:
        i+=5
        if str(dataBuku['original_title'].iloc[similarGamesEllo[i][0]])=='nan':
            print('-',dataBuku['title'].iloc[similarGamesEllo[i][0]])
        else:
            print('-',dataBuku['original_title'].iloc[similarGamesEllo[i][0]])  