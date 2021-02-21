from bs4 import BeautifulSoup
import requests
import pandas as pd
import numpy as np
from time import sleep


title=[] #List to store the titles
news=[] #List to store the news
sources=[] #List to store sources
classes=[] #List to store classes
score=[] #List to store scores

#--------------------bbc (real news)-----------------------------
#
url = 'https://www.bbc.com/news/live/explainers-51871385/page/'
l=[]
for i in np.arange(2,51):
    sleep(20)
    response2 = requests.get(url + str(i))
    soup7 = BeautifulSoup(response2.text, 'html.parser')
    h=soup7.find_all('h3')
    for i in np.arange(0,len(h)):
     try:
      # sleep(20)
      titre=h[i].get_text()
      link=h[i].find('a')["href"]
      link='https://www.bbc.com'+link
      response3 = requests.get(link)
      soup8 = BeautifulSoup(response3.text, 'html.parser')
      d=soup8.find_all('div',class_='ssrcss-rgov1k-MainColumn e1sbfw0p0')
      if len(d) == 1:
          l.append(h[i])
          paragraphes=d[0].find_all('p')
          list_paragraphs=[]
          for j in np.arange(0,len(paragraphes)):
              paragraph = paragraphes[j].get_text()
              list_paragraphs.append(paragraph)
          final_article = " ".join(list_paragraphs)
          news.append(final_article)
          sources.append(link)
          title.append(titre)
          score.append(5)
          classes.append(1)
     except:
      continue

#--------------corvelva (fake news)----------------------

link ="https://www.corvelva.it"

page = requests.get("https://www.corvelva.it/en/approfondimenti/notizie/covid19.html")

soup = BeautifulSoup(page.content, 'html.parser')

for div in soup.findAll('div', attrs={'class':'col-md-6'}):
    # sleep(20)
    div2 = div.find('div', attrs={'class':'article-header'})
    a= div2.find('a')["href"]
    a=link+a
    page1 = requests.get(a)
    soup1 = BeautifulSoup(page1.content, 'html.parser')
    for div3 in soup1.findAll('div', attrs={'class':'article-header'}):
        titre=div3.find('h1')
        title.append(titre.text)
        # print(titre.text)
    for div4 in soup1.findAll('div', attrs={'class':'article-details-wrapper gazette-custom-font'}):
        mytext=""
        for h2 in div4.findAll("h2"):
            text1 = h2
            mytext += text1.text
        for p in div4.findAll("p"):
            text1 = p
            mytext += text1.text
        news.append(mytext)
        # print(mytext)

    sources.append(link)
    score.append(1)
    n = 0
    classes.append(n)

page = requests.get("https://www.corvelva.it/en/approfondimenti/notizie/covid19.html?start=21")

soup = BeautifulSoup(page.content, 'html.parser')

for div in soup.findAll('div', attrs={'class':'col-md-6'}):
    # sleep(20)
    div2 = div.find('div', attrs={'class':'article-header'})
    a= div2.find('a')["href"]
    a=link+a
    page1 = requests.get(a)
    soup1 = BeautifulSoup(page1.content, 'html.parser')
    for div3 in soup1.findAll('div', attrs={'class':'article-header'}):
        titre=div3.find('h1')
        title.append(titre.text)
    for div4 in soup1.findAll('div', attrs={'class':'article-details-wrapper gazette-custom-font'}):
        mytext=""
        for h2 in div4.findAll("h2"):
            text1 = h2
            mytext += text1.text
        for p in div4.findAll("p"):
            text1 = p
            mytext += text1.text
        news.append(mytext)

    sources.append(link)
    n = 0
    classes.append(n)
    score.append(1)

page = requests.get("https://www.corvelva.it/en/approfondimenti/notizie/covid19.html?start=42")

soup = BeautifulSoup(page.content, 'html.parser')

for div in soup.findAll('div', attrs={'class':'col-md-6'}):
    # sleep(20)
    div2 = div.find('div', attrs={'class':'article-header'})
    a= div2.find('a')["href"]
    a=link+a
    page1 = requests.get(a)
    soup1 = BeautifulSoup(page1.content, 'html.parser')
    for div3 in soup1.findAll('div', attrs={'class':'article-header'}):
        titre=div3.find('h1')
        title.append(titre.text)
    for div4 in soup1.findAll('div', attrs={'class':'article-details-wrapper gazette-custom-font'}):
        mytext=""
        for h2 in div4.findAll("h2"):
            text1 = h2
            mytext += text1.text
        for p in div4.findAll("p"):
            text1 = p
            mytext += text1.text
        news.append(mytext)

    sources.append(link)
    n = 0
    classes.append(n)
    score.append(1)

page = requests.get("https://www.corvelva.it/en/approfondimenti/notizie/covid19.html?start=63")

soup = BeautifulSoup(page.content, 'html.parser')

for div in soup.findAll('div', attrs={'class':'col-md-6'}):
    # sleep(20)
    div2 = div.find('div', attrs={'class':'article-header'})
    a= div2.find('a')["href"]
    a=link+a
    page1 = requests.get(a)
    soup1 = BeautifulSoup(page1.content, 'html.parser')
    for div3 in soup1.findAll('div', attrs={'class':'article-header'}):
        titre=div3.find('h1')
        title.append(titre.text)
        # print(titre.text)
    for div4 in soup1.findAll('div', attrs={'class':'article-details-wrapper gazette-custom-font'}):
        mytext=""
        for h2 in div4.findAll("h2"):
            text1 = h2
            mytext += text1.text
        for p in div4.findAll("p"):
            text1 = p
            mytext += text1.text
        news.append(mytext)
        # print(mytext)

    sources.append(link)
    n = 0
    classes.append(n)
    score.append(1)

# #----------------healthnutnews (fake news)------------------------

def make_soup(url):
    page = requests.get(url)
    soupdata = BeautifulSoup(page.content, "html.parser")
    return soupdata

linksNumber=[ "0","10","20","30","40","50",
             "60","70","80","90","100", "110", "120"]
for numb in linksNumber:
        # sleep(20)
        soup3 = make_soup("https://www.healthnutnews.com/?s=covid&limit=10&bpaged="+numb)
        for div in soup3.findAll('h2', attrs={'class': 'bsearch-entry-title'}):
            a = div.find('a')["href"]
            page5 = requests.get(a)
            soup5 = BeautifulSoup(page5.content, 'html.parser')
            for div5 in soup5.findAll('div', attrs={'class': 'post-header'}):
                titre = div5.find('h1', attrs={'class': 'entry-title'})
                title.append(titre.text)
            sources.append("https://www.healthnutnews.com")
            classes.append(0)
            score.append(2)

            for div6 in soup5.findAll('div', attrs={'class': 'post-content entry-content'}):
                text5 = ''
                for p in div6.findAll("p"):
                    par = p
                    text5+=par.text
            news.append(text5)

#----------------------save ti csv file---------------------------

df = pd.DataFrame({'Title':title,'News':news,'Score': score, 'Source':sources,'Class':classes})
df.to_csv('News.csv', index=False, encoding='utf-8')