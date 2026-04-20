from newsapi import NewsApiClient

newsapi = NewsApiClient(api_key='ad3528fc093a40038e27cbef40f348c3')

all_articles = newsapi.get_everything(q='bitcoin',
                                      sources='bbc-news,the-verge',
                                      domains='bbc.co.uk,techcrunch.com',
                                      from_param='2017-12-01',
                                      to='2017-12-12',
                                      language='en',
                                      sort_by='relevancy',
                                      page=2)


print(all_articles)