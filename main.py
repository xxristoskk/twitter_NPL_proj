import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from dateutil.parser import parse
import time

#read it
df = pd.read_json('/home/xristsos/Documents/bot/botplay/twitter data/tweets.json')


import os
from google.oauth2 import service_account
from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types
credentials = service_account.Credentials.from_service_account_file("/home/xristsos/Downloads/DDAssistant-3023eb496920.json")
client = language.LanguageServiceClient(credentials=credentials)

def analyzer(tweet):
    text = tweet
    document = types.Document(
        content=text,
        type=enums.Document.Type.PLAIN_TEXT)

    sentiment = client.analyze_sentiment(document=document).document_sentiment

    return (sentiment.score,sentiment.magnitude)

def looper(data):
    sents = []
    mags = []
    for tweet in data:
        try:
            sent, mag = analyzer(tweet)
        except:
            del tweet
        mags.append(mag)
        sents.append(sent)
    return sents, mags

sen, mag = looper(df['full_text'])

df['sentiment'] = sen
df['magnitude'] = mag

## SENTIMENT FROM APRIL TO AUG (OR SOMETHING LIKE THAT)
df_s = pd.read_pickle('/home/xristsos/Documents/bot/botplay/big_tweet_with_sent.pkl')
df_s.columns
df_s['2017':'2019'].resample('W')
df_s.set_index('created_at',inplace=True)
df_s.head()

corr = df_s.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
df_s['normalized'] = (df_s['sentiment']-df_s['sentiment'].mean())/(df_s['sentiment'].std())
df_s['normalized'].plot.hist()
df_s['sentiment'].plot.hist()
# df_s[df_s['engagements']<200].plot.scatter(x='magnitude',y='user profile clicks')
# sns.heatmap(corr,mask=mask,cmap=cmap,square=True)


snt = df_s[['normalized']]
snt = snt.set_index(snt['time'])
snt = snt['2016':'2019']
monthly_snt = snt.resample('M').mean()
weekly_snt = snt.resample('W').mean()
yearly_snt = snt.resample('Y').mean()
yearly_snt.plot.line(figsize=(17,6))
weekly_snt.plot.line(figsize=(15,6))
monthly_snt.plot.line(figsize=(15,6))
weekly_diff = weekly_snt['normalized'].diff().rename(index={'sentiment':  'difference observed'})
diff = monthly_snt['normalized'].diff().rename(index={'sentiment':  'difference observed'})
weekly_diff.plot(figsize=(17,5))
diff.plot(figsize=(17,5))


# from statsmodels.graphics.tsaplots import plot_acf
# plot_acf(monthly_snt[:-1],alpha=.05)
#
# from pmdarima.arima import auto_arima
# model = auto_arima(monthly_snt, trace=True, error_action='ignore', suppress_warnings=True)
# model_fit = model.fit(monthly_snt)
# future_value = model.predict(n_periods=1)[0]

future_value


from statsmodels.graphics.tsaplots import plot_pacf
plot_pacf(monthly_snt[:-1],alpha=.05)

df.to_pickle('big_tweet_with_sent.pkl')

from statsmodels.tsa.arima_model import ARIMA
#fit ARIMA model (3,1,0))

model = ARIMA(monthly_snt[:-1], order=(1,0,2))
model_fit = model.fit(disp=0)
print(model_fit.summary())

residuals = pd.DataFrame(model_fit.resid)
residuals.plot()
plt.show()

residuals = pd.DataFrame(model_fit.resid)
residuals.plot(kind='kde')
plt.show()

model_fit.plot_predict()

fig, ax = plt.subplots(figsize=(7,6))
sns.lineplot(x=df.time,y=df.sentiment,ax=ax)
sns.distplot(df.sentiment,ax=ax)
fig

df.drop('Tweet permalink', axis=1, inplace=True)
df.sort_values('engagement rate').tail()
sns.pairplot(data=df,x_vars='sentiment',y_vars='engagement rate',height=6)

df.describe()
