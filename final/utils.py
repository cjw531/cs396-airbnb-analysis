''' Utils for NLP '''
from textblob import TextBlob

def sentiment_calc(text):
    try:
        return TextBlob(text).sentiment.polarity
    except:
        return None

''' Utils for dataframe '''
def price_df_split(df):
    price_100 = df[df['price'] < 100]
    price_200 = df[df['price'] < 200]
    price_200 = price_200[price_200['price'] >= 100]
    price_300 = df[df['price'] < 300]
    price_300 = price_300[price_300['price'] >= 200]
    price_400 = df[df['price'] < 400]
    price_400 = price_400[price_400['price'] >= 300]
    price_500 = df[df['price'] < 500]
    price_500 = price_500[price_500['price'] >= 400]
    price_1000 = df[df['price'] < 1000]
    price_1000 = price_1000[price_1000['price'] >= 500]
    price_2000 = df[df['price'] < 2000]
    price_2000 = price_2000[price_2000['price'] >= 1000]

    return price_100, price_200, price_300, price_400, price_500, price_1000, price_2000