import streamlit as st
import yfinance as yf
import pandas as pd

# scraping s&p500 data from wikipedia
url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
sp500_table = pd.read_html(url)[0]

# Get the stock symbols
sp500_symbols = list(sp500_table['Symbol'])
all_stocks = sp500_symbols
all_stocks.append("^GSPC")

st.title('Stock Market Prediction')
selected_stock = st.selectbox("Select a stock", all_stocks)

# load the stock data
@st.cache
def load_stock(ticker):
    stock = yf.Ticker(ticker)
    st.subheader(stock.info["shortName"])

load_stock(selected_stock)
