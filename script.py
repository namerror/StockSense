import streamlit as st
import yfinance as yf
import pandas as pd
from plotly import graph_objs as go
# from datetime import date

# scraping s&p500 data from wikipedia
url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
sp500_table = pd.read_html(url)[0]

# get the stock symbols
sp500_symbols = list(sp500_table['Symbol'])
all_stocks = sp500_symbols
all_stocks.append("^GSPC")

# set time range
# START = "2016-01-01"
# TODAY = date.today().strftime("%Y-%m-%d")

# components
st.title('Stock Analysis')
selected_stock = st.selectbox("Select a stock", all_stocks)


# load the stock data
@st.cache_data
def load_stock_data(ticker):
    stock = yf.Ticker(ticker)
    st.subheader(stock.info["shortName"])
    data = stock.history(period="max")

    return data

df = load_stock_data(selected_stock)
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close Price"))
fig.layout.update(title_text=f"{selected_stock} Stock price history",
                  xaxis_rangeslider_visible=True,
                  xaxis_title="Date",
                  yaxis_title="Price"
                  )

st.plotly_chart(fig)
