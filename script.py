import os
import streamlit as st
import yfinance as yf
import pandas as pd
from plotly import graph_objs as go
from prophet import Prophet
from prophet.plot import plot_plotly
from datetime import datetime, timedelta
from openai import OpenAI
from dotenv import load_dotenv

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

# plotting the graph
df = load_stock_data(selected_stock)
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close Price"))
fig.layout.update(title_text=f"{selected_stock} Stock price history (close price)",
                  xaxis_rangeslider_visible=True,
                  xaxis_title="Date",
                  yaxis_title="Price"
                  )
st.plotly_chart(fig)

# raw data
st.subheader('Raw Data')
st.write(df.tail())

# Load environment variables from .env file
load_dotenv()

# Set up the OpenAI client 
api_key = os.environ["OPENAI_API_KEY"]
client = OpenAI(api_key=api_key)

# Summarize Raw data
completion1 = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": f"Summarize {df.tail()} such that it is easy to understand its overview and trends"
        }
    ]
)
output1 = completion1.choices[0].message.content
output1

# preparation
df_train = df.reset_index()[["Date", "Close"]]
df_train["Date"] = df_train["Date"].dt.tz_localize(None) # Remove timezone
max_years = pd.Timestamp.now().year - df_train["Date"].iloc[0].year

st.write("[Prophet](%s) is a forecasting tool that can be used to predict future trends" % "https://facebook.github.io/prophet/")
forecast_button = st.button('Run forecast with Prophet')
forecast_range = st.slider("Time of prediction(days)", 1, 1460)
training_years = st.slider("Select training data(years)", 1, max_years, 5) # how many years back are the training data

def run_prophet_forecast(df_train, training_years, forecast_range):
    start_date = datetime.now() - timedelta(days=training_years*365)
    df_train = df_train[df_train["Date"] >= start_date] # filter the range
    df_train.columns = ['ds', 'y']
    model = Prophet()
    model.fit(df_train)
    future_df = model.make_future_dataframe(periods=forecast_range)
    forecast = model.predict(future_df)
    fig_forecast = plot_plotly(model, forecast)
    f_title = f"{selected_stock} Stock forecast for the next {forecast_range} day(s) based on recent {training_years} years"
    fig_forecast.layout.update(title_text=f_title,
                               xaxis_rangeslider_visible=True,
                               xaxis_title="Date",
                               yaxis_title="Price")
    st.plotly_chart(fig_forecast)

if forecast_button:
    run_prophet_forecast(df_train, training_years, forecast_range)

#OpenAI prompt

prompt = st.text_input("Got any questions? Ask away! ")
completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": prompt
        }
    ]
)
output = completion.choices[0].message.content
output
