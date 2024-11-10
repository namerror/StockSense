import streamlit as st
import yfinance as yf
import pandas as pd
from plotly import graph_objs as go
from prophet import Prophet
from prophet.plot import plot_plotly
from datetime import datetime, timedelta
from openai import OpenAI

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
st.title('StockSense')
st.write('An AI-powered tool for financial analysis')
selected_stock = st.selectbox("Select a stock", all_stocks)


# load the stock data
@st.cache_data
def load_stock_data(ticker):
    stock = yf.Ticker(ticker)
    st.subheader(stock.info["shortName"])
    data = stock.history(period="max")
    return data

@st.cache_data
def load_financial_info(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    return info

# plotting the graph of stock history
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
st.write('Example Raw Data')
st.write(df.tail())

# preparation
df_train = df.reset_index()[["Date", "Close"]]
df_train["Date"] = df_train["Date"].dt.tz_localize(None) # Remove timezone
max_years = pd.Timestamp.now().year - df_train["Date"].iloc[0].year

# widgets
st.markdown("##### Predict the stock trend with machine learning")
st.write("[Prophet](%s) is a forecasting model that can be used to predict future trends" % "https://facebook.github.io/prophet/")
forecast_button = st.button('Run forecast with Prophet')
forecast_range = st.slider("Time of prediction(days)", 1, 1460)
training_years = st.slider("Select training data(years)", 1, max_years, 5) # how many years back are the training data

# prophet forecast
def run_prophet_forecast(df_train, training_years, forecast_range):
    # setup
    start_date = datetime.now() - timedelta(days=training_years*365)
    df_train = df_train[df_train["Date"] >= start_date] # filter the range
    df_train.columns = ['ds', 'y']

    # prediction
    model = Prophet()
    model.fit(df_train)
    future_df = model.make_future_dataframe(periods=forecast_range)
    forecast = model.predict(future_df)

    # plot
    fig_forecast = plot_plotly(model, forecast)
    f_title = f"{selected_stock} Stock forecast for the next {forecast_range} day(s) based on recent {training_years} years"
    fig_forecast.layout.update(title_text=f_title,
                               xaxis_rangeslider_visible=True,
                               xaxis_title="Date",
                               yaxis_title="Price")
    st.plotly_chart(fig_forecast)

if forecast_button:
    run_prophet_forecast(df_train, training_years, forecast_range)


# using OpenAI to summarize financial situation of the company

financial_info = load_financial_info(selected_stock)


@st.cache_data
def get_ai_summary_plus(financial_info, api_key):
    client = OpenAI(api_key=api_key)
    data_summary = f"""
    Based on the current financial information (provided below) for {financial_info['longName']}, summarize and comment on the company's financial position:
    trailing PE: {financial_info["trailingPE"]},
    forward PE: {financial_info["forwardPE"]},
    fifty-day average: {financial_info["fiftyDayAverage"]},
    200-day average: {financial_info["twoHundredDayAverage"]}
    profit margins: {financial_info["profitMargins"]},
    52-week change: {financial_info["52WeekChange"]},
    current price: {financial_info["currentPrice"]},
    current ratio: {financial_info["currentRatio"]},
    recommendation key: {financial_info["recommendationKey"]}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": data_summary}]
    )

    summary = response['choices'][0]['message']['content']
    return summary

# @st.cache_data
# def get_ai_summary(financial_info, api_key):
#     client = OpenAI(api_key=api_key)
#     data_summary = f"""
#     Based on the current financial information {financial_info['longName']}, summarize and comment on the company's financial position
#     """

#     response = client.chat.completions.create(
#         model="gpt-3.5-turbo",
#         messages=[{"role": "user", "content": data_summary}]
#     )

#     summary = response['choices'][0]['message']['content']
#     return summary

st.markdown("##### Integrated ChatGPT summary")
openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
"[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
"[View the source code](https://github.com/namerror/StockSense)"
generate_response = st.button("Generate financial summary")
if generate_response:
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
    else:
        try:
            ai_summary = get_ai_summary_plus(financial_info, openai_api_key)
            st.markdown("## Financial summary (AI generated)")
            st.write(ai_summary)
        except:
            st.warning("Invalid API key")

#OpenAI prompt
try:
    client = OpenAI(api_key=openai_api_key)
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
    st.write(output)
except:
    st.info("To ask chatgpt, you need to enter a valid api.")
