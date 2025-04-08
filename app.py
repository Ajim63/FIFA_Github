import streamlit as st
import pandas as pd
import plotly.express as px

import numpy as np
import re
import string

from sklearn.preprocessing import MinMaxScaler
import nltk
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer

# Download NLTK resources
nltk.download('stopwords')
nltk.download('vader_lexicon')

# Load the CSV file
df = pd.read_csv("March_madness_1st_and_2nd_month.csv")

# Preprocessing
stop_words = set(stopwords.words('english'))
gambling_keywords = ['bet', 'betting', 'odds', 'parlay', 'draftkings', 'fanduel', 'wager', 'spread']
schools = ['Duke', 'Kansas', 'Alabama', 'UConn', 'Gonzaga', 'Florida', 'Michigan', 'UCLA']

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|@\w+|#\w+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = " ".join(word for word in text.split() if word not in stop_words)
    return text

df['Clean_Text'] = df['Text'].apply(clean_text)
df['Gambling_Related'] = df['Clean_Text'].apply(lambda x: any(word in x for word in gambling_keywords))
df = df[~df['Gambling_Related']].reset_index(drop=True)

def extract_school(text):
    for school in schools:
        if school.lower() in text.lower():
            return school
    return None

df['School_Mentioned'] = df['Text'].apply(extract_school)

sia = SentimentIntensityAnalyzer()
df['Sentiment_Score'] = df['Clean_Text'].apply(lambda x: sia.polarity_scores(x)['compound'])

def label_sentiment(score):
    if score >= 0.1:
        return "Positive"
    elif score <= -0.1:
        return "Negative"
    else:
        return "Neutral"

df['Sentiment_Label'] = df['Sentiment_Score'].apply(label_sentiment)

df['Engagement_Score'] = (df['Likes'] + 2 * df['Retweets'] + 1.5 * df['Replies'] + 1.2 * df['Quotes']) / (df['Impressions'] + 1)
df['Engagement_Score'] = MinMaxScaler().fit_transform(df[['Engagement_Score']])

df['Created_At'] = pd.to_datetime(df['Created At'])
df['Date'] = df['Created_At'].dt.date

# Game Quality Analysis (Text-based)
game_quality_keywords = ['amazing game',  'wow', 'intense', 'overtime', 'clutch', 'buzzer beater', 'epic', 'crazy finish', 'NailBiter', 'thriller', 'tight game', 'Teamwork', 'perfect', 'fire']

def assess_game_quality(text):
    text = text.lower()
    return any(keyword in text for keyword in game_quality_keywords)

df['game_quality_mention'] = df['Text'].apply(assess_game_quality)

daily_game_quality = df.groupby('Date')['game_quality_mention'].mean().reset_index()
daily_game_quality.columns = ['Date', 'GameQualityScore']
game_quality_fig = px.line(daily_game_quality, x='Date', y='GameQualityScore', title='Game Quality Score Over Time')
game_quality_fig.update_layout(xaxis_title='Date', yaxis_title='Game Quality Score (0 to 1)')

sentiment_summary = df[df['School_Mentioned'].notnull()].groupby(['Date', 'School_Mentioned']).agg(
    Engagement=('Engagement_Score', 'mean'),
    Positivity=('Sentiment_Label', lambda x: np.mean(x == 'Positive'))
).reset_index()

us_states_abbrev = {
    'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI', 'ID', 'IL',
    'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT',
    'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI',
    'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY'
}

def extract_state(location):
    try:
        state_match = re.findall(r'\b[A-Z]{2}\b', str(location))
        for abbr in state_match:
            if abbr in us_states_abbrev:
                return abbr
    except:
        pass
    return np.nan

df['State'] = df['Location'].apply(extract_state)

daily_state_counts = df.groupby(['Date', 'State']).size().reset_index(name='Tweet Count')

location_fig = px.choropleth(
    daily_state_counts,
    locations='State',
    locationmode="USA-states",
    color='Tweet Count',
    scope="usa",
    animation_frame='Date',
    color_continuous_scale="Oranges",
    range_color=(0, daily_state_counts['Tweet Count'].max()),
    title="Daily Tweet Volume by US State",
    height=700,
    width=1000
)

# Streamlit Layout
st.title("March Madness Twitter Analysis Dashboard")
tabs = st.tabs(["Game Quality",  "Daily Sentiment",  'Daily Engagement', "Location Heatmap",])

with tabs[0]:
    st.subheader("Game Quality Analysis")
    st.write("This chart shows the daily measure of game quality based on sentiment analysis of tweets.")
    st.plotly_chart(game_quality_fig)

with tabs[1]:
    st.subheader("📈 Daily Sentiment")
    fig1 = px.line(sentiment_summary, x='Date', y='Positivity', color='School_Mentioned', markers=True)
    st.plotly_chart(fig1)

with tabs[2]:
    st.subheader("🔥 Daily Engagement")
    fig2 = px.line(sentiment_summary, x='Date', y='Engagement', color='School_Mentioned', markers=True)
    st.plotly_chart(fig2)

with tabs[3]:
    st.subheader("Tweet Location Heatmap")
    st.write("This animated map shows the daily tweet volume by U.S. state.")
    st.plotly_chart(location_fig, use_container_width=True)