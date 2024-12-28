import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px
from textblob import TextBlob
import folium
from folium.plugins import HeatMap
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from io import BytesIO

data = {
    'review': [
        'Great hotel! Friendly staff and clean rooms.',
        'Not bad, but the room was noisy.',
        'Loved the breakfast and the pool was amazing!',
        'The hotel was alright but could be cleaner.',
        'Terrible experience, dirty rooms and rude staff.',
        'Very comfortable, perfect location, but the service was slow.',
        'Amazing view, but the room had a strange smell.'
    ],
    'score': [5, 3, 4, 2, 1, 4, 2],
    'country': ['US', 'UK', 'IN', 'US', 'FR', 'US', 'FR'],
    'room_type': ['Standard', 'Suite', 'Standard', 'Deluxe', 'Standard', 'Deluxe', 'Suite'],
    'guest_type': ['Leisure', 'Business', 'Leisure', 'Business', 'Leisure', 'Leisure', 'Business'],
    'stay_date': ['2024-12-01', '2024-12-02', '2024-12-03', '2024-12-04', '2024-12-05', '2024-12-06', '2024-12-07']
}

df = pd.DataFrame(data)

def analyze_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

df['sentiment'] = df['review'].apply(analyze_sentiment)

def plot_sentiment_distribution(df):
    plt.figure(figsize=(10, 6))
    sns.histplot(df['sentiment'], kde=True, color="skyblue")
    plt.title("Sentiment Distribution of Reviews")
    plt.xlabel('Sentiment')
    plt.ylabel('Frequency')
    st.pyplot(plt)

def multi_dimensional_sentiment_analysis(df):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='score', y='sentiment', hue='room_type', style='guest_type', s=100)
    plt.title("Sentiment vs. Review Score with Room and Guest Type")
    plt.xlabel('Review Score')
    plt.ylabel('Sentiment')
    st.pyplot(plt)

def geo_map(df):
    m = folium.Map(location=[20,0], zoom_start=2)
    for i, row in df.iterrows():
        folium.CircleMarker(location=[20, 0], radius=9, popup=row['review'], color='blue', fill=True).add_to(m)
    st.write(m)

def time_series_analysis(df):
    df['stay_date'] = pd.to_datetime(df['stay_date'])
    df['month'] = df['stay_date'].dt.to_period('M')
    monthly_sentiment = df.groupby('month')['sentiment'].mean().reset_index()

    plt.figure(figsize=(10, 6))
    sns.lineplot(x='month', y='sentiment', data=monthly_sentiment, marker='o', color='purple')
    plt.title("Average Sentiment Over Time (Monthly)")
    plt.xlabel('Month')
    plt.ylabel('Average Sentiment')
    st.pyplot(plt)

def word_cloud(df):
    text = " ".join(review for review in df['review'])
    wordcloud = WordCloud(width=800, height=400).generate(text)

    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title("Word Cloud of Review Phrases")
    st.pyplot(plt)

def comparison_visualization(df):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='room_type', y='sentiment', data=df, palette='Set2')
    plt.title("Comparison of Sentiment by Room Type")
    plt.xlabel('Room Type')
    plt.ylabel('Sentiment')
    st.pyplot(plt)

def sentiment_by_room_and_guest(df):
    room_sentiment = df.groupby('room_type')['sentiment'].mean().reset_index()
    guest_sentiment = df.groupby('guest_type')['sentiment'].mean().reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.barplot(x='room_type', y='sentiment', data=room_sentiment, ax=axes[0])
    axes[0].set_title('Sentiment by Room Type')
    
    sns.barplot(x='guest_type', y='sentiment', data=guest_sentiment, ax=axes[1])
    axes[1].set_title('Sentiment by Guest Type')
    
    st.pyplot(fig)

def export_to_pdf(df):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    
    c.drawString(100, 750, "Hotel Review Analysis Report")
    
    c.drawString(100, 730, f"Total Reviews: {len(df)}")
    c.drawString(100, 710, f"Average Sentiment: {df['sentiment'].mean():.2f}")
    
    c.showPage()
    c.save()
    
    buffer.seek(0)
    st.download_button(
        label="Download PDF Report",
        data=buffer,
        file_name="hotel_review_analysis_report.pdf",
        mime="application/pdf"
    )

def main():
    st.sidebar.title('Navigation')
    option = st.sidebar.radio('Select Analysis', ['Sentiment Distribution', 'Sentiment vs. Score', 'Geo-mapping', 'Time Series', 'Word Cloud', 'Room Type Sentiment', 'Export to PDF'])

    if option == 'Sentiment Distribution':
        st.title("Sentiment Distribution")
        plot_sentiment_distribution(df)
    
    elif option == 'Sentiment vs. Score':
        st.title("Sentiment vs. Review Score with Room and Guest Type")
        multi_dimensional_sentiment_analysis(df)
    
    elif option == 'Geo-mapping':
        st.title("Geo-mapping of Sentiment by Country")
        geo_map(df)
    
    elif option == 'Time Series':
        st.title("Average Sentiment Over Time (Monthly)")
        time_series_analysis(df)
    
    elif option == 'Word Cloud':
        st.title("Word Cloud of Review Phrases")
        word_cloud(df)
    
    elif option == 'Room Type Sentiment':
        st.title("Sentiment Analysis by Room Type and Guest Type")
        sentiment_by_room_and_guest(df)
    
    elif option == 'Export to PDF':
        st.title("Export Analysis as PDF")
        export_to_pdf(df)

if __name__ == "__main__":
    main()
