"""Visualization utilities for NLP text analytics templates."""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
import re


def create_text_analytics_dashboard(text_df, sentiment_df):
    """Create interactive text analytics dashboard with sentiment analysis."""
    
    # Add text metrics
    text_df['length'] = text_df['text'].str.len()
    text_df['word_count'] = text_df['text'].str.split().str.len()
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Sentiment Distribution', 'Text Length', 'Word Count', 'Top Keywords'),
        specs=[[{"type": "bar"}, {"type": "histogram"}],
               [{"type": "histogram"}, {"type": "bar"}]]
    )
    
    # 1. Sentiment distribution
    if 'sentiment' in sentiment_df.columns:
        sentiment_counts = sentiment_df['sentiment'].value_counts()
        colors = ['green' if s == 'POSITIVE' else 'red' if s == 'NEGATIVE' else 'orange' 
                 for s in sentiment_counts.index]
        fig.add_trace(
            go.Bar(x=sentiment_counts.index, y=sentiment_counts.values,
                  marker_color=colors, name="Sentiment"),
            row=1, col=1
        )
    
    # 2. Text length distribution
    fig.add_trace(
        go.Histogram(x=text_df['length'], nbinsx=30, marker_color='skyblue',
                    name="Text Length"),
        row=1, col=2
    )
    
    # 3. Word count distribution
    fig.add_trace(
        go.Histogram(x=text_df['word_count'], nbinsx=25, marker_color='lightgreen',
                    name="Word Count"),
        row=2, col=1
    )
    
    # 4. Top keywords
    all_text = ' '.join(text_df['text'].str.lower())
    words = re.findall(r'\b[a-zA-Z]{4,}\b', all_text)
    stop_words = {'this', 'that', 'with', 'have', 'from', 'they', 'been', 'said', 'which', 'their'}
    filtered_words = [w for w in words if w not in stop_words]
    
    if filtered_words:
        word_counts = Counter(filtered_words).most_common(10)
        words_list, counts_list = zip(*word_counts)
        fig.add_trace(
            go.Bar(x=list(words_list), y=list(counts_list),
                  marker_color='lightcoral', name="Keywords"),
            row=2, col=2
        )
    
    fig.update_layout(title_text="Text Analytics Dashboard", height=700, showlegend=False)
    return fig


def create_word_clouds(sentiment_df):
    """Create word clouds for positive and negative sentiment."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    if 'sentiment' in sentiment_df.columns:
        # Positive sentiment word cloud
        positive_texts = sentiment_df[sentiment_df['sentiment'].str.contains('POSITIVE', na=False)]['text'].tolist()
        if positive_texts:
            wordcloud_pos = WordCloud(width=700, height=400, background_color='white',
                                     colormap='Greens').generate(' '.join(positive_texts))
            axes[0].imshow(wordcloud_pos, interpolation='bilinear')
            axes[0].set_title('Positive Sentiment Word Cloud', fontweight='bold', fontsize=14)
            axes[0].axis('off')
        
        # Negative sentiment word cloud
        negative_texts = sentiment_df[sentiment_df['sentiment'].str.contains('NEGATIVE', na=False)]['text'].tolist()
        if negative_texts:
            wordcloud_neg = WordCloud(width=700, height=400, background_color='white',
                                     colormap='Reds').generate(' '.join(negative_texts))
            axes[1].imshow(wordcloud_neg, interpolation='bilinear')
            axes[1].set_title('Negative Sentiment Word Cloud', fontweight='bold', fontsize=14)
            axes[1].axis('off')
    
    plt.tight_layout()
    return fig


def create_simple_sentiment_chart(sentiment_df):
    """Create simple sentiment distribution chart."""
    if 'sentiment' in sentiment_df.columns:
        sentiment_counts = sentiment_df['sentiment'].value_counts().reset_index()
        fig = px.bar(sentiment_counts, x='sentiment', y='count',
                    title='Sentiment Distribution',
                    color='sentiment',
                    color_discrete_map={'POSITIVE': 'green', 'NEGATIVE': 'red', 'NEUTRAL': 'orange'})
        return fig
    return None


def create_text_complexity_analysis(text_df):
    """Create text complexity and readability analysis visualization."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Text Complexity', 'Average Sentence Length')
    )
    
    # Complexity score distribution
    if 'complexity_score' in text_df.columns:
        fig.add_trace(
            go.Histogram(x=text_df['complexity_score'], nbinsx=20,
                        marker_color='purple', name="Complexity"),
            row=1, col=1
        )
    
    # Sentence length analysis
    if 'avg_sentence_length' in text_df.columns:
        fig.add_trace(
            go.Box(y=text_df['avg_sentence_length'], marker_color='orange',
                  name="Sentence Length"),
            row=1, col=2
        )
    
    fig.update_layout(title_text="Text Complexity Analysis", height=400, showlegend=False)
    return fig


def create_entity_frequency_chart(entities_list):
    """Create entity frequency visualization from extracted entities."""
    if not entities_list:
        return None
    
    # Flatten and count entities
    from collections import Counter
    all_entities = [ent for sublist in entities_list for ent in sublist]
    entity_counts = Counter(all_entities).most_common(15)
    
    if entity_counts:
        entities, counts = zip(*entity_counts)
        fig = px.bar(x=list(counts), y=list(entities), orientation='h',
                    title='Most Frequent Named Entities',
                    labels={'x': 'Frequency', 'y': 'Entity'},
                    color=list(counts), color_continuous_scale='Blues')
        fig.update_layout(height=500, showlegend=False)
        return fig
    return None

