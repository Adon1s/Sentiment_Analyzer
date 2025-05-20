import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np
import os


def analyze_financial_news(json_file):
    """
    Comprehensive analysis of financial news JSON data
    """
    try:
        # Load JSON data with explicit UTF-8 encoding
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Extract results
        results = data.get('results', [])
        print(f"Found {len(results)} articles in the dataset")

        # Create DataFrame directly from results
        df = pd.DataFrame(results)

        # Extract scores from the nested dictionaries
        # This avoids using json_normalize with record_path
        relevance_scores = []
        impact_scores = []

        for article in results:
            scores = article.get('scores', {})
            relevance_scores.append(scores.get('market_relevance'))
            impact_scores.append(scores.get('market_impact'))

        df['relevance_score'] = relevance_scores
        df['impact_score'] = impact_scores

        # Extract word count and processing time
        word_counts = []
        processing_times = []

        for article in results:
            stats = article.get('stats', {})
            word_counts.append(stats.get('word_count'))
            processing_times.append(stats.get('processing_time_seconds'))

        df['word_count'] = word_counts
        df['processing_time'] = processing_times

        # Keep only valid articles with scores
        valid_df = df.dropna(subset=['relevance_score', 'impact_score']).copy()
        print(f"Found {len(valid_df)} articles with valid scores out of {len(df)} total")

        if len(valid_df) == 0:
            print("No valid scored articles found. Cannot proceed with analysis.")
            return df

        # Basic statistical summary
        print("=== BASIC STATISTICS ===")
        stats = valid_df[['relevance_score', 'impact_score', 'word_count']].describe()
        print(stats)

        # Calculate score difference (relevance - impact)
        valid_df['score_gap'] = valid_df['relevance_score'] - valid_df['impact_score']

        # Sort by different metrics
        top_relevance = valid_df.sort_values('relevance_score', ascending=False).head(5)[
            ['article_id', 'title', 'relevance_score']]
        print("\n=== TOP 5 ARTICLES BY MARKET RELEVANCE ===")
        print(top_relevance)

        top_impact = valid_df.sort_values('impact_score', ascending=False).head(5)[
            ['article_id', 'title', 'impact_score']]
        print("\n=== TOP 5 ARTICLES BY MARKET IMPACT ===")
        print(top_impact)

        largest_gap = valid_df.sort_values('score_gap', ascending=False).head(5)[
            ['article_id', 'title', 'relevance_score', 'impact_score', 'score_gap']]
        print("\n=== TOP 5 ARTICLES WITH LARGEST GAP BETWEEN RELEVANCE AND IMPACT ===")
        print(largest_gap)

        # Topic analysis - simple keyword extraction from titles
        print("\n=== KEYWORD FREQUENCY IN TITLES ===")
        all_titles = ' '.join(valid_df['title'].str.lower())
        common_words = ['the', 'and', 'to', 'in', 'of', 'a', 'its', 'for', 'with', 'on', 'as', 'at']
        word_freq = {}

        for word in all_titles.split():
            word = word.strip(',.:;()[]{}\'\"').lower()
            if len(word) > 3 and word not in common_words:
                word_freq[word] = word_freq.get(word, 0) + 1

        sorted_freq = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        print(sorted_freq[:15])

        # Correlation analysis
        print("\n=== CORRELATION ANALYSIS ===")
        correlation = valid_df[['relevance_score', 'impact_score', 'word_count', 'processing_time']].corr()
        print(correlation)

        # Word count relationship to scores
        bins = [0, 200, 400, 600, 800, 1000, float('inf')]
        labels = ['0-200', '201-400', '401-600', '601-800', '801-1000', '1000+']
        valid_df['word_count_bin'] = pd.cut(valid_df['word_count'], bins=bins, labels=labels)

        word_count_analysis = valid_df.groupby('word_count_bin')[['relevance_score', 'impact_score']].mean()
        print("\n=== SCORES BY ARTICLE LENGTH ===")
        print(word_count_analysis)

        # Create visualizations
        create_visualizations(valid_df, data.get('metadata', {}).get('analysis_timestamp', 'Unknown'))

        return valid_df

    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


def create_visualizations(df, timestamp):
    """
    Create visualizations from the financial news data
    """
    # Set style
    sns.set(style="whitegrid")

    try:
        # Create a figure with multiple subplots
        fig = plt.figure(figsize=(15, 12))

        # 1. Score distribution histogram
        ax1 = plt.subplot(2, 2, 1)
        sns.histplot(data=df[['relevance_score', 'impact_score']], kde=True, ax=ax1)
        ax1.set_title('Distribution of Scores')
        ax1.set_xlabel('Score Value')
        ax1.set_ylabel('Frequency')

        # 2. Relevance vs Impact scatter plot
        ax2 = plt.subplot(2, 2, 2)
        if 'word_count' in df.columns:
            sns.scatterplot(data=df, x='relevance_score', y='impact_score', size='word_count',
                            sizes=(20, 200), alpha=0.7, ax=ax2)
        else:
            sns.scatterplot(data=df, x='relevance_score', y='impact_score', alpha=0.7, ax=ax2)
        ax2.set_title('Market Relevance vs. Market Impact')
        ax2.set_xlabel('Market Relevance Score')
        ax2.set_ylabel('Market Impact Score')

        # 3. Top articles by combined score
        ax3 = plt.subplot(2, 1, 2)
        df['combined_score'] = df['relevance_score'] + df['impact_score']
        top_articles = df.sort_values('combined_score', ascending=False).head(10)

        # Truncate titles for better visualization
        top_articles['short_title'] = top_articles['title'].str[:40] + '...'

        sns.barplot(data=top_articles, x='combined_score', y='short_title', ax=ax3)
        ax3.set_title('Top 10 Articles by Combined Score (Relevance + Impact)')
        ax3.set_xlabel('Combined Score')
        ax3.set_ylabel('Article Title')

        if not top_articles.empty:
            ax3.set_xlim(0, top_articles['combined_score'].max() + 1)

        # Add timestamp and adjust layout
        plt.suptitle(f'Financial News Analysis - {timestamp}', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # Save the visualization
        plt.savefig('financial_news_analysis.png', dpi=300, bbox_inches='tight')
        print("\nVisualization saved as 'financial_news_analysis.png'")

        # Word count vs scores
        plt.figure(figsize=(10, 6))
        word_count_plot = df.groupby('word_count_bin')[['relevance_score', 'impact_score']].mean().plot(kind='bar')
        plt.title('Average Scores by Article Length')
        plt.ylabel('Average Score')
        plt.xlabel('Word Count Range')
        plt.savefig('scores_by_length.png', dpi=300, bbox_inches='tight')
        print("Word count analysis saved as 'scores_by_length.png'")

        # Correlation heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(df[['relevance_score', 'impact_score', 'word_count', 'processing_time']].corr(),
                    annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Correlation Between Metrics')
        plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
        print("Correlation heatmap saved as 'correlation_heatmap.png'")

    except Exception as e:
        print(f"Error creating visualizations: {e}")


# To run the analysis:
if __name__ == "__main__":
    df = analyze_financial_news('analysis_results.json')
