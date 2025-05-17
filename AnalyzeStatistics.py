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

        # Check if data is dictionary with results key or directly a list
        if isinstance(data, dict) and 'results' in data:
            results = data['results']
            metadata = data.get('metadata', {'analysis_timestamp': datetime.now().isoformat()})
        elif isinstance(data, list):
            results = data
            metadata = {'analysis_timestamp': datetime.now().isoformat()}
        else:
            print("Warning: Unexpected data structure. Expected 'results' key or list.")
            results = []
            metadata = {'analysis_timestamp': datetime.now().isoformat()}

        print(f"Found {len(results)} articles in the dataset")

        # Handle case where data structure is not as expected
        if not results:
            print("No articles found to analyze.")
            return pd.DataFrame()

        # Extract scores if they exist
        scores_data = []
        article_data = []

        for idx, article in enumerate(results):
            article_info = {
                'article_id': article.get('article_id', idx + 1),
                'title': article.get('title', 'Untitled'),
                'url': article.get('url', ''),
                'timestamp': article.get('timestamp', ''),
                'relevance_explain': article.get('relevance_explain', ''),
                'impact_explain': article.get('impact_explain', ''),
                'confidence': article.get('confidence', 0),
                'word_count': article.get('stats', {}).get('word_count',
                                                           len(article.get('text',
                                                                           '').split()) if 'text' in article else 0),
                'processing_time': article.get('stats', {}).get('processing_time_seconds', 0)
            }

            # Get scores if they exist
            if 'scores' in article and isinstance(article['scores'], dict):
                score_info = {
                    'market_relevance': article['scores'].get('market_relevance'),
                    'market_impact': article['scores'].get('market_impact')
                }
                score_info.update(article_info)  # Combine with article info
                scores_data.append(score_info)
            else:
                article_data.append(article_info)

        # Create DataFrame from scores if available, otherwise from article data
        if scores_data:
            df = pd.DataFrame(scores_data)
        else:
            df = pd.DataFrame(article_data)
            print("No scores found in the data. Basic analysis only.")

        # Rename columns for clarity
        if 'market_relevance' in df.columns and 'market_impact' in df.columns:
            df = df.rename(columns={
                'market_relevance': 'relevance_score',
                'market_impact': 'impact_score',
            })

        # Basic statistical summary
        print("\n=== BASIC STATISTICS ===")
        if 'relevance_score' in df.columns and 'impact_score' in df.columns:
            stats_cols = ['relevance_score', 'impact_score']
            if 'word_count' in df.columns:
                stats_cols.append('word_count')
            stats = df[stats_cols].describe()
            print(stats)

            # Calculate score difference (relevance - impact)
            df['score_gap'] = df['relevance_score'] - df['impact_score']

            # Sort by different metrics
            print("\n=== TOP 5 ARTICLES BY MARKET RELEVANCE ===")
            top_relevance = df.sort_values('relevance_score', ascending=False).head(5)[
                ['article_id', 'title', 'relevance_score']]
            print(top_relevance)

            print("\n=== TOP 5 ARTICLES BY MARKET IMPACT ===")
            top_impact = df.sort_values('impact_score', ascending=False).head(5)[
                ['article_id', 'title', 'impact_score']]
            print(top_impact)

            print("\n=== TOP 5 ARTICLES WITH LARGEST GAP BETWEEN RELEVANCE AND IMPACT ===")
            largest_gap = df.sort_values('score_gap', ascending=False).head(5)[
                ['article_id', 'title', 'relevance_score', 'impact_score', 'score_gap']]
            print(largest_gap)
        else:
            if 'word_count' in df.columns:
                word_stats = df['word_count'].describe()
                print("Word count statistics:")
                print(word_stats)

        # Topic analysis - simple keyword extraction from titles
        print("\n=== KEYWORD FREQUENCY IN TITLES ===")
        all_titles = ' '.join(df['title'].str.lower())
        common_words = ['the', 'and', 'to', 'in', 'of', 'a', 'its', 'for', 'with', 'on', 'as', 'at']
        word_freq = {}

        for word in all_titles.split():
            word = word.strip(',.:;()[]{}\'\"').lower()
            if len(word) > 3 and word not in common_words:
                word_freq[word] = word_freq.get(word, 0) + 1

        sorted_freq = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        print(sorted_freq[:15])

        # Skip remaining analysis if scores not available
        if 'relevance_score' not in df.columns or 'impact_score' not in df.columns:
            print("\nNo score data available for further analysis.")
            return df

        # Correlation analysis
        print("\n=== CORRELATION ANALYSIS ===")
        corr_columns = ['relevance_score', 'impact_score']
        if 'word_count' in df.columns:
            corr_columns.append('word_count')
        if 'processing_time' in df.columns:
            corr_columns.append('processing_time')

        correlation = df[corr_columns].corr()
        print(correlation)

        # Word count relationship to scores if word count available
        if 'word_count' in df.columns:
            bins = [0, 200, 400, 600, 800, 1000, float('inf')]
            labels = ['0-200', '201-400', '401-600', '601-800', '801-1000', '1000+']
            df['word_count_bin'] = pd.cut(df['word_count'], bins=bins, labels=labels)

            word_count_analysis = df.groupby('word_count_bin')[['relevance_score', 'impact_score']].mean()
            print("\n=== SCORES BY ARTICLE LENGTH ===")
            print(word_count_analysis)

        # Create visualizations
        create_visualizations(df, metadata.get('analysis_timestamp', 'Unknown'))

        return df

    except FileNotFoundError:
        print(f"Error: File '{json_file}' not found.")
        return pd.DataFrame()
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in '{json_file}': {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error during analysis: {e}")
        return pd.DataFrame()


def create_visualizations(df, timestamp):
    """
    Create visualizations from the financial news data
    """
    try:
        # Check if required columns exist
        if 'relevance_score' not in df.columns or 'impact_score' not in df.columns:
            print("Required score columns missing. Skipping visualizations.")
            return

        # Set style
        sns.set(style="whitegrid")

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

        # Truncate long titles for better visualization
        top_articles['short_title'] = top_articles['title'].str[:40] + '...'

        sns.barplot(data=top_articles, x='combined_score', y='short_title', hue='relevance_score',
                    palette='viridis', ax=ax3)
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
        if 'word_count_bin' in df.columns:
            plt.figure(figsize=(10, 6))
            word_count_data = df.groupby('word_count_bin')[['relevance_score', 'impact_score']].mean()
            if not word_count_data.empty:
                word_count_data.plot(kind='bar')
                plt.title('Average Scores by Article Length')
                plt.ylabel('Average Score')
                plt.xlabel('Word Count Range')
                plt.savefig('scores_by_length.png', dpi=300, bbox_inches='tight')
                print("Word count analysis saved as 'scores_by_length.png'")

        # Correlation heatmap
        corr_columns = ['relevance_score', 'impact_score']
        if 'word_count' in df.columns:
            corr_columns.append('word_count')
        if 'processing_time' in df.columns:
            corr_columns.append('processing_time')

        plt.figure(figsize=(8, 6))
        sns.heatmap(df[corr_columns].corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Correlation Between Metrics')
        plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
        print("Correlation heatmap saved as 'correlation_heatmap.png'")

    except Exception as e:
        print(f"Error creating visualizations: {e}")


# To run the analysis:
if __name__ == "__main__":
    try:
        # Get the file path from user if not provided
        json_file = 'extracted_articles.json'
        if not os.path.exists(json_file):
            print(f"Warning: '{json_file}' not found in current directory.")
            json_file = input("Please enter the full path to your JSON file: ")

        df = analyze_financial_news(json_file)
        print("\nAnalysis completed successfully!")
    except Exception as e:
        print(f"Error: {e}")