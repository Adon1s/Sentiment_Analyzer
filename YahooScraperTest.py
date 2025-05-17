import requests
from bs4 import BeautifulSoup
import json
import time
from datetime import datetime
import os


def load_existing_articles():
    try:
        if os.path.exists('articles.json'):
            with open('articles.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        return []
    except json.JSONDecodeError:
        print("Error reading articles file, starting fresh")
        return []


def save_articles(articles):
    with open('articles.json', 'w', encoding='utf-8') as f:
        json.dump(articles, f, indent=2, ensure_ascii=False)


def scrape_yahoo_finance(existing_articles):
    url = 'https://finance.yahoo.com/news/'

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
    }

    new_articles_found = False

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')
        articles = soup.find_all('a', class_='subtle-link')

        # Get existing titles for quick lookup
        existing_titles = {article['title'] for article in existing_articles}

        for article in articles:
            h3_tag = article.find('h3')
            if h3_tag:
                title = h3_tag.get_text(strip=True)
            else:
                title = article.get('title', '').strip()

            if title and title not in existing_titles:
                url = article.get('href', '')
                if url and not url.startswith('http'):
                    url = 'https://finance.yahoo.com' + url

                # Create new article entry
                new_article = {
                    'title': title,
                    'url': url,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }

                existing_articles.append(new_article)
                existing_titles.add(title)
                new_articles_found = True

                print(f'New article found: {title}')

        return new_articles_found

    except requests.RequestException as e:
        print(f"Error making request: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

    return False


def monitor_articles():
    print("Starting Yahoo Finance article monitor...")
    print("Press Ctrl+C to stop")

    while True:
        try:
            # Load existing articles
            articles = load_existing_articles()

            # Scrape for new articles
            new_found = scrape_yahoo_finance(articles)

            # Save if new articles were found
            if new_found:
                save_articles(articles)
                print(f"Total articles saved: {len(articles)}")

            # Wait for 60 seconds
            print("\nWaiting 60 seconds before next check...", end='', flush=True)
            for _ in range(60):
                time.sleep(1)
                print('.', end='', flush=True)
            print('\n')

        except KeyboardInterrupt:
            print("\nStopping monitor...")
            break
        except Exception as e:
            print(f"An error occurred in the monitor loop: {e}")
            time.sleep(60)  # Wait before retrying


if __name__ == "__main__":
    monitor_articles()