"""
ExtractArticleAsync.py - Extract article content from URLs and save in clean JSON format
Usage:
python ExtractArticleAsync.py --input_file "articles.json" --output_file "extracted_articles.json"
"""
import json
import time
import asyncio
import aiohttp
import requests
from bs4 import BeautifulSoup
import logging
from urllib.parse import urlparse
import os
import traceback
import fire
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("article_extraction.log"),
        logging.StreamHandler()
    ]
)

# Configuration
MAX_CONCURRENT_REQUESTS = 10  # Maximum number of concurrent connections
BATCH_SIZE = 10  # Number of articles to process before saving
DELAY_BETWEEN_REQUESTS = 0.5  # Delay between requests in seconds


async def extract_content(soup, domain):
    """Extract the main article content using common patterns"""
    # Handle different sites with specific selectors
    if domain == 'www.forbes.com':
        if soup.select('div.article-body'):
            paragraphs = soup.select('div.article-body p')
            if paragraphs:
                return '\n\n'.join([p.text.strip() for p in paragraphs])

    # Yahoo Finance specific
    elif 'yahoo.com' in domain:
        for selector in ['div.caas-body', '.article-body', '.canvas-body', '.YfinRichtext']:
            container = soup.select_one(selector)
            if container:
                paragraphs = container.find_all('p')
                if paragraphs:
                    return '\n\n'.join([p.text.strip() for p in paragraphs])

    # Try common article container selectors
    for selector in ['article', '.article-content', '.post-content', '.entry-content', '.content-body', '.story-body']:
        article_div = soup.select_one(selector)
        if article_div:
            paragraphs = article_div.find_all('p')
            if paragraphs:
                return '\n\n'.join([p.text.strip() for p in paragraphs])

    # Fallback to all paragraphs, excluding navigation, header, footer, etc.
    main_content = soup.find('main') or soup.find('body')
    if main_content:
        # Exclude paragraphs from navigation, footers, sidebars, etc.
        ignored_parents = ['nav', 'footer', 'header', 'aside']
        paragraphs = [
            p for p in main_content.find_all('p')
            if not any(p.find_parent(tag) for tag in ignored_parents)
        ]
        if paragraphs:
            return '\n\n'.join([p.text.strip() for p in paragraphs])

    # Last resort: collect all paragraphs
    all_paragraphs = soup.find_all('p')
    if all_paragraphs:
        return '\n\n'.join([p.text.strip() for p in all_paragraphs])

    return None


def create_aiohttp_client_session():
    """Create a properly configured aiohttp ClientSession"""
    # Create TCP connector with increased limits
    connector = aiohttp.TCPConnector(
        limit=MAX_CONCURRENT_REQUESTS,
        limit_per_host=5,  # Limit connections per host to avoid overloading servers
        force_close=False,
        enable_cleanup_closed=True
    )

    # Configure client session with increased timeout and buffer sizes
    return aiohttp.ClientSession(
        connector=connector,
        timeout=aiohttp.ClientTimeout(total=30, connect=10, sock_connect=10, sock_read=10),
        raise_for_status=False,  # Don't automatically raise exceptions for HTTP errors
        headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    )


def clean_article(article):
    """Clean and standardize article data structure"""
    # If there's no title or no text and no error, this is an invalid article
    if not article.get("title") or (not article.get("text") and not article.get("error")):
        logging.warning(f"Skipping article with missing title or text: {article.get('title', 'Unknown')}")
        return None

    # Create a clean article with consistent fields
    clean_article = {
        "title": article.get("title", ""),
        "url": article.get("url", ""),
        "timestamp": article.get("timestamp", "")
    }

    # Add text and word count if available
    if article.get("text"):
        clean_article["text"] = article.get("text")
        clean_article["word_count"] = len(article.get("text", "").split())

    # Include error if present
    if "error" in article:
        clean_article["error"] = article["error"]

    return clean_article


async def extract_article(session, url, title=None, timestamp=None, semaphore=None):
    """Extract article content from URL asynchronously"""
    # Skip video URLs
    if '/video/' in url:
        print("v", end="", flush=True)  # v for video skipped
        return {
            "title": title,
            "url": url,
            "timestamp": timestamp,
            "error": "Skipped video URL"
        }

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Referer': 'https://www.google.com/',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Cache-Control': 'max-age=0'
    }

    domain = urlparse(url).netloc
    is_yahoo = 'yahoo.com' in domain

    # Use semaphore to limit concurrent requests
    async with semaphore:
        try:
            print(".", end="", flush=True)  # Show activity

            # Special handling for Yahoo Finance (which often has header size issues)
            if is_yahoo:
                try:
                    # Use synchronous requests for Yahoo
                    response = requests.get(url, headers=headers, timeout=20)
                    if response.status_code != 200:
                        print("✗", end="", flush=True)
                        return {
                            "title": title,
                            "url": url,
                            "timestamp": timestamp,
                            "error": f"HTTP error: {response.status_code}"
                        }

                    html_content = response.text
                except Exception as req_err:
                    print("✗", end="", flush=True)
                    return {
                        "title": title,
                        "url": url,
                        "timestamp": timestamp,
                        "error": f"Request error: {str(req_err)}"
                    }
            else:
                # Standard async request for non-Yahoo sites
                try:
                    async with session.get(url, headers=headers, timeout=30,
                                           read_bufsize=2 ** 20) as response:  # 1MB buffer
                        if response.status != 200:
                            print("✗", end="", flush=True)
                            return {
                                "title": title,
                                "url": url,
                                "timestamp": timestamp,
                                "error": f"HTTP error: {response.status}"
                            }

                        html_content = await response.text()
                except Exception as aio_err:
                    print("✗", end="", flush=True)
                    return {
                        "title": title,
                        "url": url,
                        "timestamp": timestamp,
                        "error": f"Request error: {str(aio_err)}"
                    }

            # Parse HTML content
            soup = BeautifulSoup(html_content, 'html.parser')

            # Extract article text
            content = await extract_content(soup, domain)

            if not content:
                print("✗", end="", flush=True)
                return {
                    "title": title,
                    "url": url,
                    "timestamp": timestamp,
                    "error": "Could not extract content"
                }

            # Create result with all needed fields
            result = {
                "title": title,
                "url": url,
                "timestamp": timestamp,
                "text": content,
                "word_count": len(content.split())
            }

            print("✓", end="", flush=True)
            return result

        except Exception as e:
            print("✗", end="", flush=True)
            logging.error(f"Error extracting {url}: {str(e)}")
            return {
                "title": title,
                "url": url,
                "timestamp": timestamp,
                "error": f"Extraction error: {str(e)}"
            }


def load_existing_articles(output_file):
    """Load existing articles from the output file if it exists"""
    try:
        if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
            with open(output_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []
    except Exception as e:
        logging.warning(f"Could not load existing articles from {output_file}: {str(e)}")
        return []


async def save_results(results, output_file):
    """Save clean results to a JSON file"""
    # Clean all articles before saving
    clean_results = []
    for article in results:
        cleaned = clean_article(article)
        if cleaned:
            clean_results.append(cleaned)

    # Create directory if it doesn't exist
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(clean_results, f, ensure_ascii=False, indent=2)

    logging.info(f"Results saved to {output_file} ({len(clean_results)} articles)")
    return len(clean_results)


async def process_batch(articles_batch, semaphore, all_results):
    """Process a batch of articles asynchronously"""
    async with create_aiohttp_client_session() as session:
        tasks = []
        for article_entry in articles_batch:
            url = article_entry['url']
            title = article_entry.get('title', '')
            timestamp = article_entry.get('timestamp', '')

            # Create task for each article
            task = asyncio.create_task(
                extract_article(session, url, title, timestamp, semaphore)
            )
            tasks.append((task, url, title))

        # Wait for all tasks to complete
        batch_results = []
        for task, url, title in tasks:
            try:
                result = await task
                batch_results.append(result)

                # Log success or failure
                if "error" not in result:
                    logging.info(f"Successfully extracted: {title}")
                else:
                    logging.warning(f"Failed to extract {url}: {result['error']}")

            except Exception as e:
                logging.error(f"Task error for {url}: {str(e)}")
                batch_results.append({
                    "title": title,
                    "url": url,
                    "error": str(e)
                })

        # Add batch results to all results
        all_results.extend(batch_results)
        return len([r for r in batch_results if "error" not in r])  # Return success count


async def async_main(input_file, output_file, skip_existing=True):
    """Main async function to run the extraction process"""
    try:
        # Load input articles
        with open(input_file, 'r', encoding='utf-8') as f:
            articles = json.load(f)

        if not isinstance(articles, list):
            articles = [articles]  # Convert to list if it's a single object

        # Load existing articles if the output file exists
        existing_articles = load_existing_articles(output_file)

        # Create a set of URLs that have already been processed
        existing_urls = set()
        if skip_existing:
            for article in existing_articles:
                if article.get("url"):
                    existing_urls.add(article["url"])

            logging.info(f"Found {len(existing_urls)} already processed articles")

        # Filter out articles that have already been processed
        articles_to_process = []
        for article in articles:
            if not skip_existing or article["url"] not in existing_urls:
                articles_to_process.append(article)
            else:
                logging.info(f"Skipping already processed article: {article.get('title', article['url'])}")

        total_articles = len(articles_to_process)
        logging.info(f"Starting extraction of {total_articles} new articles in batches of {BATCH_SIZE}")

        if total_articles == 0:
            logging.info("No new articles to process. Exiting.")
            return

        # Create a semaphore to limit concurrent connections
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

        # Process articles in batches
        new_results = []
        total_batches = (total_articles + BATCH_SIZE - 1) // BATCH_SIZE
        success_count = 0
        start_time = time.time()

        print(f"\nProcessing {total_articles} articles in {total_batches} batches:")

        for i in range(0, total_articles, BATCH_SIZE):
            batch_num = i // BATCH_SIZE + 1
            batch = articles_to_process[i:i + BATCH_SIZE]

            print(f"\nBatch {batch_num}/{total_batches} [", end="", flush=True)

            # Process batch
            batch_successes = await process_batch(batch, semaphore, new_results)
            success_count += batch_successes

            # Combine new results with existing articles
            combined_results = existing_articles + new_results

            # Save after each batch (with cleaning applied)
            saved_count = await save_results(combined_results, output_file)

            # Calculate and display progress stats
            elapsed = time.time() - start_time
            articles_per_second = (i + len(batch)) / elapsed if elapsed > 0 else 0
            estimated_remaining = (total_articles - (i + len(batch))) / articles_per_second if articles_per_second > 0 else 0

            print(f"] {saved_count} total, {success_count}/{total_articles} new articles successful")
            print(f"Speed: {articles_per_second:.2f} articles/sec, Est. remaining: {estimated_remaining / 60:.1f} min")

            # Small delay between batches
            await asyncio.sleep(1)

        # Final stats
        elapsed_time = time.time() - start_time

        print("\nExtraction complete!")
        print(f"✓ {success_count} new articles successfully extracted")
        print(f"✗ {total_articles - success_count} new articles failed")
        print(f"Total time: {elapsed_time:.1f} seconds ({elapsed_time / 60:.1f} minutes)")
        print(f"Results saved to {output_file} ({len(existing_articles) + len(new_results)} total articles)")

        logging.info(f"Extraction complete! Processed {total_articles} new articles in {elapsed_time:.1f} seconds")
        logging.info(f"Success rate: {success_count}/{total_articles} ({success_count / total_articles * 100:.1f}% of new articles)")

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        logging.debug(traceback.format_exc())
        print(f"\nError: {str(e)}")


def main(input_file="Z:/articles.json", output_file="extracted_articles.json", skip_existing=True):
    """
    Extract articles from URLs, clean data, and save as a JSON file
    :param input_file: Path to the input file containing article URLs
    :param output_file: Path to save the processed JSON data
    :param skip_existing: Skip articles that have already been processed
    """
    # Validate input file exists
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file {input_file} does not exist.")

    # For Windows compatibility
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    # Run the async main function
    asyncio.run(async_main(input_file, output_file, skip_existing))


if __name__ == "__main__":
    fire.Fire(main)