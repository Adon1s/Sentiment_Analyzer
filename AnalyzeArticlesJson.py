"""
AnalyzeArticles.py - Financial market impact analysis with Llama 3 running on LM Studio
Usage:
python AnalyzeArticles.py --input_file "news_data.json" --output_file "analysis_results.json"
"""
import json
import logging
import os
import fire
import requests
import time
import concurrent.futures
import re
from tqdm.auto import tqdm
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
import statistics

# Default values
DEFAULT_LLAMA_URL = "http://127.0.0.1:1234/v1/chat/completions"
DEFAULT_SYSTEM_PROMPT = """You are a financial news analyst evaluating articles for their market impact. Your ONLY job is to score articles based on their financial significance.

IMPORTANT: Your response MUST follow this EXACT format:
Market Relevance: [number]/10
Market Impact: [number]/10
Explanation: [Brief explanation covering both scores]

Scoring guide:
- Market Relevance: How directly related to financial markets, stocks, commodities, or economic indicators.
  * 10/10: Direct market-moving events (Fed rate decisions, earnings reports, economic data)
  * 7-9/10: Clearly financial but less immediate (company strategy, industry trends)
  * 4-6/10: Indirectly financial (political events with market effects)
  * 1-3/10: Minimal financial connection
  * 0/10: No financial relevance

- Market Impact: How likely the news could move prices or influence investor decisions.
  * 10/10: Major market shifts (economic crisis, major policy changes)
  * 7-9/10: Significant impact on specific sectors or stocks
  * 4-6/10: Moderate impact on companies or minor broader market impact
  * 1-3/10: Minimal but detectable market impact
  * 0/10: No impact on markets

Keep your explanation brief but informative."""

logging.basicConfig(
    format="%(asctime)s | %(levelname)s: %(message)s",
    datefmt="%b/%d %H:%M:%S",
    level=logging.INFO,
)


def get_timestamp():
    """Returns the current timestamp in the format YYYYMMDD_HHMM"""
    return datetime.now().strftime("%Y%b%d_%H-%M")


def parse_scores_and_explanations(generated_text: str) -> Dict:
    """
    Parse the scores and explanations from the generated text using regex.
    Returns a dictionary with the scores and explanations.
    """
    result = {
        "market_relevance": None,
        "market_impact": None,
        "relevance_explain": None,
        "impact_explain": None,
        "full_explanation": None,
        "confidence": 0.0
    }

    # Match patterns for scores
    relevance_match = re.search(r"Market Relevance:\s*(\d+(?:\.\d+)?)/10", generated_text, re.IGNORECASE)
    impact_match = re.search(r"Market Impact:\s*(\d+(?:\.\d+)?)/10", generated_text, re.IGNORECASE)

    # Match pattern for explanation
    explanation_match = re.search(r"Explanation:\s*(.*?)(?:\n\n|\Z)", generated_text, re.IGNORECASE | re.DOTALL)

    # Extract scores
    if relevance_match:
        result["market_relevance"] = float(relevance_match.group(1))
        result["confidence"] += 0.25

    if impact_match:
        result["market_impact"] = float(impact_match.group(1))
        result["confidence"] += 0.25

    # Extract full explanation
    if explanation_match:
        full_explanation = explanation_match.group(1).strip()
        result["full_explanation"] = full_explanation
        result["confidence"] += 0.25

        # Try to split the explanation between relevance and impact
        sentences = re.split(r'(?<=[.!?])\s+', full_explanation)

        # Simple heuristic - first half of sentences for relevance, second half for impact
        if len(sentences) > 1:
            half = len(sentences) // 2
            result["relevance_explain"] = ' '.join(sentences[:half]).strip()
            result["impact_explain"] = ' '.join(sentences[half:]).strip()
        else:
            # If there's only one sentence, use it for both
            result["relevance_explain"] = full_explanation
            result["impact_explain"] = full_explanation

    # Fallback: handle extreme cases where the pattern doesn't match
    if result["confidence"] < 0.5:
        lines = generated_text.strip().split('\n')

        # Look for numeric values in each line
        for line in lines:
            if "market relevance" in line.lower() and not result["market_relevance"]:
                num_match = re.search(r'(\d+(?:\.\d+)?)', line)
                if num_match:
                    result["market_relevance"] = float(num_match.group(1))
                    result["confidence"] += 0.25

            if "market impact" in line.lower() and not result["market_impact"]:
                num_match = re.search(r'(\d+(?:\.\d+)?)', line)
                if num_match:
                    result["market_impact"] = float(num_match.group(1))
                    result["confidence"] += 0.25

        # If we still don't have explanations, try to extract any text that seems explanatory
        if not result["full_explanation"]:
            for i, line in enumerate(lines):
                if any(word in line.lower() for word in ["explanation", "because", "due to", "reason"]):
                    if i + 1 < len(lines) and lines[i + 1].strip():
                        result["full_explanation"] = lines[i + 1].strip()
                        result["relevance_explain"] = result["full_explanation"]
                        result["impact_explain"] = result["full_explanation"]
                        break

    return result


def generate_text_with_llama(
        prompt: str,
        llama_url: str = DEFAULT_LLAMA_URL,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        temperature: float = 0.7,
        max_tokens: int = 100,
        retries: int = 3,
        retry_delay: float = 2.0,
) -> str:
    """
    Generates text using the Llama 3 API running on LM Studio.
    Includes retry logic for robust API calls.
    """
    headers = {
        "Content-Type": "application/json"
    }

    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False
    }

    for attempt in range(retries):
        try:
            response = requests.post(llama_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()

            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]
            else:
                logging.error(f"Unexpected response format: {result}")
                if attempt < retries - 1:
                    logging.info(f"Retrying in {retry_delay} seconds... (Attempt {attempt + 1}/{retries})")
                    time.sleep(retry_delay)
                    continue
                return "Error: Unexpected response format"

        except requests.exceptions.RequestException as e:
            logging.error(f"Request error: {e}")
            if attempt < retries - 1:
                logging.info(f"Retrying in {retry_delay} seconds... (Attempt {attempt + 1}/{retries})")
                time.sleep(retry_delay)
                retry_delay *= 1.5  # Exponential backoff
            else:
                return f"Error: {str(e)}"


def process_article(
        article: Dict,
        article_id: int,
        llama_url: str,
        system_prompt: str,
        prompt_template: str,
        temperature: float,
        max_tokens: int,
        include_article_text: bool,
        verbose: bool,
) -> Dict:
    """Process a single article and return the result dictionary"""
    logger = logging.getLogger(__name__)

    # Format the prompt using the template
    title = article.get("title", "No title")
    text = article.get("text", "")
    url = article.get("url", "")
    timestamp = article.get("timestamp", "")

    # Calculate article stats
    word_count = len(text.split())

    formatted_prompt = prompt_template.format(title=title, text=text)

    # Timing the API call
    start_time = time.time()

    # Generate text with Llama 3
    generated_text = generate_text_with_llama(
        prompt=formatted_prompt,
        llama_url=llama_url,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    processing_time = time.time() - start_time

    # Parse scores and explanations from generated text
    parsed_data = parse_scores_and_explanations(generated_text)

    # Create result object
    result = {
        "article_id": article_id,
        "title": title,
        "url": url,
        "timestamp": timestamp,
        "scores": {
            "market_relevance": parsed_data.get("market_relevance"),
            "market_impact": parsed_data.get("market_impact")
        },
        "relevance_explain": parsed_data.get("relevance_explain"),
        "impact_explain": parsed_data.get("impact_explain"),
        "full_explanation": parsed_data.get("full_explanation"),
        "confidence": parsed_data.get("confidence", 0.0),
        "raw_analysis": generated_text,
        "stats": {
            "word_count": word_count,
            "processing_time_seconds": round(processing_time, 2)
        }
    }

    # Include article text only if specified
    if include_article_text:
        result["text"] = text

    # Output the results if verbose
    if verbose:
        print(f"\nArticle {article_id}: {title}")
        print(f"Generated text:\n{generated_text}\n")
        print(f"Parsed scores: {result['scores']} (confidence: {result['confidence']:.2f})")
        print(f"Relevance explanation: {result['relevance_explain']}")
        print(f"Impact explanation: {result['impact_explain']}")
        print(f"Processing time: {processing_time:.2f} seconds")

    return result


def save_checkpoint(output_file_path: Path, data: Dict, checkpoint_suffix: str = "_partial"):
    """Save a checkpoint of current results"""
    checkpoint_path = output_file_path.with_name(
        f"{output_file_path.stem}{checkpoint_suffix}{output_file_path.suffix}"
    )
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    logging.info(f"Checkpoint saved to {checkpoint_path}")


def calculate_summary_statistics(results: List[Dict]) -> Dict:
    """Calculate summary statistics from results"""
    stats = {
        "market_relevance": {
            "scores": [],
            "avg": None,
            "median": None,
            "min": None,
            "max": None
        },
        "market_impact": {
            "scores": [],
            "avg": None,
            "median": None,
            "min": None,
            "max": None
        },
        "confidence": {
            "scores": [],
            "avg": None
        },
        "processing_time": {
            "total_seconds": 0,
            "avg_seconds": None,
            "max_seconds": 0
        },
        "word_count": {
            "total": 0,
            "avg": None,
            "max": 0
        }
    }

    for result in results:
        # Get scores
        market_relevance = result.get("scores", {}).get("market_relevance")
        market_impact = result.get("scores", {}).get("market_impact")

        if market_relevance is not None:
            stats["market_relevance"]["scores"].append(market_relevance)

        if market_impact is not None:
            stats["market_impact"]["scores"].append(market_impact)

        # Get confidence
        if "confidence" in result:
            stats["confidence"]["scores"].append(result["confidence"])

        # Get processing time
        proc_time = result.get("stats", {}).get("processing_time_seconds", 0)
        stats["processing_time"]["total_seconds"] += proc_time
        stats["processing_time"]["max_seconds"] = max(stats["processing_time"]["max_seconds"], proc_time)

        # Get word count
        word_count = result.get("stats", {}).get("word_count", 0)
        stats["word_count"]["total"] += word_count
        stats["word_count"]["max"] = max(stats["word_count"]["max"], word_count)

    # Calculate averages and other statistics
    for metric in ["market_relevance", "market_impact"]:
        scores = stats[metric]["scores"]
        if scores:
            stats[metric]["avg"] = round(sum(scores) / len(scores), 2)
            stats[metric]["median"] = round(statistics.median(scores), 2)
            stats[metric]["min"] = min(scores)
            stats[metric]["max"] = max(scores)

    if stats["confidence"]["scores"]:
        stats["confidence"]["avg"] = round(sum(stats["confidence"]["scores"]) / len(stats["confidence"]["scores"]), 2)

    if results:
        stats["processing_time"]["avg_seconds"] = round(stats["processing_time"]["total_seconds"] / len(results), 2)
        stats["word_count"]["avg"] = round(stats["word_count"]["total"] / len(results), 2)

    return stats


def main(
        input_file: str,
        output_file: Optional[str] = None,
        llama_url: str = DEFAULT_LLAMA_URL,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        output_dir: Optional[str] = None,
        prompt_template: str = "Title: {title}\n\nContent: {text}",
        include_article_text: bool = False,
        verbose: bool = True,
        parallel: bool = False,
        max_workers: int = 4,
        checkpoint_interval: int = 10,
        dry_run: bool = False,
        output_format: str = "json"
):
    """
    Main function to run the text generation script.
    :param input_file: Path to the JSON file containing articles
    :param output_file: Path for the output file (default: analysis_results_{timestamp}.json)
    :param llama_url: URL for the Llama API endpoint
    :param system_prompt: The system prompt for the model
    :param temperature: The sampling temperature (creativity) for the model. (default: 0.7)
    :param max_tokens: The maximum number of tokens in the generated text. (default: 1024)
    :param output_dir: Directory to save outputs (default: current directory)
    :param prompt_template: Template for formatting each article (default: "Title: {title}\n\nContent: {text}")
    :param include_article_text: Include the original article text in the output (default: False)
    :param verbose: Whether to print the generated text to the console
    :param parallel: Process articles in parallel using multiple workers (default: False)
    :param max_workers: Maximum number of parallel workers when parallel=True (default: 4)
    :param checkpoint_interval: Save a checkpoint after processing this many articles (default: 10)
    :param dry_run: Validate input file without making API calls (default: False)
    :param output_format: Output format: "json" or "csv" (default: "json")
    """
    logger = logging.getLogger(__name__)
    start_time = time.time()

    # Set up output directory and file
    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
    else:
        out_path = Path(".")

    if not output_file:
        timestamp = get_timestamp()
        output_file = f"analysis_results_{timestamp}.{output_format}"

    output_file_path = out_path / output_file

    # Load the input JSON file
    input_path = Path(input_file)

    # Try to find the file if it doesn't exist at the specified path
    if not input_path.exists():
        # Try with current directory
        alt_path = Path(os.getcwd()) / input_path.name
        if alt_path.exists():
            logger.info(f"Found file at alternate path: {alt_path}")
            input_path = alt_path
        else:
            raise FileNotFoundError(f"Input file not found at {input_file} or {alt_path}")

    logger.info(f"Using input file: {input_path.absolute()}")

    logger.info(f"Reading articles from {input_file}...")
    with open(input_path, "r", encoding="utf-8") as f:
        articles = json.load(f)

    if not isinstance(articles, list):
        articles = [articles]  # Convert to list if it's a single object

    # Filter out articles with errors or missing content
    valid_articles = [a for a in articles if "error" not in a and a.get("text")]
    logger.info(f"Found {len(valid_articles)} valid articles out of {len(articles)} to process.")

    if dry_run:
        logger.info("Dry run mode: Input validation complete, exiting without processing articles.")
        return

    # List to store all results
    analysis_results = []

    # Process articles
    if parallel and len(valid_articles) > 1:
        logger.info(f"Processing articles in parallel with {max_workers} workers...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}

            # Submit all jobs
            for i, article in enumerate(valid_articles, start=1):
                future = executor.submit(
                    process_article,
                    article, i, llama_url, system_prompt, prompt_template,
                    temperature, max_tokens, include_article_text, verbose
                )
                futures[future] = i

            # Process results as they complete
            for i, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                article_id = futures[future]
                try:
                    result = future.result()
                    analysis_results.append(result)

                    # Save checkpoint at intervals
                    if i % checkpoint_interval == 0:
                        output_data = {
                            "metadata": {
                                "analysis_timestamp": datetime.now().isoformat(),
                                "total_articles": len(valid_articles),
                                "processed_articles": i,
                                "llama_url": llama_url,
                                "system_prompt": system_prompt,
                                "temperature": temperature,
                                "max_tokens": max_tokens,
                                "input_file": str(input_path),
                                "status": "in_progress"
                            },
                            "results": analysis_results
                        }
                        save_checkpoint(output_file_path, output_data)

                except Exception as e:
                    logger.error(f"Error processing article {article_id}: {e}")
    else:
        # Process sequentially with progress bar
        for i, article in enumerate(tqdm(valid_articles, desc="Processing articles"), start=1):
            try:
                result = process_article(
                    article, i, llama_url, system_prompt, prompt_template,
                    temperature, max_tokens, include_article_text, verbose
                )
                analysis_results.append(result)

                # Save checkpoint at intervals
                if i % checkpoint_interval == 0:
                    output_data = {
                        "metadata": {
                            "analysis_timestamp": datetime.now().isoformat(),
                            "total_articles": len(valid_articles),
                            "processed_articles": i,
                            "llama_url": llama_url,
                            "system_prompt": system_prompt,
                            "temperature": temperature,
                            "max_tokens": max_tokens,
                            "input_file": str(input_path),
                            "status": "in_progress"
                        },
                        "results": analysis_results
                    }
                    save_checkpoint(output_file_path, output_data)

            except Exception as e:
                logger.error(f"Error processing article {i}: {e}")

    # Sort results by article_id to maintain order
    analysis_results.sort(key=lambda x: x.get("article_id", 0))

    # Calculate summary statistics
    summary_stats = calculate_summary_statistics(analysis_results)

    # Calculate total processing time
    total_processing_time = time.time() - start_time

    # Create final output with metadata
    output_data = {
        "metadata": {
            "analysis_timestamp": datetime.now().isoformat(),
            "total_articles": len(valid_articles),
            "processed_articles": len(analysis_results),
            "llama_url": llama_url,
            "system_prompt": system_prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "input_file": str(input_path),
            "total_processing_time_seconds": round(total_processing_time, 2),
            "status": "completed"
        },
        "summary": summary_stats,
        "results": analysis_results
    }

    # Save results to output file
    if output_format.lower() == "json":
        with open(output_file_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=4)
        logger.info(f"Completed processing {len(analysis_results)} articles in {total_processing_time:.2f} seconds.")
        logger.info(f"Results saved to {output_file_path}")

    elif output_format.lower() == "csv":
        import csv

        csv_file_path = output_file_path.with_suffix('.csv')

        with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['article_id', 'title', 'url', 'timestamp', 'market_relevance',
                          'market_impact', 'relevance_explain', 'impact_explain', 'confidence', 'word_count']

            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for result in analysis_results:
                writer.writerow({
                    'article_id': result.get('article_id', ''),
                    'title': result.get('title', ''),
                    'url': result.get('url', ''),
                    'timestamp': result.get('timestamp', ''),
                    'market_relevance': result.get('scores', {}).get('market_relevance', ''),
                    'market_impact': result.get('scores', {}).get('market_impact', ''),
                    'relevance_explain': result.get('relevance_explain', ''),
                    'impact_explain': result.get('impact_explain', ''),
                    'confidence': result.get('confidence', ''),
                    'word_count': result.get('stats', {}).get('word_count', '')
                })

        logger.info(f"Completed processing {len(analysis_results)} articles in {total_processing_time:.2f} seconds.")
        logger.info(f"CSV results saved to {csv_file_path}")

        # Also save the full data as JSON for reference
        with open(output_file_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=4)
        logger.info(f"Full JSON results saved to {output_file_path}")

    else:
        logger.error(f"Unsupported output format: {output_format}. Saving as JSON instead.")
        with open(output_file_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=4)


if __name__ == "__main__":
    fire.Fire(main)
