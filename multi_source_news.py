"""
Multi-Source News Aggregator
Fetches financial news from Finnhub, NewsAPI.org, and Polygon.io
Provides 30-day historical coverage with deduplication
"""

import requests
from datetime import datetime, timedelta
from difflib import SequenceMatcher
import os
from dotenv import load_dotenv

load_dotenv()


class MultiSourceNewsAggregator:
    """Aggregate news from multiple APIs for comprehensive coverage"""

    def __init__(self):
        self.finnhub_key = os.getenv("FINNHUB_API_KEY")
        self.newsapi_key = os.getenv("NEWSAPI_KEY")
        self.polygon_key = os.getenv("POLYGON_API_KEY")

    def fetch_all_sources(self, company_symbol, days_back=14):
        """
        Fetch news from all available sources

        Args:
            company_symbol: Stock ticker (e.g., 'AAPL')
            days_back: Number of days to look back

        Returns:
            List of deduplicated news articles
        """
        all_articles = []

        # Fetch from each source
        all_articles.extend(self._fetch_finnhub(company_symbol, days_back))
        all_articles.extend(self._fetch_newsapi(company_symbol, days_back))
        all_articles.extend(self._fetch_polygon(company_symbol, days_back))

        # Deduplicate by title similarity
        unique_articles = self._deduplicate_articles(all_articles)

        # Sort by date (newest first)
        unique_articles.sort(key=lambda x: x.get('datetime', 0), reverse=True)

        return unique_articles

    def _fetch_finnhub(self, symbol, days_back):
        """Fetch from Finnhub API"""
        if not self.finnhub_key:
            return []

        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)

            url = "https://finnhub.io/api/v1/company-news"
            params = {
                'symbol': symbol,
                'from': start_date.strftime('%Y-%m-%d'),
                'to': end_date.strftime('%Y-%m-%d'),
                'token': self.finnhub_key
            }

            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                articles = response.json()
                # Standardize format
                for article in articles:
                    article['source'] = 'Finnhub'
                    article['title'] = article.get('headline', '')
                    article['description'] = article.get('summary', '')
                return articles
        except Exception as e:
            print(f"Finnhub error: {e}")

        return []

    def _fetch_newsapi(self, symbol, days_back):
        """Fetch from NewsAPI.org"""
        if not self.newsapi_key:
            return []

        try:
            # Map stock symbols to company names for better search
            company_names = {
                'AAPL': 'Apple',
                'MSFT': 'Microsoft',
                'GOOGL': 'Google',
                'AMZN': 'Amazon',
                'TSLA': 'Tesla',
                'META': 'Meta Facebook',
                'NVDA': 'NVIDIA',
                'JPM': 'JPMorgan',
                'JNJ': 'Johnson',
                'V': 'Visa'
            }

            search_term = company_names.get(symbol, symbol)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)

            url = "https://newsapi.org/v2/everything"
            params = {
                'q': search_term,
                'from': start_date.strftime('%Y-%m-%d'),
                'to': end_date.strftime('%Y-%m-%d'),
                'language': 'en',
                'sortBy': 'publishedAt',
                'apiKey': self.newsapi_key,
                'pageSize': 100
            }

            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                articles = data.get('articles', [])

                # Standardize format
                standardized = []
                for article in articles:
                    standardized.append({
                        'source': 'NewsAPI',
                        'headline': article.get('title', ''),
                        'title': article.get('title', ''),
                        'summary': article.get('description', ''),
                        'description': article.get('description', ''),
                        'url': article.get('url', ''),
                        'datetime': int(datetime.fromisoformat(
                            article.get('publishedAt', '').replace('Z', '+00:00')
                        ).timestamp()) if article.get('publishedAt') else 0
                    })
                return standardized
        except Exception as e:
            print(f"NewsAPI error: {e}")

        return []

    def _fetch_polygon(self, symbol, days_back):
        """Fetch from Polygon.io"""
        if not self.polygon_key:
            return []

        try:
            url = f"https://api.polygon.io/v2/reference/news"
            params = {
                'ticker': symbol,
                'limit': 100,
                'apiKey': self.polygon_key
            }

            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                articles = data.get('results', [])

                # Filter by date and standardize format
                cutoff_date = datetime.now() - timedelta(days=days_back)
                standardized = []

                for article in articles:
                    pub_date = datetime.fromisoformat(
                        article.get('published_utc', '').replace('Z', '+00:00')
                    )

                    if pub_date >= cutoff_date:
                        standardized.append({
                            'source': 'Polygon',
                            'headline': article.get('title', ''),
                            'title': article.get('title', ''),
                            'summary': article.get('description', ''),
                            'description': article.get('description', ''),
                            'url': article.get('article_url', ''),
                            'datetime': int(pub_date.timestamp())
                        })

                return standardized
        except Exception as e:
            print(f"Polygon error: {e}")

        return []

    def _deduplicate_articles(self, articles):
        """Remove duplicate articles based on title similarity"""
        if not articles:
            return []

        unique = []

        for article in articles:
            title = article.get('headline') or article.get('title', '')
            if not title:
                continue

            is_duplicate = False
            for existing in unique:
                existing_title = existing.get('headline') or existing.get('title', '')

                # Calculate similarity ratio
                similarity = SequenceMatcher(None, title.lower(), existing_title.lower()).ratio()

                if similarity > 0.85:  # 85% similar = duplicate
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique.append(article)

        return unique

    def get_source_statistics(self, articles):
        """Get statistics about article sources"""
        source_counts = {}
        for article in articles:
            source = article.get('source', 'Unknown')
            source_counts[source] = source_counts.get(source, 0) + 1

        return source_counts


# Convenience function for easy import
def fetch_multi_source_news(symbol, days_back=14):
    """
    Fetch news from all sources

    Usage:
        articles = fetch_multi_source_news('AAPL', days_back=30)
    """
    aggregator = MultiSourceNewsAggregator()
    return aggregator.fetch_all_sources(symbol, days_back)
