import requests
from datetime import datetime, timedelta
from collections import Counter

end = datetime.now()
start = end - timedelta(days=25)
response = requests.get('https://finnhub.io/api/v1/company-news', params={
    'symbol': 'AAPL',
    'from': start.strftime('%Y-%m-%d'),
    'to': end.strftime('%Y-%m-%d'),
    'token': 'd3a318hr01qli8jcej4gd3a318hr01qli8jcej50'
})
news = response.json()
dates = [datetime.fromtimestamp(n['datetime']).date() for n in news]
date_counts = Counter(dates)

print('Last 30 days article distribution:')
recent_dates = sorted([d for d in date_counts.keys() if d >= (datetime.now().date() - timedelta(days=30))], reverse=True)
for d in recent_dates[:30]:
    print(f'{d}: {date_counts[d]} articles')
