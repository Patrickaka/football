import urllib.request
import re

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Referer': 'https://www.okooo.com/danchang/',
}

url = 'https://www.okooo.com/danchang/'
req = urllib.request.Request(url, headers=headers)
try:
    with urllib.request.urlopen(req, timeout=30) as resp:
        html = resp.read().decode('gbk', errors='replace')
        print(f'HTML长度: {len(html)}')
        
        bookmaker_patterns = re.findall(r'bet-at-home|Bet365|威廉希尔|立博|澳门', html)
        print(f'找到的bookmaker: {bookmaker_patterns[:20]}')
        
        odds_pattern = re.findall(r'(\d+\.\d{2})\s+(\d+\.\d{2})\s+(\d+\.\d{2})', html)
        print(f'找到的赔率组: {len(odds_pattern)}')
        print(f'前10组: {odds_pattern[:10]}')
        
        match_pattern = re.findall(r'<a[^>]*href="/soccer/match/(\d+)/"[^>]*>([^<]+)</a>', html)
        print(f'找到的比赛链接: {len(match_pattern)}')
        for mid, name in match_pattern[:5]:
            print(f'  {mid}: {name}')
        
        with open('okooo_page.html', 'w', encoding='utf-8') as f:
            f.write(html)
        print('页面已保存到 okooo_page.html')
        
except Exception as e:
    print(f'失败: {e}')
