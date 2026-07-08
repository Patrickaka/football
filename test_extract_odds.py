with open('okooo_page.html', 'r', encoding='utf-8') as f:
    html = f.read()

import re

table_pattern = re.findall(r'<table[^>]*>(.*?)</table>', html, re.DOTALL)
table = table_pattern[1]

rows = re.findall(r'<tr[^>]*>(.*?)</tr>', table, re.DOTALL)

for i, row in enumerate(rows):
    cells = re.findall(r'<td[^>]*>(.*?)</td>', row, re.DOTALL)
    if cells and len(cells) >= 3:
        score = re.sub(r'<[^>]+>', '', cells[5]).strip() if len(cells) >= 6 else ''
        if score != '-' or i == 6 or i == 7:
            continue
        
        team_cell = cells[2]
        
        odds_pattern = re.findall(r'<em[^>]*>([\d.]+)</em>', team_cell)
        if odds_pattern:
            print(f"\n=== 行 {i} ===")
            print(f"  赔率: {odds_pattern}")
        
        home_match = re.search(r'<span class="homenameobj[^>]*title="([^"]+)"[^>]*>([^<]+)</span>', team_cell)
        away_match = re.search(r'<span class="awaynameobj[^>]*title="([^"]+)"[^>]*>([^<]+)</span>', team_cell)
        handicap_match = re.search(r'<span class="handicapobj[^>]*>([^<]+)</span>', team_cell)
        
        if home_match and away_match:
            print(f"  主队: {home_match.group(2)}")
            print(f"  客队: {away_match.group(2)}")
            print(f"  让球: {handicap_match.group(1) if handicap_match else '无'}")
        
        match_id_match = re.search(r'/soccer/match/(\d+)', row)
        if match_id_match:
            print(f"  match_id: {match_id_match.group(1)}")
