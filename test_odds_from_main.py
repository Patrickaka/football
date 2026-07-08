with open('okooo_page.html', 'r', encoding='utf-8') as f:
    html = f.read()

import re

table_pattern = re.findall(r'<table[^>]*>(.*?)</table>', html, re.DOTALL)
table = table_pattern[1]

rows = re.findall(r'<tr[^>]*>(.*?)</tr>', table, re.DOTALL)

for i, row in enumerate(rows):
    cells = re.findall(r'<td[^>]*>(.*?)</td>', row, re.DOTALL)
    if cells and len(cells) >= 10:
        print(f"\n=== 行 {i} ===")
        
        match_id_match = re.search(r'/soccer/match/(\d+)', row)
        match_id = match_id_match.group(1) if match_id_match else 'N/A'
        
        score = re.sub(r'<[^>]+>', '', cells[5]).strip()
        
        for j in range(6, min(12, len(cells))):
            cell_text = re.sub(r'<[^>]+>', '', cells[j]).strip()
            if cell_text and cell_text != '&nbsp;':
                print(f"  第{j+1}列: {cell_text}")
        
        print(f"  match_id: {match_id}")
        print(f"  score: '{score}'")
