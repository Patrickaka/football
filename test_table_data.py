with open('okooo_page.html', 'r', encoding='utf-8') as f:
    html = f.read()

import re

table_pattern = re.findall(r'<table[^>]*>(.*?)</table>', html, re.DOTALL)
table = table_pattern[1]

rows = re.findall(r'<tr[^>]*>(.*?)</tr>', table, re.DOTALL)
print(f'表格总行数: {len(rows)}')

for i, row in enumerate(rows):
    cells = re.findall(r'<td[^>]*>(.*?)</td>', row, re.DOTALL)
    if cells:
        clean = []
        for c in cells:
            text = re.sub(r'<[^>]+>', '', c).strip()
            if text:
                clean.append(text[:50])
        print(f'\n行 {i}: {len(cells)}列')
        print(f'  内容: {clean}')
        
        hrefs = re.findall(r'href="([^"]+)"', row)
        if hrefs:
            print(f'  链接: {hrefs}')
