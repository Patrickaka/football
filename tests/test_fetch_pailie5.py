#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测试排列五数据抓取"""

import urllib.request
import re

def main():
    url = 'https://kaijiang.500.com/plw.shtml'
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    req = urllib.request.Request(url, headers=headers)
    html = urllib.request.urlopen(req, timeout=30).read().decode('gb2312', 'ignore')
    
    # 保存HTML以便分析
    with open('pailie5_test.html', 'w', encoding='utf-8') as f:
        f.write(html)
    
    # 搜索期号
    print('=== 搜索期号 ===')
    issues = re.findall(r'(\d{7})期', html)
    if issues:
        print(f'找到 {len(issues)} 个期号')
        print('前10个:', issues[:10])
    
    # 搜索日期
    print('\n=== 搜索日期 ===')
    dates = re.findall(r'(\d{4}-\d{2}-\d{2})', html)
    if dates:
        print(f'找到 {len(dates)} 个日期')
        print('前10个:', dates[:10])
    
    # 搜索数字球
    print('\n=== 搜索数字球 ===')
    balls = re.findall(r'<li[^>]*class="ball[^"]*">[^<]*(\d)[^<]*</li>', html)
    if balls:
        print(f'找到 {len(balls)} 个数字球')
        print('前30个:', balls[:30])

if __name__ == '__main__':
    main()
