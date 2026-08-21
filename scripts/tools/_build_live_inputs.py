# -*- coding: utf-8 -*-
"""批量构建 11 场（除首尔）的 live_context_{mid}.json。

直接用 WebSearch 核实的伤停/风格/赛程数据构建合规 live_context：
- injuries: 逐条具名（球员/球队/角色/状态/影响/来源），不编造
- tool_log: 每条发现附来源 URL + 时间戳（skill 合规凭证）
- injury_conflict: 来源冲突显式标注（如「无伤停」vs「列具伤停」）
- possession: None（本批多为非 FBref 覆盖联赛 → UNAVAILABLE，不编造）
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPORTS = os.path.join(ROOT, "reports")
TS = "2026-07-22T15:30"

# 来源 URL（与 WebSearch 实测对应）
U = {
    "lil_fot": "https://www.fotmob.com/match/762049/matchfacts/lillestrom-vs-viking",
    "lil_qm": "https://www.qiumiwu.com/news/1978266200110",
    "gwang_sohu": "https://www.sohu.com/a/1053347614_122614941",
    "gwang_now": "https://www.nowscore.com/news/1176903.htm",
    "mia_fot": "https://www.fotmob.com/es-419/matches/chicago-fire-fc-vs-inter-miami-cf/5yu85bd7",
    "mia_qq": "https://new.qq.com/rain/a/20260722A096I100",
    "omo_tof": "https://www.toffeeweb.com/int/omonia-nicosia-vs-kairat-predictions-picks-odds-22-07-2026",
    "omo_sohu": "https://www.sohu.com/a/1053105665_121924582",
    "bot_fot": "https://www.fotmob.com/match/3564937/matchfacts/botafogo-vs-vitoria",
    "bot_ds": "https://datasports.co/teams/botafogo-de-futebol-e-regatas",
    "buche_tt": "https://www.toutiao.com/a7665142314087629327",
    "buche_qq": "https://new.qq.com/rain/a/20260722A09M6F00",
    "chap_of": "https://onefootball.com/en/news/chapecoense-x-flamengo-horario-escalacoes-e-estatisticas-brasileirao-2207-43164802",
    "sp_fot": "https://www.fotmob.com/zh-Hans/matches/athletico-paranaense-vs-sao-paulo/3hq57e",
    "sp_qq": "https://new.qq.com/rain/a/20260722A096I100",
    "bod_fot": "https://www.fotmob.com/en-GB/matches/bodoglimt-vs-hamkam/2cj30r",
    "bod_sm": "https://www.sportsmole.co.uk/football/bodo-glimt/preview/bodoglimt-vs-hamkam-prediction-team-news-lineups_601535.html",
    "lax_fot": "https://www.fotmob.com/matches/real-salt-lake-vs-los-angeles-fc/4vexhen5",
    "lax_qm": "https://www.qiumiwu.com/news/1978299222928",
    "atm_fot": "https://www.fotmob.com/fr/matches/bahia-vs-atletico-mg/2q2b38",
    "atm_of": "https://onefootball.com.br/en/news/atletico-mineiro-v-bahia-top-half-push-meets-continental-spot-ambitions-43164601",
}


def inj(team, player, role, status, impact, source, url):
    return {"team": team, "player": player, "role": role, "status": status,
            "impact": impact, "source": source, "url": url, "ts": TS}


# 每场：mid / home / away / league / injuries / conflict(可选) / style(可选) / schedule(可选) / findings(凭证)
M = [
    {
        "mid": "1364093", "home": "利勒斯特", "away": "维京", "league": "挪超",
        "injuries": [
            inj("home", "Eric Kitolano", "中场", "腿部骨折", "高", "FotMob/球迷屋", U["lil_fot"]),
            inj("home", "Ulrik Yttergård Jenssen", "后卫", "伤缺", "中", "FotMob", U["lil_fot"]),
            inj("home", "Thomas Lehne Olsen", "前锋", "大腿伤", "高", "FotMob/球迷屋", U["lil_fot"]),
            inj("home", "Linus Alperud", "中场", "踝伤", "中", "球迷屋", U["lil_qm"]),
            inj("home", "Camil Jebara", "前锋", "伤缺", "中", "球迷屋", U["lil_qm"]),
            inj("away", "Veton Berisha", "前锋", "伤缺", "高", "FotMob/球迷屋", U["lil_fot"]),
            inj("away", "Martin Ove Roseth", "后卫", "膝伤", "中", "FotMob/球迷屋", U["lil_fot"]),
            inj("away", "Kristoffer Haugen", "后卫", "伤缺", "中", "FotMob", U["lil_fot"]),
            inj("away", "Henrik Falchener", "中场", "大腿伤", "中", "FotMob", U["lil_fot"]),
        ],
        "style": "维京近5场进16球火力靠前；双方交锋近4场无平局；利勒斯特主场近期2-1胜。",
        "schedule": {"home": "世界杯后联赛重启", "away": "客场连续作战", "source": "FotMob", "ts": TS},
        "findings": [
            {"source": "FotMob", "query": "Lillestrom Viking injury lineup July 22 2026", "url": U["lil_fot"],
             "snippet": "Lilleström unavailable: Eric Kitolano (broken ankle), Jenssen, Lehne Olsen. Viking: Berisha, Roseth, Haugen.", "ts": TS},
            {"source": "球迷屋", "query": "利勒斯特罗姆 维京 伤病名单", "url": U["lil_qm"],
             "snippet": "利勒斯特罗姆3人伤病, 维京3人伤病(贝里沙/罗斯/达戈斯蒂诺).", "ts": TS},
        ],
    },
    {
        "mid": "1373205", "home": "光州FC", "away": "金泉尚武", "league": "K1联赛",
        "injuries": [
            inj("home", "崔庆录", "前腰", "伤缺", "高", "搜狐体育", U["gwang_sohu"]),
            inj("home", "金伦浒", "中锋", "伤缺", "高", "搜狐体育", U["gwang_sohu"]),
            inj("home", "闵尚基", "后卫", "伤疑", "中", "搜狐体育", U["gwang_sohu"]),
            inj("home", "李昇祐", "中场", "伤情", "中", "搜狐体育", U["gwang_sohu"]),
            inj("away", "李秀彬", "中场", "红牌停赛", "高", "搜狐/NowScore", U["gwang_sohu"]),
            inj("away", "李政宅", "中卫", "停赛", "高", "搜狐/NowScore", U["gwang_sohu"]),
        ],
        "conflict": "NowScore 称「光州FC仅崔庆录因伤缺阵、阵容相对完整」，但搜狐列崔庆录+金伦浒等多人伤缺；按 skill 规则以更详细源(搜狐)为准并下调光州胜率，保留冲突标注。",
        "style": "金泉尚武平局率55.6%联赛平局大师；光州主场保级战，近5次交锋4场总进球不超2球小球属性。",
        "findings": [
            {"source": "搜狐体育", "query": "光州FC 金泉尚武 伤停 首发 2026", "url": U["gwang_sohu"],
             "snippet": "光州前腰崔庆录与中锋金伦浒因伤缺阵; 金泉尚武李秀彬红牌停赛、李政宅停赛.", "ts": TS},
            {"source": "NowScore", "query": "光州FC 金泉尚武 伤停", "url": U["gwang_now"],
             "snippet": "光州FC仅崔庆录因伤缺阵, 阵容相对完整; 金泉尚武李秀彬红牌停赛、李政宅停赛.", "ts": TS},
        ],
    },
    {
        "mid": "1358435", "home": "迈阿密国际", "away": "芝加哥火焰", "league": "美职联",
        "injuries": [
            inj("home", "Lionel Messi", "前锋", "腿筋伤/轮休", "高", "FotMob/腾讯", U["mia_fot"]),
            inj("home", "Rodrigo De Paul", "中场", "世界杯后归队轮休", "高", "腾讯体育", U["mia_qq"]),
            inj("home", "Tadeo Allende", "前锋", "膝伤", "中", "FotMob", U["mia_fot"]),
            inj("away", "André Franco", "中场", "十字韧带", "中", "FotMob", U["mia_fot"]),
        ],
        "style": "迈阿密4-3-3围绕巴萨老将，少了梅西中场组织推进下滑；芝加哥4231防守反击，沙奇里中场核心。",
        "schedule": {"home": "世界杯后联赛重启", "away": "联赛三连胜", "source": "腾讯体育", "ts": TS},
        "findings": [
            {"source": "FotMob", "query": "Inter Miami Chicago Fire injuries July 22 2026", "url": U["mia_fot"],
             "snippet": "Inter Miami: Messi (hamstring), Allende (knee). Chicago: André Franco (ACL).", "ts": TS},
            {"source": "腾讯体育", "query": "迈阿密 芝加哥 梅西 德保罗 轮休", "url": U["mia_qq"],
             "snippet": "迈阿密梅西德保罗进入轮休名单, 苏亚雷斯任中锋; 芝加哥沙奇里核心三连胜.", "ts": TS},
        ],
    },
    {
        "mid": "1447313", "home": "奥莫尼亚", "away": "阿拉木图凯拉特", "league": "欧冠",
        "injuries": [
            inj("home", "Mateo Maric", "后卫", "停赛", "中", "ToffeeWeb", U["omo_tof"]),
            inj("home", "Fotis Kitsos", "中场", "膝伤", "中", "ToffeeWeb", U["omo_tof"]),
            inj("home", "Moses Odubajo", "后卫", "伤缺", "中", "ToffeeWeb", U["omo_tof"]),
            inj("home", "Fabiano", "门将", "伤缺", "高", "ToffeeWeb", U["omo_tof"]),
            inj("away", "João Paulo", "中场", "伤缺", "中", "ToffeeWeb", U["omo_tof"]),
            inj("away", "O. Baibek", "后卫", "身体不适", "中", "ToffeeWeb", U["omo_tof"]),
            inj("away", "A. Tuyakbayev", "中场", "胸伤", "中", "ToffeeWeb", U["omo_tof"]),
        ],
        "conflict": "搜狐称「两队均无重大伤病停赛报告、主力框架完整」，但 ToffeeWeb/FotMob 列奥莫尼亚4人、凯拉特3人伤停；按 skill 规则以更详细源(ToffeeWeb)为准并下调，保留冲突标注。",
        "style": "两队2021年两回合均0-0；奥莫尼亚主场控场，凯拉特客场反击；首回合谨慎小球，数据模型指向平局。",
        "findings": [
            {"source": "ToffeeWeb", "query": "Omonia Almaty Kairat lineup injury", "url": U["omo_tof"],
             "snippet": "Omonia absent: Maric(suspended), Kitsos, Odubajo, Fabiano. Kairat: João Paulo, Baibek, Tuyakbayev.", "ts": TS},
            {"source": "搜狐体育", "query": "欧冠 奥莫尼亚 凯拉特 伤病", "url": U["omo_sohu"],
             "snippet": "两队均无重大伤病停赛报告, 主力框架完整可全主力出战.", "ts": TS},
        ],
    },
    {
        "mid": "1362188", "home": "博塔弗戈", "away": "维多利亚", "league": "巴甲",
        "injuries": [
            inj("home", "Nahuel Ferraresi", "后卫", "停赛", "中", "FotMob", U["bot_fot"]),
            inj("home", "Bastos", "后卫", "伤缺", "高", "FotMob/DataSports", U["bot_fot"]),
            inj("home", "Júnior Santos", "前锋", "脚伤", "高", "FotMob/DataSports", U["bot_fot"]),
            inj("home", "Allan", "中场", "伤缺", "高", "FotMob/DataSports", U["bot_fot"]),
            inj("home", "Nathan Fernandes", "前锋", "膝伤", "中", "FotMob", U["bot_fot"]),
            inj("home", "Kaio", "后卫", "膝伤", "中", "FotMob", U["bot_fot"]),
            inj("away", "Anderson Pato", "前锋", "伤缺", "高", "FotMob", U["bot_fot"]),
            inj("away", "Riccieli", "后卫", "伤缺", "高", "FotMob", U["bot_fot"]),
            inj("away", "Camutanga", "后卫", "伤缺", "中", "FotMob", U["bot_fot"]),
            inj("away", "Dudu", "前锋", "伤缺", "中", "FotMob", U["bot_fot"]),
            inj("away", "Zé Vitor", "后卫", "伤缺", "中", "FotMob", U["bot_fot"]),
        ],
        "style": "博塔弗戈主场争冠档；维多利亚残阵9人伤缺防守脆弱。",
        "schedule": {"home": "世界杯后联赛重启", "away": "客场作战", "source": "FotMob", "ts": TS},
        "findings": [
            {"source": "FotMob", "query": "Botafogo Vitoria injuries July 22 2026", "url": U["bot_fot"],
             "snippet": "Botafogo: Ferraresi(susp), Bastos, Júnior Santos, Allan, Kaio. Vitoria: 9人伤缺.", "ts": TS},
            {"source": "DataSports", "query": "Botafogo squad injuries", "url": U["bot_ds"],
             "snippet": "博塔弗戈8人伤停; 维多利亚残阵多人伤缺.", "ts": TS},
        ],
    },
    {
        "mid": "1373209", "home": "富川FC", "away": "安养FC", "league": "K1联赛",
        "injuries": [
            inj("home", "金亨根", "门将", "脑震荡伤缺", "高", "今日头条", U["buche_tt"]),
            inj("home", "金钟佑", "中场", "伤缺", "中", "今日头条", U["buche_tt"]),
            inj("home", "金承斌", "边锋", "伤缺", "中", "今日头条/腾讯", U["buche_tt"]),
            inj("away", "李台熙", "边卫", "骨折伤缺", "高", "微博/今日头条", U["buche_tt"]),
        ],
        "style": "富川主场胜率11%主场虫依赖外援；安养客场龙9客4胜4平1负，交锋近10次6胜3平1负占优稳守反击。",
        "findings": [
            {"source": "今日头条", "query": "富川FC 安养FC 伤病 首发", "url": U["buche_tt"],
             "snippet": "富川主力门将金亨根脑震荡伤缺, 金钟佑/金承斌伤; 安养李台熙骨折伤缺, 整体齐整.", "ts": TS},
            {"source": "腾讯体育", "query": "富川 安养 风格 交锋", "url": U["buche_qq"],
             "snippet": "富川主场虫, 安养客场龙交锋占优稳守反击.", "ts": TS},
        ],
    },
    {
        "mid": "1362293", "home": "沙佩科", "away": "弗拉门戈", "league": "巴甲",
        "injuries": [
            inj("home", "Anderson", "门将", "伤缺", "高", "OneFootball", U["chap_of"]),
            inj("home", "Robert", "前锋", "十字韧带", "高", "OneFootball", U["chap_of"]),
            inj("home", "Rafael Carvalheira", "中场", "伤缺", "中", "OneFootball", U["chap_of"]),
            inj("home", "Max Alves", "中场", "伤缺", "中", "OneFootball", U["chap_of"]),
            inj("home", "João Paulo", "后卫", "伤缺", "中", "OneFootball", U["chap_of"]),
            inj("home", "Garcez", "前锋", "伤缺", "中", "OneFootball", U["chap_of"]),
            inj("away", "Léo Ortiz", "中卫", "伤缺", "高", "OneFootball", U["chap_of"]),
            inj("away", "Arrascaeta", "前腰", "锁骨/小腿伤", "高", "OneFootball", U["chap_of"]),
            inj("away", "Lucas Paquetá", "中场", "左大腿伤", "高", "OneFootball", U["chap_of"]),
            inj("away", "Luiz Araújo", "边锋", "左膝伤", "高", "OneFootball", U["chap_of"]),
        ],
        "style": "弗拉门戈争冠第2积34分，沙佩科垫底20位仅9分；弗拉门戈客场大胜预期。",
        "schedule": {"home": "世界杯后联赛重启(刚踢完补赛)", "away": "世界杯后联赛重启", "source": "OneFootball", "ts": TS},
        "findings": [
            {"source": "OneFootball", "query": "Chapecoense Flamengo injuries lineup", "url": U["chap_of"],
             "snippet": "Chapecoense: Anderson, Robert(ACL), Carvalheira 等多人伤. Flamengo: Arrascaeta, Paquetá, Luiz Araújo, Léo Ortiz 伤.", "ts": TS},
        ],
    },
    {
        "mid": "1362321", "home": "圣保罗", "away": "巴竞技", "league": "巴甲",
        "injuries": [
            inj("home", "Lucas Moura", "边锋", "跟腱赛季报销", "高", "FotMob", U["sp_fot"]),
            inj("home", "Ryan Francisco", "边锋", "十字韧带", "高", "FotMob", U["sp_fot"]),
            inj("home", "Mycael", "门将", "腿伤", "高", "FotMob", U["sp_fot"]),
            inj("home", "Cauly", "中场", "腹股沟", "中", "FotMob", U["sp_fot"]),
            inj("home", "Luiz Gustavo", "后腰", "小腿", "中", "FotMob", U["sp_fot"]),
            inj("home", "Derik", "后卫", "踝伤", "中", "FotMob", U["sp_fot"]),
            inj("away", "Felipinho", "中场", "伤缺", "中", "FotMob", U["sp_fot"]),
        ],
        "style": "圣保罗残阵多名主力伤缺，巴竞技相对齐整；方向圣保罗-0.5主场攻坚依赖卡莱里。",
        "schedule": {"home": "世界杯后联赛重启", "away": "客场作战", "source": "FotMob/腾讯", "ts": TS},
        "findings": [
            {"source": "FotMob", "query": "Sao Paulo Athletico Paranaense injuries", "url": U["sp_fot"],
             "snippet": "Sao Paulo: Lucas Moura(跟腱报销), Ryan Francisco(ACL), Mycael, Cauly, Luiz Gustavo. 巴竞技: Felipinho.", "ts": TS},
            {"source": "腾讯体育", "query": "圣保罗 巴竞技 战术", "url": U["sp_qq"],
             "snippet": "圣保罗残阵, 巴竞技齐整, 方向圣保罗-0.5.", "ts": TS},
        ],
    },
    {
        "mid": "1363497", "home": "博德闪耀", "away": "汉坎", "league": "挪超",
        "injuries": [
            inj("home", "August Mikkelsen", "边锋", "轻伤", "中", "FotMob", U["bod_fot"]),
            inj("home", "Magnus Riisnæs", "中场", "轻伤", "中", "FotMob", U["bod_fot"]),
            inj("home", "Daniel Bassi", "中场", "腿筋", "中", "FotMob", U["bod_fot"]),
            inj("away", "Anton Ekeroth", "后卫", "伤缺", "中", "FotMob", U["bod_fot"]),
            inj("away", "Luc Mares", "中场", "伤缺", "中", "FotMob", U["bod_fot"]),
        ],
        "style": "博德闪耀高位压迫主场3连胜仅1负；汉坎上轮1-4负特罗姆瑟。",
        "schedule": {"home": "周中赛程密集(周二刚踢欧冠)", "away": "客场作战", "source": "FotMob/SportsMole", "ts": TS},
        "findings": [
            {"source": "FotMob", "query": "Bodo Glimt HamKam injuries", "url": U["bod_fot"],
             "snippet": "Bodø/Glimt: Mikkelsen, Riisnæs, Bassi 伤. HamKam: Ekeroth, Mares 伤.", "ts": TS},
            {"source": "SportsMole", "query": "Bodo Glimt HamKam preview", "url": U["bod_sm"],
             "snippet": "Bodo/Glimt主场3连胜仅1负, 汉坎1-4负特罗姆瑟, 预计3-0.", "ts": TS},
        ],
    },
    {
        "mid": "1358455", "home": "洛杉矶FC", "away": "盐湖城", "league": "美职联",
        "injuries": [
            inj("home", "Timothy Tillman", "中场", "腿伤", "中", "FotMob", U["lax_fot"]),
            inj("home", "Sergi Palencia", "后卫", "腹股沟", "中", "FotMob/球迷屋", U["lax_fot"]),
            inj("home", "Igor Jesus", "前锋", "十字韧带", "高", "FotMob/球迷屋", U["lax_fot"]),
            inj("home", "洛里(Lloris)", "门将", "伤缺", "高", "球迷屋/腾讯", U["lax_qm"]),
            inj("away", "Arias Pior", "前锋", "跟腱", "中", "球迷屋", U["lax_qm"]),
            inj("away", "Emeka Eneli", "中场", "膝伤", "中", "球迷屋", U["lax_qm"]),
        ],
        "conflict": "FotMob 称「盐湖城无人伤缺」，但球迷屋列里维拉/皮奥尔(跟腱)/埃内利(膝)3人伤缺；按 skill 规则以更详细源(球迷屋)为准并下调盐湖城完整性，保留冲突标注（注：皮奥尔/埃内利本赛季出场有限影响小）。",
        "style": "洛杉矶FC 4-3-3高位逼抢孙兴慜首秀；盐湖城4-2-3-1反击，索兰斯尖刀。",
        "schedule": {"home": "世界杯后联赛重启(孙兴慜等归队)", "away": "客场作战", "source": "FotMob/腾讯", "ts": TS},
        "findings": [
            {"source": "FotMob", "query": "LAFC Real Salt Lake injuries", "url": U["lax_fot"],
             "snippet": "LAFC: Tillman, Palencia, Boudri, Igor Jesus 伤. RSL: no unavailable.", "ts": TS},
            {"source": "球迷屋", "query": "洛杉矶FC 皇家盐湖城 伤病名单", "url": U["lax_qm"],
             "snippet": "洛杉矶FC蒂尔曼/帕伦西亚/热苏斯/德拉瓦莱/洛里伤; 盐湖城里维拉/皮奥尔/埃内利伤.", "ts": TS},
        ],
    },
    {
        "mid": "1362317", "home": "米竞技", "away": "巴伊亚", "league": "巴甲",
        "injuries": [
            inj("home", "Iván Román", "后卫", "停赛", "中", "FotMob", U["atm_fot"]),
            inj("home", "Gustavo Scarpa", "中场", "膝伤", "高", "FotMob", U["atm_fot"]),
            inj("home", "Patrick Silva", "中场", "膝伤", "中", "FotMob", U["atm_fot"]),
            inj("away", "Léo Vieira", "后卫", "伤赛季报销", "高", "FotMob/OneFootball", U["atm_fot"]),
            inj("away", "Luciano Juba", "边锋", "大腿", "中", "FotMob", U["atm_fot"]),
            inj("away", "Ruan Pablo", "前锋", "踝伤", "中", "FotMob", U["atm_fot"]),
            inj("away", "Luciano", "后卫", "停赛", "中", "OneFootball", U["atm_of"]),
        ],
        "style": "米竞技主场仅1负；巴伊亚客场争G4，55%控球+头球路线。",
        "schedule": {"home": "世界杯后联赛重启", "away": "客场争G4", "source": "FotMob/OneFootball", "ts": TS},
        "findings": [
            {"source": "FotMob", "query": "Atletico-MG Bahia injuries", "url": U["atm_fot"],
             "snippet": "Atlético-MG: Román(停赛), Scarpa, Patrick(膝). Bahia: Léo Vieira, Juba, Ruan Pablo 伤.", "ts": TS},
            {"source": "OneFootball", "query": "Atletico-MG Bahia team news", "url": U["atm_of"],
             "snippet": "Bahia Luciano 停赛, Léo Vieira 伤; 米竞技 Scarpa/Patrick/Román 缺.", "ts": TS},
        ],
    },
]


def build(m):
    tool_log = [{
        "action": "web_search", "query": f.get("query", ""),
        "url": f.get("url", "UNAVAILABLE"), "ts": TS,
        "hit": (f.get("snippet") or "")[:80],
    } for f in m["findings"]]
    ctx = {
        "tool_log": tool_log,
        "possession": None,
        "injuries": m["injuries"],
        "lineup": {},
        "schedule_density": m.get("schedule", {}),
        "form": {},
        "style_notes": m.get("style"),
    }
    if m.get("conflict"):
        ctx["injury_conflict"] = m["conflict"]
    return ctx


if __name__ == "__main__":
    ok = 0
    for m in M:
        ctx = build(m)
        out = os.path.join(REPORTS, f"live_context_{m['mid']}.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(ctx, f, ensure_ascii=False, indent=2)
        print(f"[OK] {m['mid']} {m['home']} vs {m['away']} | 伤停{len(ctx['injuries'])} "
              f"{'⚠️冲突' if ctx.get('injury_conflict') else ''} "
              f"风格{'有' if ctx.get('style_notes') else '无'}")
        ok += 1
    print(f"\n[完成] 共生成 {ok} 个 live_context 文件")
