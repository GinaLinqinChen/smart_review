import os, json, uuid
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import json, re

def parse_rules_from_llm(resp: str):
    if not resp:
        return []

    s = resp.strip()

    # 1) 去掉 ```json 和 ``` 代码块包裹
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)

    # 2) 找到第一个 JSON 数组 [...]（非贪婪）
    m = re.search(r"\[[\s\S]*\]", s)
    if not m:
        return []

    json_str = m.group(0)

    # 3) 解析
    return json.loads(json_str)


load_dotenv()

llm = ChatOpenAI(
    model=os.environ["DEEPSEEK_CHAT_MODEL"],
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url=os.environ["DEEPSEEK_BASE_URL"],
    temperature=0
)

SYSTEM = """你是法规规则工程师。只允许基于给定法条原文生成规则。
每条规则必须包含：rule_id、rule_name、trigger、checks、evidence_required、severity、citations。
citations必须写明第几条（article_no），不要编造法条。
如果该条款不产生可执行规则（如纯背景/定义），输出空数组 []。
只输出 JSON 数组本体，不要使用 ```json 代码块，不要输出任何解释文字。
如果没有规则，输出 []。
"""

USER_TMPL = """法条编号：{article_no}
法条原文：
{article_text}

请输出规则数组（JSON）。"""

def main():
    items = json.load(open("outputs/retrieved_articles.json", "r", encoding="utf-8"))
    all_rules = []
    for it in items:
        article_no = it["article_no"]
        article_text = it["text"]
        msg = [{"role":"system","content":SYSTEM},
               {"role":"user","content":USER_TMPL.format(article_no=article_no, article_text=article_text)}]
        resp = llm.invoke(msg).content


        # 假设模型返回的是JSON数组字符串（demo里够用）
        rules = parse_rules_from_llm(resp)
        for r in rules:
            r.setdefault("rule_id", str(uuid.uuid4()))
        all_rules.extend(rules)

    # 轻量去重：rule_name + article_no
    seen = set()
    dedup = []
    for r in all_rules:
        key = (r.get("rule_name",""), json.dumps(r.get("citations",[]), ensure_ascii=False))
        if key in seen: 
            continue
        seen.add(key)
        dedup.append(r)

    with open("outputs/rules.json", "w", encoding="utf-8") as f:
        json.dump(dedup, f, ensure_ascii=False, indent=2)
    print("Saved outputs/rules.json")

if __name__ == "__main__":
    main()
