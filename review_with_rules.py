import os
import json
import re
import requests
from typing import List
from dotenv import load_dotenv

from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI

load_dotenv()

RULES_PATH = os.getenv("RULES_PATH", "outputs/rules.json")
CONTRACT_INDEX_DIR = os.getenv("CONTRACT_INDEX_DIR", "outputs/contract_faiss")
OUT_PATH = os.getenv("FINDINGS_PATH", "outputs/findings.json")
TOPK_EVIDENCE = int(os.getenv("TOPK_EVIDENCE", "5"))


class OllamaNativeEmbeddings(Embeddings):
    def __init__(self, base_url: str, model: str, timeout: int = 60):
        self.base_url = base_url.rstrip("/")
        self.model = model.replace(":latest", "")
        self.timeout = timeout

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        url = f"{self.base_url}/api/embed"
        payload = {"model": self.model, "input": texts}
        r = requests.post(url, json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()["embeddings"]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


def build_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.environ["DEEPSEEK_CHAT_MODEL"],
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url=os.environ["DEEPSEEK_BASE_URL"],
        temperature=0,
    )


SYSTEM = """你是文件审查员。只能依据提供的“证据片段”判断，不得编造。
你必须输出一个JSON对象，字段必须包含：
status: "PASS" | "FAIL" | "UNCERTAIN"
why: 简要理由（必须引用证据内容）
contract_evidence: 数组，每项包含 chunk_id, approx_loc, snippet
fix_suggestion: 修改/补充建议
missing_info: 若UNCERTAIN，写清缺什么；否则为空字符串
仅输出JSON，不要输出markdown代码块，不要输出解释文字。
"""

USER_TMPL = """审查规则（JSON）：
{rule_json}

证据片段（可能相关，已按相关性排序）：
{evidence_text}

请输出JSON："""


def build_query(rule: dict) -> str:
    rule_name = rule.get("rule_name", "")
    trigger = rule.get("trigger", "")
    checks = rule.get("checks", [])
    checks_text = "；".join([str(x) for x in checks[:4]]) if isinstance(checks, list) else str(checks)
    return f"{rule_name} {trigger} {checks_text}".strip()


def safe_json_load(s: str) -> dict:
    s = (s or "").strip()
    # 去掉 ```json 包裹
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)
    if not s:
        return {}
    if not s.startswith("{"):
        m = re.search(r"\{[\s\S]*\}", s)
        if m:
            s = m.group(0)
    return json.loads(s)


def main():
    if not os.path.exists(RULES_PATH):
        raise FileNotFoundError(f"rules.json not found: {RULES_PATH}")
    if not os.path.exists(CONTRACT_INDEX_DIR):
        raise FileNotFoundError(f"contract_faiss not found: {CONTRACT_INDEX_DIR}")

    rules = json.load(open(RULES_PATH, "r", encoding="utf-8"))

    emb = OllamaNativeEmbeddings(
        base_url=os.getenv("EMBED_BASE_URL", "http://127.0.0.1:11434"),
        model=os.getenv("EMBED_MODEL", "bge-m3"),
    )

    vs = FAISS.load_local(CONTRACT_INDEX_DIR, emb, allow_dangerous_deserialization=True)
    llm = build_llm()

    findings = []

    for rule in rules:
        query = build_query(rule)
        docs = vs.similarity_search(query, k=TOPK_EVIDENCE)

        evidence_blocks = []
        for d in docs:
            snippet = (d.page_content or "").strip()[:600]
            evidence_blocks.append(
                f"[{d.metadata.get('chunk_id')} | {d.metadata.get('approx_loc')}]\n{snippet}"
            )

        user_msg = USER_TMPL.format(
            rule_json=json.dumps(rule, ensure_ascii=False),
            evidence_text="\n\n---\n\n".join(evidence_blocks) if evidence_blocks else "(无证据命中)",
        )

        resp = llm.invoke(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_msg}]
        ).content

        out = safe_json_load(resp)

        findings.append(
            {
                "rule_id": rule.get("rule_id"),
                "rule_name": rule.get("rule_name"),
                "status": out.get("status", "UNCERTAIN"),
                "why": out.get("why", ""),
                "contract_evidence": out.get("contract_evidence", []),
                "fix_suggestion": out.get("fix_suggestion", ""),
                "missing_info": out.get("missing_info", ""),
                "rule_citations": rule.get("citations", []),
                "query_used": query,
            }
        )

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(findings, f, ensure_ascii=False, indent=2)

    print(f"Saved findings to {OUT_PATH}, findings={len(findings)}")


if __name__ == "__main__":
    main()
