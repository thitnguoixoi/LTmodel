#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI​‑WAF service:
1. Nhận JSON chứa request thô (headers, url, body…)
2. Phân loại tấn công bằng LTModel
3. Tạo ModSecurity rule khớp payload thực tế
4. Đăng rule lên https://sample.com và trả lại URL rule
"""
from __future__ import annotations

import datetime
import hashlib
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import requests
import torch
from flask import Flask, jsonify, request

from LTModel import LTModel
import HTokenizer

###############################################################################
# CẤU HÌNH CHUNG
###############################################################################
MODEL_CHECKPOINT = "model\\final_model_complete.pt"
VOCAB_FILE = "final_filtered_vocab.json"
MAX_SEQ_LENGTH = 256

SAMPLE_API = "https://sample.com/api/rules"        # REST endpoint publish rule
SAMPLE_BEARER = "INSERT_YOUR_TOKEN_HERE"           # Bearer / API​‑Key nếu cần

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("aiwaf")

###############################################################################
# KHỚI TẠO MÔ HÌNH & TOKENIZER
###############################################################################
with open(VOCAB_FILE, encoding="utf-8") as f:
    VOCAB: Dict[str, int] = json.load(f)

tokenizer = HTokenizer.HTokenizer()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model() -> LTModel:
    ckpt = torch.load(MODEL_CHECKPOINT, map_location=device)
    model = LTModel(
        vocab_size=len(VOCAB),
        hidden_size=128,
        num_layers=2,
        num_heads=8,
        ff_size=2048,
        dropout=0.1,
        num_classes=8,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model

model = get_model()

###############################################################################
# HÀM XỨ LÝ TOKEN
###############################################################################
def tokenize_and_pad(json_data: Dict[str, Any]) -> list[str]:
    tokens = tokenizer.tokenize_json(json_data)
    if len(tokens) < MAX_SEQ_LENGTH:
        tokens += ["<pad>"] * (MAX_SEQ_LENGTH - len(tokens))
    else:
        tokens = tokens[:MAX_SEQ_LENGTH]
    return tokens

def tokens_to_ids(tokens: list[str]) -> list[int]:
    return [VOCAB.get(t, VOCAB.get("<unk>", 0)) for t in tokens]

###############################################################################
# RULE​‑GEN BUILDER
###############################################################################
def _rule_id() -> int:
    iso = datetime.datetime.utcnow().strftime("%y%m%d%H%M%S")
    suffix = hashlib.md5(iso.encode()).hexdigest()[:2]
    return int(iso + suffix, 16) % 2_147_483_647

def _extract_payload(req: Dict[str, Any]) -> str:
    return f"{req.get('url','')} {req.get('body','')}"

def _regex_escape_payload(text: str) -> str:
    cleaned = text.strip().replace("\n", " ").replace("\r", " ")
    cleaned = re.sub(r"\s+", " ", cleaned)[:300]
    escaped = re.escape(cleaned)
    return f"(?i){escaped}"

def generate_modsec_rule(req_json: Dict[str, Any], attack_type: str) -> Tuple[int | None, str | None]:
    if attack_type in ("Normal", "Unknown"):
        return None, None

    rid = _rule_id()
    req = req_json.get("request", {})
    raw_payload = _extract_payload(req)
    escaped_pattern = _regex_escape_payload(raw_payload)

    if attack_type == "SQL Injection":
        rule = f'SecRule ARGS "@rx {escaped_pattern}" "id:{rid},phase:2,deny,status:403,msg:\'AI​‑SQLi\',log"'
    elif attack_type == "Directory Traversal":
        rule = f'SecRule REQUEST_URI "@rx {escaped_pattern}" "id:{rid},phase:2,deny,status:403,msg:\'AI​‑Traversal\',log"'
    elif attack_type == "XSS":
        rule = f'SecRule ARGS|REQUEST_URI "@rx {escaped_pattern}" "id:{rid},phase:2,deny,status:403,msg:\'AI​‑XSS\',log"'
    elif attack_type == "Log Forging":
        rule = f'SecRule REQUEST_URI|ARGS "@rx {escaped_pattern}" "id:{rid},phase:2,deny,status:403,msg:\'AI​‑Log​‑Forging\',log"'
    elif attack_type == "Cookie Injection":
        rule = f'SecRule REQUEST_HEADERS:Cookie "@rx {escaped_pattern}" "id:{rid},phase:2,deny,status:403,msg:\'AI​‑Cookie​‑Inject\',log"'
    elif attack_type == "RCE":
        rule = f'SecRule ARGS "@rx {escaped_pattern}" "id:{rid},phase:2,deny,status:403,msg:\'AI​‑RCE\',log"'
    elif attack_type == "LOG4J":
        rule = f'SecRule REQUEST_HEADERS:User-Agent "@rx {escaped_pattern}" "id:{rid},phase:2,deny,status:403,msg:\'AI​‑Log4Shell\',log"'
    else:
        return None, None

    return rid, rule

###############################################################################
# PUBLISHER
###############################################################################
def publish_rule(rule_id: int, rule_txt: str) -> str | None:
    headers = {"Authorization": f"Bearer {SAMPLE_BEARER}"} if SAMPLE_BEARER else {}
    try:
        resp = requests.post(
            SAMPLE_API,
            json={"id": rule_id, "rule": rule_txt},
            headers=headers,
            timeout=5,
        )
        resp.raise_for_status()
        return resp.json().get("url")
    except requests.RequestException as exc:
        logger.warning("Publish rule failed: %s", exc)
        return None

###############################################################################
# FLASK APP
###############################################################################
app = Flask(__name__)

LABEL_MAP = {
    0: "Normal",
    1: "Directory Traversal",
    2: "SQL Injection",
    3: "XSS",
    4: "Log Forging",
    5: "Cookie Injection",
    6: "RCE",
    7: "LOG4J",
}

@app.post("/detect")
def detect() -> Any:
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify(error="JSON payload required"), 400

    toks = tokenize_and_pad(data)
    ids = tokens_to_ids(toks)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)

    with torch.no_grad():
        out = model(input_ids)
        logits = out["logits"]
        pred_cls = logits.argmax(dim=-1).item()

    attack_type = LABEL_MAP.get(pred_cls, "Unknown")

    rule_id, rule_txt = generate_modsec_rule(data, attack_type)
    rule_url = publish_rule(rule_id, rule_txt) if rule_txt else None

    return jsonify(
        predicted_id=pred_cls,
        attack_type=attack_type,
        modsecurity_rule=rule_txt,
        rule_url=rule_url,
    )

###############################################################################
# MAIN
###############################################################################
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8888)
