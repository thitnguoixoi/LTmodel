#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI‑WAF service (v5 – simple POST)
---------------------------------
• Nhận JSON request thô → phân loại → tạo ModSecurity rule.
• Giờ **chỉ cần POST** rule tới một URL bất kỳ (mặc định dùng posttestserver.dev).
"""
from __future__ import annotations

import base64
import datetime
import hashlib
import json
import logging
import re
import sys
import urllib.parse
from typing import Any, Dict, List, Tuple

import requests
import torch
from flask import Flask, jsonify, request

from LTModel import LTModel
import HTokenizer

###############################################################################
# CONFIG
###############################################################################
MODEL_CHECKPOINT = "model\\final_model_complete.pt"
VOCAB_FILE = "final_filtered_vocab.json"
MAX_SEQ_LENGTH = 256

# URL đích để gửi rule – có thể thay bằng bất kỳ URL nào người vận hành muốn
POST_ENDPOINT = "<URL rule>"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s", handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("aiwaf")

###############################################################################
# MODEL / TOKENIZER INIT
###############################################################################
with open(VOCAB_FILE, encoding="utf-8") as f:
    VOCAB: Dict[str, int] = json.load(f)

tokenizer = HTokenizer.HTokenizer()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model() -> LTModel:
    ckpt = torch.load(MODEL_CHECKPOINT, map_location=device)
    model = LTModel(vocab_size=len(VOCAB), hidden_size=128, num_layers=2, num_heads=8, ff_size=2048, dropout=0.1, num_classes=8).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model

model = get_model()

###############################################################################
# TOKEN UTILITIES
###############################################################################

def tokenize_and_pad(data: Dict[str, Any]) -> List[str]:
    print("Tokenizing input data...")  # Debug: print message
    print("Input data:", data)  # Debug: print input data
    toks = tokenizer.tokenize_json(data)
    print(toks)  # Debug: print tokens")
    return (toks + ["<pad>"] * MAX_SEQ_LENGTH)[:MAX_SEQ_LENGTH]

def tokens_to_ids(tokens: List[str]) -> List[int]:
    return [VOCAB.get(t, VOCAB["<unk>"]) for t in tokens]

###############################################################################
# PATTERN DICTIONARIES (unchanged)
###############################################################################
SQLI_KW = [r"union\s+select", r"select\s+.+\bfrom\b", r"insert\s+into", r"update\s+\w+\s+set", r"delete\s+from", r"drop\s+table", r"(or|and)\s+1=1", r"sleep\(\d+\)", r"benchmark\(", r"limit\s*\d+"]
XSS_KW = [r"<script\b", r"onerror\s*=", r"onload\s*=", r"alert\(", r"javascript:"]
TRAV_KW = [r"\.\./", r"%2e%2e/"]
RCE_KW = [r"(;|&&|\|)\s*(bash|sh|nc|wget|curl|python)"]
LOG4J_KW = [r"\$\{jndi:(ldap[s]?:|rmi:|dns:)"]
COOKIE_KW = [r"jsessionid=", r"phpsessid="]
LOGFORGE_KW = [r"%0d%0a", r"\n", r"\r"]
PATTERN_MAP: Dict[str, List[str]] = {
    "SQL Injection": SQLI_KW,
    "XSS": XSS_KW,
    "Directory Traversal": TRAV_KW,
    "RCE": RCE_KW,
    "LOG4J": LOG4J_KW,
    "Cookie Injection": COOKIE_KW,
    "Log Forging": LOGFORGE_KW,
}

###############################################################################
# HELPER FUNCTIONS (decode, detect field, build pattern)
###############################################################################

def _rule_id() -> int:
    iso = datetime.datetime.utcnow().strftime("%y%m%d%H%M%S")
    return int(iso + hashlib.md5(iso.encode()).hexdigest()[:2], 16) % 2_147_483_647


def _decode_variants(value: str) -> List[str]:
    variants = [value]
    try:
        decoded_url = urllib.parse.unquote_plus(value)
        if decoded_url not in variants:
            variants.append(decoded_url)
    except Exception:
        pass
    b = value.strip()
    if len(b) % 4:
        b += "=" * (4 - len(b) % 4)
    try:
        decoded_b64 = base64.b64decode(b, validate=True)
        decoded_text = decoded_b64.decode("utf-8", errors="ignore")
        if decoded_text and decoded_text not in variants:
            variants.append(decoded_text)
    except Exception:
        pass
    return variants


def _flatten(req: Any) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    if isinstance(req, str):
        out.append(("", req))
    elif isinstance(req, dict):
        for k, v in req.items():
            for _, s in _flatten(v):
                out.append((k, s))
    elif isinstance(req, list):
        for item in req:
            out.extend(_flatten(item))
    return out


def _detect_field(req: Dict[str, Any], kw_list: List[str]) -> str:
    """Xác định trường chứa payload bằng cách:
    • Lấy từng giá trị (URL, body, header…)
    • Tokenize bằng `HTokenizer.process_attack`
    • Ghép token lại thành chuỗi rồi so khớp từ khóa
    """
    regex = re.compile("|".join(kw_list), re.I)

    # URL
    url_val = req.get("url", "")
    if url_val:
        tok_text = " ".join(tokenizer.process_attack(url_val))
        if regex.search(tok_text):
            return "REQUEST_URI"

    # BODY / ARGS
    body_val = req.get("body", "")
    if body_val:
        tok_text = " ".join(tokenizer.process_attack(body_val))
        if regex.search(tok_text):
            return "ARGS"

    # HEADERS
    headers: Dict[str, str] = req.get("headers", {})
    for h_name, h_val in headers.items():
        if h_name.lower() == "cookie":
            tok_text = " ".join(tokenizer.process_cookie(h_val))  # dùng hàm đặc thù cho Cookie
        else:
            tok_text = " ".join(tokenizer.process_attack(h_val))
        if regex.search(tok_text):
            return "REQUEST_HEADERS:Cookie" if h_name.lower() == "cookie" else f"REQUEST_HEADERS:{h_name}"

    return "REQUEST_URI|ARGS|REQUEST_HEADERS"


def _build_pattern(kw_list: List[str], req: Dict[str, Any]) -> str:
    """Ghép token (process_attack & process_cookie) tất cả trường rồi chọn keyword."""
    tokens: List[str] = []

    # URL & Body
    for key in ("url", "body"):
        val = req.get(key, "")
        if val:
            tokens.extend(tokenizer.process_attack(val))

    # Headers
    for h_name, h_val in req.get("headers", {}).items():
        if h_name.lower() == "cookie":
            tokens.extend(tokenizer.process_cookie(h_val))
        else:
            tokens.extend(tokenizer.process_attack(h_val))

    text_pool = " ".join(tokens)[:2000]
    picks = [k for k in kw_list if re.search(k, text_pool, re.I)]
    if not picks:
        picks = kw_list[:3]
    return "(?i)(" + "|".join(picks) + ")"

###############################################################################
# RULE GENERATOR
###############################################################################

def generate_modsec_rule(req_json: Dict[str, Any], attack_type: str) -> Tuple[int | None, str | None]:
    if attack_type in ("Normal", "Unknown"):
        return None, None
    kw_list = PATTERN_MAP.get(attack_type)
    if not kw_list:
        return None, None
    req = req_json.get("request", {})
    variable = _detect_field(req, kw_list)
    pattern = _build_pattern(kw_list, req)
    rid = _rule_id()
    rule = f'SecRule {variable} "@rx {pattern}" "id:{rid},phase:2,deny,status:403,msg:\'AI-{attack_type.replace(" ", "-")}\',log"'
    return rid, rule

###############################################################################
# SIMPLE PUBLISHER – chỉ POST JSON tới POST_ENDPOINT
###############################################################################

def publish_rule(rule_id: int, rule_txt: str) -> str | None:
    try:
        resp = requests.post(POST_ENDPOINT, json={"id": rule_id, "rule": rule_txt}, timeout=5)
        logger.info("Posted rule to %s status=%s", POST_ENDPOINT, resp.status_code)
        return POST_ENDPOINT  # posttestserver giữ log, URL đủ tra cứu
    except requests.RequestException as exc:
        logger.warning("Publish rule failed: %s", exc)
        return None

###############################################################################
# FLASK ENDPOINT
###############################################################################
app = Flask(__name__)
LABEL_MAP = {0: "Normal", 1: "Directory Traversal", 2: "SQL Injection", 3: "XSS", 4: "Log Forging", 5: "Cookie Injection", 6: "RCE", 7: "LOG4J"}

@app.post("/detect")
def detect() -> Any:
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify(error="JSON payload required"), 400

    toks = tokenize_and_pad(data)
    input_ids = torch.tensor([tokens_to_ids(toks)], dtype=torch.long, device=device)
    with torch.no_grad():
        pred_cls = model(input_ids)["logits"].argmax(dim=-1).item()
    attack_type = LABEL_MAP.get(pred_cls, "Unknown")
    rule_id, rule_txt = generate_modsec_rule(data, attack_type)
    rule_url = publish_rule(rule_id, rule_txt) if rule_txt else None
    return jsonify(predicted_id=pred_cls, attack_type=attack_type, modsecurity_rule=rule_txt, rule_url=rule_url)

###############################################################################
# RUN
###############################################################################
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8888)
