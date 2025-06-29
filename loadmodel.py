#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI‑WAF service (v6‑r3 – raw‑value rules)
----------------------------------------
• Nhận JSON request thô → mô hình Torch phân loại → sinh ModSecurity rule.
• Logic đáp ứng các yêu cầu:
    1. Cookie Injection → viết rule khớp **giá trị gốc** của trường Cookie.
    2. Các kiểu tấn công khác → xác định trường đầu tiên chứa keyword, viết rule
       khớp **giá trị gốc** của chính trường đó.
    3. Vẫn decode URL & Base64 cho từng trường để *phát hiện* keyword.
      (không dùng giá trị decode để tạo pattern).
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
import HTokenizer  # cung cấp process_attack & process_cookie

###############################################################################
# CONFIG
###############################################################################
MODEL_CHECKPOINT = "model\\final_model_complete.pt"
VOCAB_FILE = "final_filtered_vocab.json"
MAX_SEQ_LENGTH = 256

# URL đích để gửi rule – thay bằng endpoint thật trong môi trường sản xuất
POST_ENDPOINT = "<URL rule>"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("aiwaf")

###############################################################################
# MODEL / TOKENIZER INIT
###############################################################################
with open(VOCAB_FILE, encoding="utf-8") as f:
    VOCAB: Dict[str, int] = json.load(f)

tokenizer = HTokenizer.HTokenizer()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _load_model() -> LTModel:
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


model = _load_model()

###############################################################################
# TOKEN UTILITIES
###############################################################################

def _tokenize_and_pad(data: Dict[str, Any]) -> List[str]:
    toks = tokenizer.tokenize_json(data)
    return (toks + ["<pad>"] * MAX_SEQ_LENGTH)[:MAX_SEQ_LENGTH]


def _tokens_to_ids(tokens: List[str]) -> List[int]:
    return [VOCAB.get(t, VOCAB["<unk>"]) for t in tokens]

###############################################################################
# PATTERN DICTIONARIES
###############################################################################
SQLI_KW = [
    r"union\s+select",
    r"select\s+.+\bfrom\b",
    r"insert\s+into",
    r"update\s+\w+\s+set",
    r"delete\s+from",
    r"drop\s+table",
    r"(or|and)\s+1=1",
    r"sleep\(\d+\)",
    r"benchmark\(",
    r"limit\s*\d+",
]
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
# HELPER FUNCTIONS
###############################################################################

def _rule_id() -> int:
    """Sinh ID ModSecurity pseudo‑unique dựa trên thời gian UTC."""
    iso = datetime.datetime.utcnow().strftime("%y%m%d%H%M%S")
    return int(iso + hashlib.md5(iso.encode()).hexdigest()[:2], 16) % 2_147_483_647


def _decode_variants(value: str) -> List[str]:
    """Trả về [raw, url‑decoded, base64‑decoded*] (nếu decode thành công)."""
    variants = [value]

    # URL decode
    try:
        dec_url = urllib.parse.unquote_plus(value)
        if dec_url not in variants:
            variants.append(dec_url)
    except Exception:
        pass

    # Base64 decode (thêm padding nếu thiếu)
    b = value.strip()
    if len(b) % 4:
        b += "=" * (4 - len(b) % 4)
    try:
        dec_b64 = base64.b64decode(b, validate=True)
        dec_txt = dec_b64.decode("utf-8", errors="ignore")
        if dec_txt and dec_txt not in variants:
            variants.append(dec_txt)
    except Exception:
        pass

    return variants

###############################################################################
# FIELD DETECTION & PATTERN BUILDING
###############################################################################

def _detect_field(
    req: Dict[str, Any], kw_list: List[str], attack_type: str
) -> Tuple[str, str | None]:
    """Xác định trường chứa payload.
    • Trả về (ModSec variable, raw_value) – raw_value là GIÁ TRỊ GỐC.
    • Cookie Injection → luôn header Cookie.
    • Kiểu khác: duyệt URL, body, headers; decode để dò keyword nhưng luôn
      trả về raw_value.
    """
    if attack_type == "Cookie Injection":
        raw_cookie = (
            req.get("headers", {}).get("Cookie")
            or req.get("headers", {}).get("cookie")
            or ""
        )
        return "REQUEST_HEADERS:Cookie", raw_cookie

    regex = re.compile("|".join(kw_list), re.I)

    # URL
    url_raw = req.get("url", "")
    for v in _decode_variants(url_raw):
        if regex.search(" ".join(tokenizer.process_attack(v))):
            return "REQUEST_URI", url_raw

    # BODY / ARGS
    body_raw = req.get("body", "")
    for v in _decode_variants(body_raw):
        if regex.search(" ".join(tokenizer.process_attack(v))):
            return "ARGS", body_raw

    # HEADERS
    for h_name, h_val_raw in req.get("headers", {}).items():
        proc = tokenizer.process_cookie if h_name.lower() == "cookie" else tokenizer.process_attack
        for v in _decode_variants(h_val_raw):
            if regex.search(" ".join(proc(v))):
                var = (
                    "REQUEST_HEADERS:Cookie"
                    if h_name.lower() == "cookie"
                    else f"REQUEST_HEADERS:{h_name}"
                )
                return var, h_val_raw

    # fallback
    return "REQUEST_URI|ARGS|REQUEST_HEADERS", None


def _build_raw_pattern(raw_value: str) -> str:
    """Tạo regex case‑insensitive khớp NGUYÊN GIÁ TRỊ GỐC."""
    if raw_value is None:
        return "(?!)"
    return "(?i)" + re.escape(raw_value)

###############################################################################
# RULE GENERATOR
###############################################################################

def generate_modsec_rule(
    req_json: Dict[str, Any], attack_type: str
) -> Tuple[int | None, str | None]:
    if attack_type in ("Normal", "Unknown"):
        return None, None

    kw_list = PATTERN_MAP.get(attack_type)
    if not kw_list:
        return None, None

    req = req_json.get("request", {})
    variable, raw_val = _detect_field(req, kw_list, attack_type)
    if not raw_val:
        return None, None

    pattern = _build_raw_pattern(raw_val)
    rid = _rule_id()
    rule = (
        f'SecRule {variable} "@rx {pattern}" '
        f'"id:{rid},phase:2,deny,status:403,msg:\'AI-{attack_type.replace(" ", "-")}\',log"'
    )
    return rid, rule

###############################################################################
# RULE PUBLISHER
###############################################################################

def publish_rule(rule_id: int, rule_txt: str) -> str | None:
    """POST rule JSON tới POST_ENDPOINT; trả về URL log (nếu có)."""
    if not POST_ENDPOINT or POST_ENDPOINT.startswith("<"):
        logger.warning("POST_ENDPOINT not exist – skip publish.")
        return None
    try:
        resp = requests.post(
            POST_ENDPOINT, json={"id": rule_id, "rule": rule_txt}, timeout=5
        )
        logger.info("Posted rule to %s status=%s", POST_ENDPOINT, resp.status_code)
        return POST_ENDPOINT
    except requests.RequestException as exc:
        logger.warning("Publish rule failed: %s", exc)
        return None

###############################################################################
# FLASK SERVICE
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

    toks = _tokenize_and_pad(data)
    input_ids = torch.tensor([_tokens_to_ids(toks)], dtype=torch.long, device=device)

    with torch.no_grad():
        pred_cls = model(input_ids)["logits"].argmax(dim=-1).item()

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
# RUN APP
###############################################################################
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8888)
