import os
import requests
import pandas as pd
import re
import urllib.parse
from collections import Counter
from nltk.tokenize import RegexpTokenizer
import base64
import math
import string
from urllib.parse import unquote

# Configuration
DATA_DIR = "http_headers_data"
VOCAB_DIR = os.path.join(DATA_DIR, "vocab")


# Data source URLs
URLS = {
    "ssti": "https://raw.githubusercontent.com/swisskyrepo/PayloadsAllTheThings/master/Server%20Side%20Template%20Injection/Intruder/ssti.fuzz",
    "cmd_inject": "https://raw.githubusercontent.com/swisskyrepo/PayloadsAllTheThings/master/Command%20Injection/Intruder/command-execution-unix.txt",
    "sql_inject": "https://github.com/foospidy/payloads/blob/master/owasp/fuzzing_code_database/sqli/sqli.txt",
    "xss": "https://raw.githubusercontent.com/swisskyrepo/PayloadsAllTheThings/master/XSS%20Injection/Intruders/xss_alert.txt",
    "path_trav": "https://raw.githubusercontent.com/swisskyrepo/PayloadsAllTheThings/master/Directory%20Traversal/Intruder/directory_traversal.txt",
}


class VocabGen:
    """Vocabulary and Regex Generator for HTTP traffic analysis."""

    def __init__(self):
        """Initialize the vocabulary generator."""
        self.word_counter = Counter()
        self.normal_tokens = set()
        self.anomalous_tokens = set()
        self.combined_tokens = set()
        self.tld_set = self.load_tld_list()
        # Initialize tokenizers for different header types
        self.tokenizer = RegexpTokenizer(
            r'(?:ldap|rmi|ldaps|dns|iiop|nis|http|https|ftp|file)(?=[a-z])'
            r'|\.\./|\.\.\\|:\/\/'
            r'|[a-zA-Z0-9_.-]+\.(?:txt|csv|log|exe|py|php|html?|json|js|xml|conf|sh|bat|dll|zip|tar|gz|pdf|docx|doc|pptx|xlsx|ppt|xls|docm|dotx|dotm|potx|potm|ppsx|ppsm|pps|dot|jpg|jpeg|png|gif|bmp|tiff|tif|svg|webp|mp4|avi|mov|mkv|wmv|flv|mp3|wav|ogg|aac|wma|m4a|aspx)'
            r'|\(|\)|\;|\=|\:|\,|\&|\%|\\|\/|\+|\?|\#|\[|\]|\{|\}|\~'
            r'|\d{2}:\d{2}:\d{2}'
            r'|[A-Za-z0-9\.]+'
            r'|[A-Za-z0-9]+'
        )
        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(VOCAB_DIR, exist_ok=True)

    def run(self):
        """Run the complete vocabulary generation process."""
        self.collect_http_headers()
        self.collect_dataset()
        self.collect_attack_payloads()
        self.save_vocab_files()

    def collect_dataset(self):
        """Collect tokens from the dataset."""
        print("Collecting tokens from the dataset...")
        # Read the CSV file and extract the headers
        df = pd.read_csv("dataset\\alldata_balanced.csv", low_memory=False)
        df = df.drop(columns=["request.Attack_Tag"], axis=1)
        # Process each header
        for header in df.columns:
            header_name = header.split(".")[-1]
            header_name = header_name.lower()
            print(f"Processing header: {header_name}")
            if header_name not in self.normal_tokens:
                self.normal_tokens.add(header_name)
            if header_name == "cookie" or header_name == "set-cookie":
                self.process_tokenize_cookie(
                    df[header].unique().tolist())
            else:
                self.process_tokenize(
                    df[header].unique().tolist())

    def collect_http_headers(self):
        print("Collecting tokens from HTTP headers...")
        df = pd.read_csv("dataset\\Book1.csv", low_memory=False)
        # Read the CSV file and extract the headers
        # Process each header
        for header in df.columns:
            header = header.lower()
            print(f"Processing header: {header}")
            if header not in self.normal_tokens:
                self.normal_tokens.add(header)
            if header == "cookie" or header == "set-cookie":
                self.process_tokenize_cookie(
                    df[header].unique().tolist())
            else:
                self.process_tokenize(
                    df[header].unique().tolist())

    def collect_attack_payloads(self):
        """Collect tokens from attack payloads."""
        print("Collecting tokens from attack payloads...")
        attack_payload_keys = ["path_trav",
                               "cmd_inject", "sql_inject", "xss", "ssti"]
        for key in attack_payload_keys:
            if key not in URLS:
                continue

            try:
                response = requests.get(URLS[key])
                if response.status_code == 200:
                    payload_text = response.content.decode(errors="ignore")
                    for line in payload_text.splitlines():
                        # Store the whole line as an anomalous token
                        decoded = urllib.parse.unquote(line.strip())
                        decoded = decoded.replace("--", " ")
                        decoded = decoded.replace("_", " ")
                        decoded = decoded.replace("'", "")
                        decoded = decoded.replace("[", "")
                        decoded = decoded.replace("]", "")
                        tokens = self.tokenizer.tokenize(decoded)
                        tokens = self.clean_tokens(tokens)
                        tokens = [token.lower() for token in tokens]
                        self.anomalous_tokens.update(tokens)
                else:
                    print(
                        f"Failed to fetch {key}: HTTP {response.status_code}")
            except Exception as e:
                print(f"Failed to fetch {key}: {str(e)}")

    def save_vocab_files(self):
        """Save vocabulary files to disk."""
        print("Saving vocabulary files...")
        os.makedirs(VOCAB_DIR, exist_ok=True)

        with open(os.path.join(VOCAB_DIR, "vocab.txt"), "w", encoding="utf-8") as f:
            self.combined_tokens = sorted(
                self.normal_tokens.union(self.anomalous_tokens)
            )
            # Add <unk> and <pad> at the end of the combined tokens
            self.combined_tokens.append("<unk>")
            self.combined_tokens.append("<pad>")

            for token in self.combined_tokens:
                f.write(f"{token}\n")

        vocab_index = {token: idx for idx,
                       token in enumerate(self.combined_tokens)}

        with open(os.path.join(VOCAB_DIR, "vocab.json"), "w", encoding="utf-8") as f_json:
            import json
            json.dump(vocab_index, f_json, ensure_ascii=False, indent=4)

        print(f"Vocabulary files saved to {VOCAB_DIR}")
        print(f"  - Normal tokens: {len(self.normal_tokens)}")
        print(f"  - Anomalous tokens: {len(self.anomalous_tokens)}")
        print(f"  - Combined total: {len(self.combined_tokens)}")

    def calculate_entropy(self, s):
        """
        Tính entropy trung bình của chuỗi (đo độ ngẫu nhiên).
        """
        if not s:
            return 0
        counter = Counter(s)
        total = len(s)
        entropy = -sum((count/total) * math.log2(count/total)
                       for count in counter.values())
        return entropy

    def is_random_string(self, s, entropy_threshold=3.5):
        """
        Nhận biết chuỗi có phải chuỗi ngẫu nhiên hay không.

        - entropy_threshold: càng cao thì độ ngẫu nhiên càng lớn (giá trị 4.0 là hợp lý)
        """

        s = s.strip()
        if not s:
            return False
        # 1. Nếu ký tự đầu là chữ cái và lặp lại liên tiếp → random
        first_char = s[0]
        if first_char.isalpha():  # chỉ xét nếu là chữ cái a-z, A-Z
            repeat_count = 1
            for i in range(1, len(s)):
                if s[i] == first_char:
                    repeat_count += 1
                else:
                    break
            if repeat_count >= 2:
                return True

        # 2. Nếu chứa cả chữ và số (không phải chỉ toàn số) thì đánh dấu là random
        if re.search(r"[A-Za-z]", s) and re.search(r"\d", s):
            return True

        # 3. Tính entropy
        entropy = self.calculate_entropy(s)

        # 4. Nếu chứa nhiều ký tự đặc biệt và entropy cao => random
        special_chars = set(s) - set(string.ascii_letters + string.digits)
        if entropy >= entropy_threshold and len(special_chars) >= 2:
            return True

        # 5. Nếu entropy cao và có nhiều loại ký tự khác nhau => khả năng random cao
        if entropy >= entropy_threshold:
            return True

        return False

    def is_file_path(self, s):
        """
        Nhận biết chuỗi là file path hoặc tên file (Windows, Unix, hoặc tên file đơn lẻ).
        """
        # Pattern kiểm tra path hoặc file có phần mở rộng
        pattern = re.compile(
            r"""(
                (?:[a-zA-Z]:\\|/)?                     # ổ đĩa hoặc bắt đầu với /
                # thư mục (có thể có nhiều cấp)
                (?:[\w\-\. ]+[\\/])*
                [\w\-\. ]+\.(?:txt|csv|log|exe|py|php|html?|json|js|xml|conf|sh|bat|dll|zip|tar|gz|pdf|docx|doc|pptx|xlsx|ppt|xls|docm|dotx|dotm|potx|potm|ppsx|ppsm|pps|dot|jpg|jpeg|png|gif|bmp|tiff|tif|svg|webp|mp4|avi|mov|mkv|wmv|flv|mp3|wav|ogg|aac|wma|m4a|aspx?)
            )$""",
            re.VERBOSE | re.IGNORECASE
        )
        return bool(pattern.match(s))

    def load_tld_list(self):
        url = "https://data.iana.org/TLD/tlds-alpha-by-domain.txt"
        response = requests.get(url)
        response.raise_for_status()
        lines = response.text.splitlines()
        tlds = set(line.lower() for line in lines if not line.startswith("#"))
        return tlds

    def is_domain(self, s):
        """
        Kiểm tra xem chuỗi s có phải là tên miền hợp lệ không.
        """
        if not s or not isinstance(s, str):
            return False

        # Loại bỏ dấu chấm đầu (.) hoặc tiền tố như www.
        s = s.lstrip('.').lower()
        s = re.sub(r'^www\.', '', s)

        # Regex cơ bản kiểm tra định dạng tên miền
        pattern = re.compile(r"^(?!:\/\/)([a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}$")
        match = pattern.match(s)
        if not match:
            return False

        # Lấy tất cả phần sau mỗi dấu chấm và thử kiểm tra TLD
        parts = s.split('.')
        for i in range(len(parts)):
            potential_tld = '.'.join(parts[i:])
            if potential_tld in self.tld_set:
                return True

        return False

    def decode_base64(self, encoded_str):
        """
        Attempt to decode a base64 encoded string
        """
        try:
            # Check if string is likely base64
            if not re.match(r'^[A-Za-z0-9+/=]+$', encoded_str):
                return encoded_str

            # Decode the base64 content
            decoded = base64.b64decode(encoded_str)

            # Try to decode as UTF-8
            try:
                return decoded.decode('utf-8', errors='replace')
            except UnicodeDecodeError:
                # If it can't be decoded as text, return None
                return encoded_str
        except:
            return encoded_str

    def process_tokenize(self, values):
        all_tokens = set()
        for value in values:
            # print(f"value: {value}")
            decoded = unquote(str(value))
            decoded = decoded.replace("--", " ")
            decoded = decoded.replace("_", " ")
            decoded = decoded.replace("'", "")
            decoded = decoded.replace("[", "")
            decoded = decoded.replace("]", "")
            # print(f"Decoded URL: {decoded}")
            tokens = self.tokenizer.tokenize(decoded)
            print("Tokens before clean:", tokens)
            tokens = [token.lower() for token in tokens]
            tokens = self.clean_tokens(tokens)
            all_tokens.update(tokens)
            print("Tokens after clean:", tokens)

        for token in all_tokens:
            if token not in self.normal_tokens:
                self.normal_tokens.add(token)
            self.word_counter[token] += 1

    def process_tokenize_cookie(self, values):
        alltokens = []
        for value in values:
            # print(f"Attack Tag: {value}")
            parts = re.split(r'(;)', str(value))
            for part in parts:
                # Nếu part chứa dấu "=" và dấu "=" nằm trước nửa chiều dài của part
                if "=" in part:
                    eq_index = part.find("=")
                    if eq_index < (len(part) / 2):
                        # Tách part thành 3 phần: phần trước dấu "=", dấu "=" và phần sau dấu "="
                        first_part, remainder = part.split("=", 1)
                        subparts = [first_part, "=", remainder]
                    else:
                        subparts = [part]
                else:
                    subparts = [part]
                for i, subpart in enumerate(subparts):
                    # print(f"Part: {subpart}")
                    decoded = unquote(str(subpart))
                    # Chỉ thực hiện decode base64 đối với phần remainder (index 2)
                    if i == 2:
                        decoded = self.decode_base64(decoded)
                    decoded = decoded.replace("--", " ")
                    decoded = decoded.replace("_", " ")
                    decoded = decoded.replace("'", "")
                    decoded = decoded.replace("[", "")
                    decoded = decoded.replace("]", "")
                    print(f"Decoded URL: {decoded}")
                    tokens = self.tokenizer.tokenize(decoded)
                    print("Tokens before clean:", tokens)
                    cleantokens = [token.lower() for token in tokens]
                    cleantokens = self.clean_tokens(cleantokens)
                    print("Tokens after clean:", cleantokens)
                    alltokens.extend(tokens)
            print("All tokens:", alltokens)
        for token in alltokens:
            if token not in self.normal_tokens:
                self.normal_tokens.add(token)
            self.word_counter[token] += 1

    def clean_tokens(self, tokens):
        """Add a token to the normal tokens set and increment its counter.

        Args:
            token: Token string to add
        """
        new_tokens = []
        for i, token in enumerate(tokens):
            if re.fullmatch(r"\d+x\d+", token):
                token = "<res>"
            elif re.fullmatch(r"(?:\d{1,3}\.){3}\d{1,3}", token):
                token = ("<ip>")
            elif re.fullmatch(r"[\d\.]+", token):
                token = "<num>"
            elif re.fullmatch(r"\d{2}:\d{2}:\d{2}", token):
                token = "<time>"
            elif token.lstrip('-').isdigit():
                token = "<num>"
            elif self.is_file_path(token):
                token = "<file>"
            elif self.is_domain(token):
                token = "<domain>"
            elif token == '../' or token == '..\\':
                new_tokens.append(token)
                continue
            elif self.is_random_string(token):
                token = "<random>"
            elif '.' in token and not self.is_file_path(token) and not self.is_domain(token):
                token = "<random>"
            new_tokens.append(token)
        return new_tokens


# vgen = VocabGen()

# vgen.run()
