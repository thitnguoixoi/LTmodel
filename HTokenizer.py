from nltk.tokenize import RegexpTokenizer
import requests
import math
from collections import Counter
import re
import string
import base64
from urllib.parse import unquote


class HTokenizer:
    def __init__(self):
        """
        Initializes the HTokenizer with a specified regex pattern.

        :param pattern: A regex pattern to tokenize the text. Default is r'\w+' which matches words.
        """
        self.tld_list = self.load_tld_list()
        self.tokenizer = {
            "user-agent": RegexpTokenizer(r'[A-Za-z]+'),
            "date": RegexpTokenizer(r'\d{4}-\d{2}-\d{2}|\d{2}:\d{2}:\d{2}|[A-Za-z0-9]+'),
            "host": RegexpTokenizer(r'[A-Za-z0-9\.]+|:\\|\:'),
            "default": RegexpTokenizer(r'[a-zA-Z0-9]+'),
            "content-length": RegexpTokenizer(r'[0-9\.]+'),
            "attack": RegexpTokenizer(
                r'(?:ldap|rmi|ldaps|dns|iiop|nis|http|https|ftp|file)(?=[a-z])'
                r'|\.\./|\.\.\\|:\/\/'
                r'|[a-zA-Z0-9_.-]+\.(?:txt|csv|log|exe|py|php|html?|json|js|xml|conf|sh|bat|dll|zip|tar|gz|pdf|docx|doc|pptx|xlsx|ppt|xls|docm|dotx|dotm|potx|potm|ppsx|ppsm|pps|dot|jpg|jpeg|png|gif|bmp|tiff|tif|svg|webp|mp4|avi|mov|mkv|wmv|flv|mp3|wav|ogg|aac|wma|m4a|aspx)'
                r'|\(|\)|\;|\=|\:|\,|\&|\%|\\|\/|\+|\?|\#|\[|\]|\{|\}|\~'
                r'|\d{2}:\d{2}:\d{2}'
                r'|[A-Za-z0-9\.]+'
                r'|[A-Za-z0-9]+'
            )
        }

    def load_tld_list(self):
        url = "https://data.iana.org/TLD/tlds-alpha-by-domain.txt"
        response = requests.get(url)
        response.raise_for_status()
        lines = response.text.splitlines()
        tlds = set(line.lower() for line in lines if not line.startswith("#"))
        return tlds

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
            if potential_tld in self.tld_list:
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
            if re.fullmatch(r"[\d\.]+", token):
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

    def process_user(self, value):
        """
        Process user-agent strings to extract tokens.
        """
        # print(f"User-Agent: {value}")
        decoded = unquote(str(value))
        decoded = decoded.replace("--", " ")
        decoded = decoded.replace("_", " ")
        decoded = decoded.replace("'", "")
        decoded = decoded.replace("[", "")
        decoded = decoded.replace("]", "")
        # print(f"Decoded User-Agent: {decoded}")
        tokens = self.tokenizer["user-agent"].tokenize(decoded)
        # print("Tokens before clean:", tokens)
        tokens = [token.lower() for token in tokens]
        tokens = self.clean_tokens(tokens)
        # print("Tokens after clean:", tokens)
        return tokens

    def process_date(self, value):
        """
        Process user-agent strings to extract tokens.
        """
        # print(f"User-Agent: {value}")
        decoded = unquote(str(value))
        decoded = decoded.replace("--", " ")
        decoded = decoded.replace("_", " ")
        decoded = decoded.replace("'", "")
        decoded = decoded.replace("[", "")
        decoded = decoded.replace("]", "")
        # print(f"Decoded User-Agent: {decoded}")
        tokens = self.tokenizer["date"].tokenize(decoded)
        # print("Tokens before clean:", tokens)
        tokens = [token.lower() for token in tokens]
        tokens = self.clean_tokens(tokens)
        # print("Tokens after clean:", tokens)
        return tokens

    def process_host(self, value):
        """
        Process user-agent strings to extract tokens.
        """
        # print(f"User-Agent: {value}")
        decoded = unquote(str(value))
        decoded = decoded.replace("--", " ")
        decoded = decoded.replace("_", " ")
        decoded = decoded.replace("'", "")
        decoded = decoded.replace("[", "")
        decoded = decoded.replace("]", "")
        # print(f"Decoded User-Agent: {decoded}")
        tokens = self.tokenizer["host"].tokenize(decoded)
        # print("Tokens before clean:", tokens)
        tokens = [token.lower() for token in tokens]
        tokens = self.clean_tokens(tokens)
        # print("Tokens after clean:", tokens)
        return tokens

    def process_content_length(self, value):
        """
        Process content-length strings to extract tokens.
        """
        # print(f"Content-Length: {value}")
        decoded = unquote(str(value))
        decoded = decoded.replace("--", " ")
        decoded = decoded.replace("_", " ")
        decoded = decoded.replace("'", "")
        decoded = decoded.replace("[", "")
        decoded = decoded.replace("]", "")
        # print(f"Decoded Content-Length: {decoded}")
        tokens = self.tokenizer["content-length"].tokenize(decoded)
        # print("Tokens before clean:", tokens)
        tokens = [token.lower() for token in tokens]
        tokens = self.clean_tokens(tokens)
        # print("Tokens after clean:", tokens)
        return tokens

    def process_default(self, value):
        """
        Process user-agent strings to extract tokens.
        """
        # print(f"User-Agent: {value}")
        decoded = unquote(str(value))
        decoded = decoded.replace("--", " ")
        decoded = decoded.replace("_", " ")
        decoded = decoded.replace("'", "")
        decoded = decoded.replace("[", "")
        decoded = decoded.replace("]", "")
        # print(f"Decoded User-Agent: {decoded}")
        tokens = self.tokenizer["default"].tokenize(decoded)
        # print("Tokens before clean:", tokens)
        tokens = [token.lower() for token in tokens]
        tokens = self.clean_tokens(tokens)
        # print("Tokens after clean:", tokens)
        return tokens

    def process_attack(self, value):
        """
        Process user-agent strings to extract tokens.
        """
        # print(f"User-Agent: {value}")
        decoded = unquote(str(value))
        decoded = decoded.replace("--", " ")
        decoded = decoded.replace("_", " ")
        decoded = decoded.replace("'", "")
        decoded = decoded.replace("[", "")
        decoded = decoded.replace("]", "")
        # print(f"Decoded User-Agent: {decoded}")
        tokens = self.tokenizer["attack"].tokenize(decoded)
        # print("Tokens before clean:", tokens)
        tokens = [token.lower() for token in tokens]
        tokens = self.clean_tokens(tokens)
        # print("Tokens after clean:", tokens)
        return tokens

    def process_cookie(self, value):

        # print(f"Attack Tag: {value}")
        parts = re.split(r'(;)', str(value))
        alltokens = []
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
                # print(f"Decoded URL: {decoded}")
                tokens = self.tokenizer['attack'].tokenize(decoded)
                # print("Tokens before clean:", tokens)
                tokens = [token.lower() for token in tokens]
                tokens = self.clean_tokens(tokens)
                # print("Tokens after clean:", tokens)
                alltokens.extend(tokens)
        # print("All tokens:", alltokens)
        return alltokens

    def tokenize_df(self, row, columns,):
        """
        Tokenizes the input value and returns a list of tokens.
        """
        tokens = []
        for col in columns:
            lower_col = col.lower()
            if "cookie" in lower_col:
                tokens += self.process_cookie(row[col])
            elif "user-agent" in lower_col:
                tokens += self.process_user(row[col])
            elif "date" in lower_col:
                tokens += self.process_date(row[col])
            elif "host" in lower_col:
                tokens += self.process_host(row[col])
            elif "content-length" in lower_col:
                tokens += self.process_content_length(row[col])
            elif any(x in lower_col for x in ["sec-fetch-user", "accept", "sec-fetch-mode", "cache-control", "connection", "sec-ch-ua-mobile", "sec-ch-ua-platform"]):
                tokens += self.process_default(row[col])
            else:
                tokens += self.process_attack(row[col])
        return tokens

    def tokenize_json(self, json_data):
        """
        Tokenizes the input JSON data and returns a list of tokens.
        Processes the 'request' section similarly to how tokenize_df processes a DataFrame row.
        """
        tokens = []
        request = json_data.get("request", {})

        # First process the headers if available.
        headers = request.get("headers", {})
        if isinstance(headers, dict):
            for hkey, hvalue in headers.items():
                lower_hkey = hkey.lower()
                if "cookie" in lower_hkey:
                    tokens += self.process_cookie(hvalue)
                elif "user-agent" in lower_hkey:
                    tokens += self.process_user(hvalue)
                elif "date" in lower_hkey:
                    tokens += self.process_date(hvalue)
                elif "host" in lower_hkey:
                    tokens += self.process_host(hvalue)
                elif "content-length" in lower_hkey:
                    tokens += self.process_content_length(hvalue)
                elif any(x in lower_hkey for x in ["sec-fetch-user", "accept", "sec-fetch-mode", "cache-control", "connection", "sec-ch-ua-mobile", "sec-ch-ua-platform"]):
                    tokens += self.process_default(hvalue)
                else:
                    tokens += self.process_attack(hvalue)
        elif isinstance(headers, list):
            for item in headers:
                tokens += self.process_default(item)

        # Process the remaining keys in request (excluding headers)
        for key, value in request.items():
            if key == "headers":
                continue
            if isinstance(value, str):
                tokens += self.process_attack(value)
            elif isinstance(value, list):
                for item in value:
                    tokens += self.process_attack(item)

        return tokens
