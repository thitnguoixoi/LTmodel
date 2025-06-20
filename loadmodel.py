from flask import Flask, request, jsonify
import json
import torch
from LTModel import LTModel
import HTokenizer

app = Flask(__name__)

# Label mapping
label_map = {
    0: "Normal",
    1: "Directory Traversal",
    2: "SQL Injection",
    3: "XSS",
    4: "Log Forging",
    5: "Cookie Injection",
    6: "RCE",
    7: "LOG4J"
}


def get_model():
    model = LTModel(
        vocab_size=len(vocab),
        hidden_size=128,
        num_layers=2,
        num_heads=8,
        ff_size=1024,       # MATCHED to checkpoint feed-forward layer size
        dropout=0.1,        # MATCHED to training default
        num_classes=8       # MATCHED to training (number of attack types)
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def tokenize_and_pad(json_data):
    tokens = tokenizer_instance.tokenize_json(json_data)
    if len(tokens) < MAX_SEQ_LENGTH:
        tokens += ["<pad>"] * (MAX_SEQ_LENGTH - len(tokens))
    else:
        tokens = tokens[:MAX_SEQ_LENGTH]
    return tokens


def convert_tokens_to_ids(tokens):
    return [vocab.get(token, vocab.get("<unk>", 0)) for token in tokens]


@app.route('/detect', methods=['POST'])
def detect():
    input_json = request.get_json()
    if not input_json:
        return jsonify({"error": "No JSON payload provided"}), 400

    tokens = tokenize_and_pad(input_json)
    token_ids = convert_tokens_to_ids(tokens)
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.get("logits", outputs)
        predicted_class = logits.argmax(dim=-1).item()

    return jsonify({
        "predicted_id": predicted_class,
        "attack_type": label_map.get(predicted_class, "Unknown")
    })


if __name__ == '__main__':
    with open('final_filtered_vocab.json', 'r', encoding='utf-8') as f:
        vocab = json.load(f)

    MAX_SEQ_LENGTH = 160
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(
        'final_model_complete.pt', map_location=device)

    tokenizer_instance = HTokenizer.HTokenizer()
    model = get_model()

    app.run(host='0.0.0.0', port=8888, debug=True)
