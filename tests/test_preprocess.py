import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from core.preprocess import clean_text, preprocess

def test_clean_text_removes_urls_and_emojis():
    raw = "AI 😊 is awesome! Visit https://example.com now!"
    cleaned = clean_text(raw)
    assert "http" not in cleaned
    assert "😊" not in cleaned
    assert "AI" in cleaned

def test_preprocess_generates_tokens():
    text = "Artificial intelligence is transforming industries."
    cleaned, tokens = preprocess(text)
    assert isinstance(tokens, list)
    assert len(tokens) > 2
    assert "artificial" in tokens
