import json, requests

def main():
    with open("eval/seed_qas.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            r = requests.post("http://localhost:8000/chat", json={"q": ex["q"], "mode": "short", "filters": ex.get("filters")})
            print("\nQ:", ex["q"])
            if r.ok:
                js = r.json()
                print("A:", js["answer"])
                print("Citations:", js["citations"])
            else:
                print("Error:", r.status_code, r.text)

if __name__ == "__main__":
    main()