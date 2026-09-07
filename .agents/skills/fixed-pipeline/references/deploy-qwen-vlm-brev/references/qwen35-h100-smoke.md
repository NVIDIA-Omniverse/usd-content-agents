# Qwen3.5 H100 Multi-Image Smoke

Use this after the H100 NIM path starts and before handing the endpoint to an
agent workload that sends many rendered views in one prompt.

Verify the container image limit:

```bash
ssh wu-vlm-h100-hs \
  'docker inspect qwen35b-nim --format "{{range .Config.Env}}{{println .}}{{end}}" | \
   grep NIM_MAX_IMAGES_PER_PROMPT'
```

Run a 20-image prompt against the remote localhost endpoint:

```bash
ssh wu-vlm-h100-hs 'python3 - <<'"'"'PY'"'"'
import json
import urllib.request

image = "data:image/gif;base64,R0lGODlhAQABAPAAAP8AAAAAACwAAAAAAQABAAACAkQBADs="
content = [{"type": "text", "text": "Answer exactly: twenty-images-ok"}]
content.extend({"type": "image_url", "image_url": {"url": image}} for _ in range(20))
payload = {
    "model": "qwen/qwen3.5-35b-a3b",
    "messages": [{"role": "user", "content": content}],
    "max_tokens": 12,
    "temperature": 0,
    "chat_template_kwargs": {"enable_thinking": False},
}
req = urllib.request.Request(
    "http://localhost:8000/v1/chat/completions",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(req, timeout=180) as resp:
    body = json.loads(resp.read().decode("utf-8"))
    print(resp.status)
    print(body["choices"][0]["message"]["content"].strip())
PY'
```
