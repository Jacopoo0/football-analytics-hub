"""Secret scanning script for pre-publication audit."""
import os, re

root = "."
ignore_dirs = {"venv", "__pycache__", ".git", ".pytest_cache", "node_modules"}
ignore_extensions = {".pyc", ".pyo", ".png", ".jpg", ".jpeg", ".parquet", ".gif", ".ico"}
ignore_files = {".env"}  # known secret file, gitignored

patterns = {
    "GROQ_API_KEY": r"gsk_[a-zA-Z0-9]{20,}",
    "OpenAI_API_KEY": r"sk-[a-zA-Z0-9]{20,}",
    "Email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
    "LocalPathWin": r"[A-Z]:\\\\Users\\\\",
    "LocalPathUnix": r"/home/|/Users/",
    "AWS Key": r"AKIA[0-9A-Z]{16}",
    "Bearer token": r"bearer\s+[a-zA-Z0-9\-_\.]{20,}",
    "Private RSA key": r"-----BEGIN (RSA |EC )?PRIVATE KEY-----",
}

found = []
for root_dir, dirs, files in os.walk(root):
    dirs[:] = [d for d in dirs if d not in ignore_dirs]
    for fname in files:
        fpath = os.path.join(root_dir, fname)
        ext = os.path.splitext(fname)[1]
        if ext in ignore_extensions or fname in ignore_files:
            continue
        try:
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except Exception:
            continue
        for pname, pattern in patterns.items():
            for match in re.finditer(pattern, content, re.IGNORECASE):
                found.append((pname, fpath, match.group()[:60]))

if found:
    print("SECURITY ISSUES FOUND:")
    for pname, fpath, match in found:
        masked = match[:8] + "..." + match[-4:] if len(match) > 20 else match
        print(f"  [{pname}] {fpath}: {masked}")
    print(f"\nTotal: {len(found)} issues")
else:
    print("No security issues found in source files.")

# Also check if .env contains real key
env_path = os.path.join(root, ".env")
if os.path.exists(env_path):
    with open(env_path) as f:
        env_content = f.read()
    real_keys = re.findall(r"gsk_[a-zA-Z0-9]{30,}", env_content)
    if real_keys:
        for k in real_keys:
            masked = k[:8] + "..." + k[-4:]
            print(f"\nWARNING: Real GROQ_API_KEY in .env: {masked}")
            print(f"  .env is gitignored, but ensure you DON'T accidentally add it to git.")
    placeholder_keys = re.findall(r"sk-[a-zA-Z0-9]{5,20}", env_content)
    if placeholder_keys:
        print(f"\nPlaceholder keys found in .env: {placeholder_keys}")
else:
    print("\nNo .env file found (safe).")

# Check git status
if not os.path.exists(os.path.join(root, ".git")):
    print("\nNOTE: No .git repository initialized yet.")
    print("  Run 'git init' then verify .gitignore excludes .env, data/raw/, reports/output/")
else:
    print("\nGit repository exists.")

print("\nSecret scan complete.")
