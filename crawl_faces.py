import os
import glob
import requests
from tavily import TavilyClient
from concurrent.futures import ThreadPoolExecutor, as_completed

PEOPLE = ["小日向美香", "林鼓子", "立石凛", "羊宫妃那", "青木阳菜"]
NUM_IMAGES = 20
FACES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "faces")
TAVILY_KEY = "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"
BAIDU_URL = "https://qianfan.baidubce.com/v2/ai_search/web_search"
BAIDU_KEY = "Bearer bce-v3/ALTAK-9DFA6lV6DyUBQUHjkne2d/5532b6df103be0ef1f9f09c0dadb2289953322c6"


def download_image(url, save_path):
    try:
        r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code == 200 and len(r.content) > 5000:
            with open(save_path, "wb") as f:
                f.write(r.content)
            return True
    except Exception:
        pass
    return False


def tavily_search(name):
    client = TavilyClient(TAVILY_KEY)
    urls = []
    for query in [f"{name} 声優 写真", f"{name} 声優 画像"]:
        try:
            r = client.search(query=query, search_depth="advanced", include_images=True)
            urls.extend(r.get("images", []))
        except Exception as e:
            print(f"  Tavily error: {e}")
    return list(dict.fromkeys(urls))


def baidu_search(name):
    urls = []
    try:
        r = requests.post(
            BAIDU_URL,
            headers={"Content-Type": "application/json", "Authorization": BAIDU_KEY},
            json={"messages": [{"role": "user", "content": f"{name} 声優 图片 写真"}]},
            timeout=15,
        )
        data = r.json()
        for ref in data.get("references", []):
            for key in ["image", "icon"]:
                u = ref.get(key)
                if u and u.startswith("http"):
                    urls.append(u)
    except Exception as e:
        print(f"  Baidu error: {e}")
    return urls


def rename_sequential(person_dir):
    count = 1
    for f in sorted(glob.glob(os.path.join(person_dir, "*"))):
        ext = os.path.splitext(f)[1].lower()
        if ext not in (".jpg", ".jpeg", ".png", ".webp"):
            continue
        new_ext = ".jpg" if ext == ".webp" else ext
        new_name = os.path.join(person_dir, f"{count}{new_ext}")
        if f != new_name:
            os.rename(f, new_name)
        count += 1


def crawl_person(name):
    person_dir = os.path.join(FACES_DIR, name)
    os.makedirs(person_dir, exist_ok=True)
    existing = len(glob.glob(os.path.join(person_dir, "*")))
    print(f"[{name}] existing: {existing}, target: +{NUM_IMAGES}")

    urls = []
    urls.extend(tavily_search(name))
    urls.extend(baidu_search(name))
    urls = list(dict.fromkeys(urls))
    print(f"  Found {len(urls)} image URLs")

    downloaded = 0
    idx = existing + 1
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {}
        for url in urls:
            ext = ".jpg"
            if ".png" in url:
                ext = ".png"
            save_path = os.path.join(person_dir, f"{idx}{ext}")
            futures[executor.submit(download_image, url, save_path)] = save_path
            idx += 1

        for future in as_completed(futures):
            if future.result():
                downloaded += 1
            else:
                path = futures[future]
                if os.path.exists(path):
                    os.remove(path)

    rename_sequential(person_dir)
    total = len(glob.glob(os.path.join(person_dir, "*")))
    print(f"[{name}] downloaded: {downloaded}, total: {total}")


def main():
    for name in PEOPLE:
        crawl_person(name)


if __name__ == "__main__":
    main()
