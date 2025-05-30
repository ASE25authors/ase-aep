import requests
from transformers import pipeline
import os
import sys
import traceback
import gc
import csv
import shutil
from openpyxl import Workbook
import hashlib
import json
import threading
import time
import datetime
import psutil

try:
    import pynvml
    pynvml.nvmlInit()
except Exception:
    print("pynvml/nvidia-smi not available, cannot check VRAM.")
    sys.exit(2)

ENDPOINTS_FILE = 'endpoints.txt'
SCENARIOS_FILE = 'scenarios.txt'
OUTPUT_CSV = 'llm_raw_outputs_tmp.csv'
OUTPUT_FILE = 'llm_raw_outputs.xlsx'
ERROR_LOG = 'llm_run_errors.log'
OUTPUT_JSON = 'llm_raw_outputs_tmp.json'
BACKUP_DIR = 'backups'
BATCH_HEADER_LOG = 'llm_batches_done.log'
MIN_RAM_GB = 4
MIN_VRAM_GB = 8
API_TIMEOUT_MIN = 60
API_TIMEOUT_MAX = 300
THROTTLE_SEC = 2
FLUSH_BATCH = 10

lock = threading.Lock()

def vram_check():
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        vram_gb = meminfo.free / 1024**3
        if vram_gb < MIN_VRAM_GB:
            print(f"Insufficient VRAM: {vram_gb:.2f} GB free, required: {MIN_VRAM_GB} GB. Exiting.")
            sys.exit(3)
    except Exception:
        print("Unable to check VRAM. Exiting.")
        sys.exit(2)

def log_error(msg):
    with lock:
        with open(ERROR_LOG, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')

def md5sum(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()

def load_llms(path):
    llms = []
    with open(path, 'r', encoding='utf-8') as f:
        for lineno, line in enumerate(f, 1):
            if line.strip():
                items = line.strip().split('\t')
                if len(items) < 2:
                    log_error(f'Invalid entry at line {lineno}: {line}')
                    continue
                while len(items) < 4:
                    items.append("")
                name, typ, endpoint, key = items[:4]
                llms.append({'name': name, 'type': typ, 'endpoint': endpoint, 'key': key})
    return llms

def prompt_template(scenario, custom=None):
    if custom: return custom.format(scenario=scenario)
    return (
        f"Given the following scenario: {scenario} "
        "1) Which ethical theory best applies to this situation: utilitarianism, deontology, or virtue ethics? "
        "2) Based on the theory you selected, is the action morally acceptable (yes/no)? "
        "3) Provide a brief explanation."
    )

def parse_api_response(llm, rjson):
    if "openai.com" in llm['endpoint']:
        try: return rjson['choices'][0]['message']['content']
        except: return str(rjson)
    if "anthropic.com" in llm['endpoint']:
        content = rjson.get('content', rjson.get('completion', ""))
        if isinstance(content, list) and 'text' in content[0]: return content[0]['text']
        if isinstance(content, list): return str(content[0])
        return str(content)
    if "googleapis.com" in llm['endpoint'] and "gemini" in llm['endpoint']:
        cands = rjson.get('candidates', [])
        if cands and 'content' in cands[0]:
            cont = cands[0]['content']
            if isinstance(cont, dict) and 'parts' in cont and cont['parts']:
                return cont['parts'][0].get('text', str(cont))
            return str(cont)
        return str(cands)
    if "cohere.ai" in llm['endpoint']: return rjson.get('text', str(rjson))
    if "mistral.ai" in llm['endpoint']:
        choices = rjson.get('choices', [])
        if choices: return choices[0]['message'].get('content', str(choices[0]))
        return str(choices)
    if "openrouter.ai" in llm['endpoint']:
        choices = rjson.get('choices', [])
        if choices: return choices[0]['message'].get('content', str(choices[0]))
        return str(choices)
    if "x.ai" in llm['endpoint']:
        choices = rjson.get('choices', [])
        if choices and 'text' in choices[0]: return choices[0]['text']
        return str(choices)
    if "dashscope.aliyun.com" in llm['endpoint'] or "dashscope.aliyuncs.com" in llm['endpoint']:
        return rjson.get('output', {}).get('text', str(rjson))
    return str(rjson)

def get_timeout(scenario):
    words = len(scenario.split())
    timeout = min(max(API_TIMEOUT_MIN, words * 4), API_TIMEOUT_MAX)
    return timeout

def api_req_safe(fn, *args, **kwargs):
    for delay in (0, 10, 30):
        if delay: time.sleep(delay)
        try: return fn(*args, **kwargs)
        except Exception as e:
            log_error(f"API RETRY: {e}")
            continue
    return "API retry fail"

def call_api(llm, prompt, scenario):
    url = llm['endpoint']
    key = llm['key']
    timeout = get_timeout(scenario)
    time.sleep(THROTTLE_SEC)
    try:
        if "openai.com" in url:
            headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
            data = {"model": "gpt-4o", "messages": [{"role": "user", "content": prompt}], "max_tokens": 512}
            r = api_req_safe(requests.post, url, headers=headers, json=data, timeout=timeout)
            return parse_api_response(llm, r.json()) if hasattr(r, 'json') else str(r)
        if "anthropic.com" in url:
            headers = {"x-api-key": key, "anthropic-version": "2023-06-01", "Content-Type": "application/json"}
            data = {"model": "claude-3-sonnet-20240229", "max_tokens": 512, "messages": [{"role": "user", "content": prompt}]}
            r = api_req_safe(requests.post, url, headers=headers, json=data, timeout=timeout)
            return parse_api_response(llm, r.json()) if hasattr(r, 'json') else str(r)
        if "googleapis.com" in url and "gemini" in url:
            headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
            data = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
            r = api_req_safe(requests.post, url, headers=headers, json=data, timeout=timeout)
            return parse_api_response(llm, r.json()) if hasattr(r, 'json') else str(r)
        if "cohere.ai" in url:
            headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
            data = {"message": prompt, "model": "command-r-plus"}
            r = api_req_safe(requests.post, url, headers=headers, json=data, timeout=timeout)
            return parse_api_response(llm, r.json()) if hasattr(r, 'json') else str(r)
        if "mistral.ai" in url:
            headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
            data = {"model": "mistral-large-latest", "messages": [{"role": "user", "content": prompt}], "max_tokens": 512}
            r = api_req_safe(requests.post, url, headers=headers, json=data, timeout=timeout)
            return parse_api_response(llm, r.json()) if hasattr(r, 'json') else str(r)
        if "openrouter.ai" in url:
            headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
            data = {"model": llm['name'], "messages": [{"role": "user", "content": prompt}], "max_tokens": 512}
            r = api_req_safe(requests.post, url, headers=headers, json=data, timeout=timeout)
            return parse_api_response(llm, r.json()) if hasattr(r, 'json') else str(r)
        if "x.ai" in url:
            headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
            data = {"model": "grok-1", "prompt": prompt, "max_tokens": 512}
            r = api_req_safe(requests.post, url, headers=headers, json=data, timeout=timeout)
            return parse_api_response(llm, r.json()) if hasattr(r, 'json') else str(r)
        if "dashscope.aliyun.com" in url or "dashscope.aliyuncs.com" in url:
            headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
            data = {"model": "qwen-3-235b-a22b", "input": {"prompt": prompt}, "parameters": {"max_new_tokens": 512}}
            r = api_req_safe(requests.post, url, headers=headers, json=data, timeout=timeout)
            return parse_api_response(llm, r.json()) if hasattr(r, 'json') else str(r)
        return "[No API handler]"
    except Exception as e:
        log_error(f'API error for {llm["name"]}: {str(e)}\n{traceback.format_exc()}')
        return f"API error: {e}"

def try_pipeline(model_id, ptype, trust=False):
    try: return pipeline(ptype, model=model_id, tokenizer=model_id, trust_remote_code=trust)
    except: return None

def offline_sanity_check(model_id):
    ptypes = ['text2text-generation', 'text-generation']
    for ptype in ptypes:
        for trust in [False, True]:
            pl = try_pipeline(model_id, ptype, trust)
            if pl:
                try:
                    res = pl("Say 'OK'.", max_new_tokens=8)
                    if isinstance(res, list) and len(res) and isinstance(res[0], dict):
                        if 'generated_text' in res[0] or 'text' in res[0]:
                            del pl
                            gc.collect()
                            return ptype, trust
                except: pass
    log_error(f"Model offline sanity check failed for: {model_id}")
    return None, None

def check_mem(min_gb=MIN_RAM_GB):
    v = psutil.virtual_memory()
    if v.available < min_gb * 1024**3:
        gc.collect()
        time.sleep(2)
        v = psutil.virtual_memory()
        if v.available < min_gb * 1024**3:
            log_error(f"RAM critically low: {v.available // 1024**2}MB")
            print("RAM critically low, aborting.")
            sys.exit(4)

def call_offline(model_id, prompt, conf):
    check_mem()
    vram_check()
    ptype, trust = conf.get(model_id, (None, None))
    if not ptype:
        return "Offline error: Model unavailable"
    try:
        pl = try_pipeline(model_id, ptype, trust)
        result = pl(prompt, max_new_tokens=512)
        del pl
        gc.collect()
        if isinstance(result, list) and len(result) and isinstance(result[0], dict):
            if 'generated_text' in result[0]:
                return result[0]['generated_text']
            if 'text' in result[0]:
                return result[0]['text']
            return str(result[0])
        return str(result)
    except Exception as e:
        gc.collect()
        log_error(f'Offline error for {model_id}: {str(e)}\n{traceback.format_exc()}')
        return f"Offline error: {e}"

def file_check(path):
    if not os.path.isfile(path):
        print(f"Missing file: {path}")
        sys.exit(1)

def read_existing_csv(path):
    existing = set()
    if os.path.isfile(path):
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if row: existing.add(row[0])
    return existing

def batch_worker(args):
    scenario, llms, offline_conf, prompt_str = args
    row = [scenario]
    json_out = {'scenario': scenario}
    for llm in llms:
        prompt = prompt_template(scenario, prompt_str)
        try:
            if llm['type'] == 'api':
                resp = call_api(llm, prompt, scenario)
            else:
                resp = call_offline(llm['endpoint'], prompt, offline_conf)
            row.append(str(resp).replace('\n', ' ').replace('\r', ' ').strip())
            json_out[llm['name']] = resp
        except Exception as e:
            log_error(f"Fatal error in batch_worker: {e}\n{traceback.format_exc()}")
            row.append(f"Fatal error: {e}")
            json_out[llm['name']] = f"Fatal error: {e}"
    return row, json_out

def flush_csv_json(rows, json_rows, header, mode='a'):
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    with lock:
        write_header = not os.path.isfile(OUTPUT_CSV) or mode == 'w'
        with open(OUTPUT_CSV, mode, newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(header)
            for row in rows:
                writer.writerow(row)
        with open(OUTPUT_JSON, mode, encoding='utf-8') as jf:
            for item in json_rows:
                jf.write(json.dumps(item, ensure_ascii=False) + '\n')
        if not os.path.exists(BACKUP_DIR): os.makedirs(BACKUP_DIR)
        shutil.copyfile(OUTPUT_CSV, os.path.join(BACKUP_DIR, f"llm_csv_{ts}.bak"))
        shutil.copyfile(OUTPUT_JSON, os.path.join(BACKUP_DIR, f"llm_json_{ts}.bak"))

def csv_to_xlsx(csv_path, xlsx_path):
    wb = Workbook()
    ws = wb.active
    with open(csv_path, 'r', encoding='utf-8') as f:
        for row in csv.reader(f):
            ws.append(row)
    wb.save(xlsx_path)

def main():
    for fn in (ENDPOINTS_FILE, SCENARIOS_FILE):
        file_check(fn)
    vram_check()
    llms = load_llms(ENDPOINTS_FILE)
    with open(SCENARIOS_FILE, encoding='utf-8') as f:
        scenarios = [x.strip() for x in f if x.strip()]
    offline_llms = [l for l in llms if l['type'] != 'api']
    offline_conf = {}
    for l in offline_llms:
        print(f"Sanity checking offline model: {l['endpoint']}")
        ptype, trust = offline_sanity_check(l['endpoint'])
        offline_conf[l['endpoint']] = (ptype, trust)
    header = ["Scenario"] + [l['name'] for l in llms]
    done_scenarios = read_existing_csv(OUTPUT_CSV)
    prompt_str = None
    batch, batch_json = [], []
    try:
        for scenario in scenarios:
            if scenario in done_scenarios: continue
            batch.append((scenario, llms, offline_conf, prompt_str))
            if len(batch) >= FLUSH_BATCH:
                rows, json_rows = [], []
                for args in batch:
                    row, jrow = batch_worker(args)
                    rows.append(row)
                    json_rows.append(jrow)
                flush_csv_json(rows, json_rows, header)
                batch.clear()
                batch_json.clear()
                done_scenarios.update([r[0] for r in rows])
        if batch:
            rows, json_rows = [], []
            for args in batch:
                row, jrow = batch_worker(args)
                rows.append(row)
                json_rows.append(jrow)
            flush_csv_json(rows, json_rows, header)
    except KeyboardInterrupt:
        print("Batch interrupted by user. Flushed current batch.")
    csv_to_xlsx(OUTPUT_CSV, OUTPUT_FILE)
    shutil.copyfile(OUTPUT_CSV, OUTPUT_FILE.replace('.xlsx', '.csv'))
    print("MD5 CSV:", md5sum(OUTPUT_CSV))
    print("MD5 XLSX:", md5sum(OUTPUT_FILE))
    print("MD5 JSON:", md5sum(OUTPUT_JSON))

if __name__ == '__main__':
    main()
