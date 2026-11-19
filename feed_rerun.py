# -*- coding: utf-8 -*-
"""
FEED rerun: T5-small vs GPT-2-small feedback generation, evaluated correctly.

What this fixes (reviewer comments):
  * GPT-2 is trained on Q + A + FEEDBACK with the loss ONLY on the feedback tokens.
  * One fixed train/val/test split is made once, saved, and used for everything.
  * Metrics are computed ONLY on newly generated feedback vs the reference feedback of HELD-OUT test rows.
  * Both models: same data, same optimizer/lr/epochs/batch size, same decoding settings.
  * Several seeds + bootstrap confidence intervals + paired bootstrap for the T5-vs-GPT-2 difference.
  * Baselines: copy-the-student-answer, TF-IDF nearest-neighbour retrieval, zero-shot models.
  * Two split modes: 'random' (familiar questions) and 'question' (unseen questions).
  * Blinded human-evaluation sheet is exported.

USAGE (Colab, GPU runtime):   !pip -q install transformers sacrebleu rouge-score bert-score nltk scikit-learn
                              then run this file / the notebook cells.   Set PRESET below.
"""
import os, json, re, time, random, math, copy, sys
import numpy as np, pandas as pd

# =============================== CONFIG ===============================
ON_KAGGLE  = os.path.exists('/kaggle/input')
DATA_PATH  = os.environ.get('FEED_DATA', '/content/drive/MyDrive/OS-CUM.csv')   # the 134k file
if ON_KAGGLE and not os.path.exists(DATA_PATH) and os.environ.get('FEED_SOURCE','auto')=='local':          # auto-find the file in any attached Kaggle dataset
    import glob as _g
    _hits = _g.glob('/kaggle/input/**/OS-CUM*.csv', recursive=True) or _g.glob('/kaggle/input/**/*.csv', recursive=True)
    assert _hits, 'No CSV found in /kaggle/input: attach your OS-CUM.csv dataset (Add Input).'
    DATA_PATH = _hits[0]; print('Using data file:', DATA_PATH)
OUT_DIR    = os.environ.get('FEED_OUT', '/kaggle/working/feed_rerun_out' if ON_KAGGLE else '/content/feed_rerun_out')
PRESET     = os.environ.get('FEED_PRESET', 'quick')    # 'quick' = cheap sanity run, 'full' = paper-grade run
PRESETS = {
  #            rows used  epochs seeds    train fractions        test rows generated  split modes
  'quick': dict(max_rows=30000,  epochs=2, seeds=[42],          fracs=[1.0],           n_test=1000, modes=['random']),
  'full' : dict(max_rows=None,   epochs=3, seeds=[42, 43, 44],  fracs=[0.25, 0.5, 1.0], n_test=3000, modes=['random', 'question']),
}
PRESETS['paper_random']   = dict(max_rows=None, epochs=3, seeds=[42, 43, 44], fracs=[1.0], n_test=2000, modes=['random'])
PRESETS['paper_question'] = dict(max_rows=None, epochs=3, seeds=[42, 43, 44], fracs=[1.0], n_test=2000, modes=['question'])
PRESETS['curve']          = dict(max_rows=None, epochs=3, seeds=[42],         fracs=[0.1, 0.25, 0.5, 1.0], n_test=2000, modes=['random'])
P = PRESETS[PRESET]
MAX_ROWS, EPOCHS, SEEDS, FRACS, N_TEST, MODES = P['max_rows'], P['epochs'], P['seeds'], P['fracs'], P['n_test'], P['modes']
LR, BATCH, WARMUP = 1e-4, 16, 0.06            # identical for both models
MAX_SRC, MAX_TGT, GEN_NEW = 256, 96, 96       # input tokens, target tokens, generated tokens (same for both)
DECODE = dict(num_beams=4, no_repeat_ngram_size=3, early_stopping=True)   # identical decoding for both
N_BOOT = 1000
os.makedirs(OUT_DIR, exist_ok=True)

# =============================== DATA ===============================
GH_RAW = 'https://raw.githubusercontent.com/RAMESHDADI/FEED/main/'
BAD_FB = re.compile(r"not aligned|questions? and responses?|seems like the (?:question|response)|appears (?:that )?the (?:question|response)", re.I)

def _fetch(name):
    """local clone dir (FEED_REPO_DIR) or download from GitHub raw into OUT_DIR."""
    import urllib.request, urllib.parse
    rd = os.environ.get('FEED_REPO_DIR')
    if rd: return os.path.join(rd, name)
    dst = os.path.join(OUT_DIR, 'gh_' + name.replace('/', '_'))
    if not os.path.exists(dst):
        urllib.request.urlretrieve(GH_RAW + urllib.parse.quote(name), dst)
    return dst

def load_from_github():
    """1) OS-CUM.csv if it is in the repo (the 134k file);  2) otherwise rebuild from the repo's Excel files."""
    for nm in ['OS-CUM.csv', 'OS-CUM.xlsx']:
        try:
            p = _fetch(nm); print('Using', nm, 'from GitHub')
            return (pd.read_csv(p, low_memory=False) if nm.endswith('csv') else pd.read_excel(p)), 'github:' + nm
        except Exception as e:
            print(nm, 'not usable (', type(e).__name__, ')')
    raise FileNotFoundError('OS-CUM.csv / OS-CUM.xlsx not found in the repo. Only the student-created OS-CUM file is used for training; the other (GPT-generated) files are deliberately NOT used.')

def get_raw():
    src = os.environ.get('FEED_SOURCE', 'auto')          # 'github' | 'local' | 'auto'
    if src == 'local' or (src == 'auto' and os.path.exists(DATA_PATH)):
        return pd.read_csv(DATA_PATH, low_memory=False), 'local:' + DATA_PATH
    return load_from_github()

def load_data(path=DATA_PATH, max_rows=MAX_ROWS, seed=0):
    df, source = get_raw()
    df.columns = [str(c).strip().upper() for c in df.columns]
    need = ['PROMPT', 'RESPONSE', 'FEEDBACK']
    miss = [c for c in need if c not in df.columns]
    assert not miss, f'missing columns {miss}; found {list(df.columns)}'
    keep = [c for c in ['ID', 'PROMPT', 'RESPONSE', 'MARKS', 'FEEDBACK'] if c in df.columns]
    df = df[keep].copy()
    n0 = len(df)
    df = df.dropna(subset=need)
    for c in need: df[c] = df[c].astype(str).str.strip()
    df = df[(df.PROMPT != '') & (df.RESPONSE != '') & (df.FEEDBACK != '') & (df.FEEDBACK.str.lower() != 'nan')
            & (df.RESPONSE.str.lower() != 'nan')]
    n_bad = int(df.FEEDBACK.str.contains(BAD_FB).sum()); df = df[~df.FEEDBACK.str.contains(BAD_FB)]   # LLM 'questions and answers are not aligned' messages
    n1 = len(df)
    df = df.drop_duplicates(subset=need).reset_index(drop=True)
    stats = dict(source=source, feedback_removed_as_llm_failure_message=n_bad, rows_raw=n0, rows_after_cleaning=n1, rows_after_dedup=len(df),
                 unique_prompts=int(df.PROMPT.nunique()), unique_responses=int(df.RESPONSE.nunique()))
    if max_rows and len(df) > max_rows:       # sample whole rows (keeps all questions represented)
        df = df.sample(n=max_rows, random_state=seed).reset_index(drop=True)
    df['ROW_ID'] = np.arange(len(df))
    stats['rows_used'] = len(df)
    with open(f'{OUT_DIR}/data_stats.json', 'w') as fh: json.dump(stats, fh, indent=1)
    print('DATA', stats)
    return df

def make_split(df, mode, seed=0):
    """80/10/10. mode='random': rows shuffled. mode='question': whole questions held out."""
    rng = np.random.RandomState(seed)
    if mode == 'random':
        idx = rng.permutation(len(df)); a, b = int(.8*len(df)), int(.9*len(df))
        lab = np.empty(len(df), dtype=object); lab[idx[:a]] = 'train'; lab[idx[a:b]] = 'val'; lab[idx[b:]] = 'test'
    elif mode == 'question':
        qs = list(df.PROMPT.unique()); rng.shuffle(qs); a, b = int(.8*len(qs)), int(.9*len(qs))
        m = {q: ('train' if i < a else 'val' if i < b else 'test') for i, q in enumerate(qs)}
        lab = df.PROMPT.map(m).values
    else: raise ValueError(mode)
    out = df.copy(); out['SPLIT'] = lab
    tr = set(zip(out[out.SPLIT=='train'].PROMPT, out[out.SPLIT=='train'].RESPONSE))
    te = out[out.SPLIT == 'test']
    overlap = float(np.mean([(p, r) in tr for p, r in zip(te.PROMPT, te.RESPONSE)])) if len(te) else 0.0
    info = dict(mode=mode, n_train=int((out.SPLIT=='train').sum()), n_val=int((out.SPLIT=='val').sum()),
                n_test=int((out.SPLIT=='test').sum()),
                test_questions_seen_in_train=float(te.PROMPT.isin(set(out[out.SPLIT=='train'].PROMPT)).mean()) if len(te) else 0,
                test_pairs_identical_to_train=overlap)
    print('SPLIT', info)
    out[['ROW_ID'] + (['ID'] if 'ID' in out.columns else []) + ['SPLIT']].to_csv(f'{OUT_DIR}/split_{mode}.csv', index=False)
    json.dump(info, open(f'{OUT_DIR}/split_{mode}_info.json', 'w'), indent=1)
    return out, info

# =============================== METRICS ===============================
def _tok(s): return re.findall(r"\w+", s.lower())

def per_example_metrics(preds, refs, bert=True, meteor=True):
    """returns DataFrame of per-example scores (generated vs reference ONLY)."""
    from rouge_score import rouge_scorer
    import sacrebleu
    sc = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rows = []
    for p, r in zip(preds, refs):
        s = sc.score(r, p)                      # (target, prediction)
        d = {}
        for k in ['rouge1', 'rouge2', 'rougeL']:
            d[k+'_p'], d[k+'_r'], d[k+'_f'] = s[k].precision, s[k].recall, s[k].fmeasure
        d['bleu'] = sacrebleu.sentence_bleu(p, [r]).score / 100.0
        d['len_pred'] = len(_tok(p)); d['len_ref'] = len(_tok(r))
        rows.append(d)
    df = pd.DataFrame(rows)
    if meteor:
        try:
            import nltk
            from nltk.translate.meteor_score import meteor_score
            for pkg in ['wordnet', 'omw-1.4']:
                try: nltk.data.find('corpora/' + pkg)
                except LookupError: nltk.download(pkg, quiet=True)
            df['meteor'] = [meteor_score([_tok(r)], _tok(p)) if p.strip() else 0.0 for p, r in zip(preds, refs)]
        except Exception as e: print('METEOR skipped:', e)
    if bert:
        try:
            from bert_score import score as bs
            _, _, F = bs(list(preds), list(refs), lang='en', verbose=False,
                         device='cuda' if _cuda() else 'cpu', batch_size=64)
            df['bertscore_f1'] = F.numpy()
        except Exception as e: print('BERTScore skipped:', e)
    return df

def _cuda():
    try:
        import torch; return torch.cuda.is_available()
    except Exception: return False

def corpus_bleu(preds, refs):
    import sacrebleu
    return sacrebleu.corpus_bleu(list(preds), [list(refs)]).score / 100.0

def boot_ci(x, n=N_BOOT, seed=0):
    x = np.asarray(x, float); rng = np.random.RandomState(seed)
    m = x[rng.randint(0, len(x), (n, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))

def paired_boot(a, b, n=N_BOOT, seed=0):
    d = np.asarray(a, float) - np.asarray(b, float); rng = np.random.RandomState(seed)
    m = d[rng.randint(0, len(d), (n, len(d)))].mean(1)
    lo, hi = np.percentile(m, [2.5, 97.5])
    return dict(mean_diff=float(d.mean()), ci_lo=float(lo), ci_hi=float(hi), excludes_zero=bool(lo > 0 or hi < 0))

# =============================== BASELINES ===============================
def baseline_copy(test): return list(test.RESPONSE)            # echoes the student's answer

def baseline_retrieval(train, test):
    """TF-IDF nearest training (question+answer) -> its feedback. Same-question neighbours preferred if available."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    Xtr = vec.fit_transform((train.PROMPT + ' ' + train.RESPONSE).values)
    Xte = vec.transform((test.PROMPT + ' ' + test.RESPONSE).values)
    fb = train.FEEDBACK.values; pr = train.PROMPT.values
    byq = {}
    for i, q in enumerate(pr): byq.setdefault(q, []).append(i)
    out = []
    for j in range(Xte.shape[0]):
        cand = byq.get(test.PROMPT.iloc[j])
        if cand:
            sims = (Xtr[cand] @ Xte[j].T).toarray().ravel(); out.append(fb[cand[int(sims.argmax())]])
        else:
            sims = (Xtr @ Xte[j].T).toarray().ravel(); out.append(fb[int(sims.argmax())])
    return out

# =============================== MODELS ===============================
def seed_all(s):
    import torch
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

class Fmt:
    """builds the model inputs; GPT-2 loss is masked on the prompt part."""
    def __init__(self, kind):
        from transformers import AutoTokenizer
        self.kind = kind
        if kind == 't5':
            self.tok = AutoTokenizer.from_pretrained('t5-small')
        else:
            self.tok = AutoTokenizer.from_pretrained('gpt2'); self.tok.pad_token = self.tok.eos_token
            self.tok.padding_side = 'left'
    def t5_src(self, q, r): return f'give feedback: question: {q} answer: {r}'
    def gpt_prompt_ids(self, q, r):
        qi = self.tok.encode('Question: ' + q, add_special_tokens=False)[:80]
        ri = self.tok.encode('\nAnswer: ' + r, add_special_tokens=False)[:MAX_SRC - len(qi) - 6]
        fi = self.tok.encode('\nFeedback:', add_special_tokens=False)
        return qi + ri + fi
    def encode_train(self, q, r, f):
        if self.kind == 't5':
            x = self.tok(self.t5_src(q, r), max_length=MAX_SRC, truncation=True)
            y = self.tok(f, max_length=MAX_TGT, truncation=True)
            return dict(input_ids=x['input_ids'], attention_mask=x['attention_mask'], labels=y['input_ids'])
        p = self.gpt_prompt_ids(q, r)
        t = self.tok.encode(' ' + f, add_special_tokens=False)[:MAX_TGT - 1] + [self.tok.eos_token_id]
        ids = p + t
        return dict(input_ids=ids, attention_mask=[1]*len(ids), labels=[-100]*len(p) + t)   # loss ONLY on feedback tokens

def collate_factory(tok, kind):
    import torch
    def collate(batch):
        L = max(len(b['input_ids']) for b in batch)
        T = max(len(b['labels']) for b in batch)
        pad = tok.pad_token_id
        ii = torch.tensor([b['input_ids'] + [pad]*(L-len(b['input_ids'])) for b in batch])
        am = torch.tensor([b['attention_mask'] + [0]*(L-len(b['attention_mask'])) for b in batch])
        if kind == 't5':
            lab = torch.tensor([b['labels'] + [-100]*(T-len(b['labels'])) for b in batch])
        else:
            lab = torch.tensor([b['labels'] + [-100]*(L-len(b['labels'])) for b in batch])
        return dict(input_ids=ii, attention_mask=am, labels=lab)
    return collate

def build_model(kind):
    from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM
    return AutoModelForSeq2SeqLM.from_pretrained('t5-small') if kind == 't5' else AutoModelForCausalLM.from_pretrained('gpt2')

def n_params(m): return sum(p.numel() for p in m.parameters())

def train_model(kind, tr, va, seed, tag):
    import torch
    from torch.utils.data import DataLoader
    from transformers import get_linear_schedule_with_warmup
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed_all(seed); fm = Fmt(kind); model = build_model(kind).to(dev)
    enc = lambda d: [fm.encode_train(q, r, f) for q, r, f in zip(d.PROMPT, d.RESPONSE, d.FEEDBACK)]
    col = collate_factory(fm.tok, kind)
    dl = DataLoader(enc(tr), batch_size=BATCH, shuffle=True, collate_fn=col)
    dv = DataLoader(enc(va), batch_size=BATCH * 2, shuffle=False, collate_fn=col)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps = EPOCHS * len(dl); sch = get_linear_schedule_with_warmup(opt, int(WARMUP * steps), steps)
    use_amp = dev == 'cuda' and kind == 'gpt2'       # T5 is unstable in fp16 -> fp32
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    log, best, best_sd = [], 1e9, None
    for ep in range(EPOCHS):
        model.train(); tl = []; t0 = time.time()
        for b in dl:
            b = {k: v.to(dev) for k, v in b.items()}
            with torch.autocast('cuda', dtype=torch.float16, enabled=use_amp):
                loss = model(**b).loss
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); sch.step()
            tl.append(loss.item())
        model.eval(); vl = []
        with torch.no_grad():
            for b in dv:
                b = {k: v.to(dev) for k, v in b.items()}; vl.append(model(**b).loss.item())
        rec = dict(tag=tag, kind=kind, seed=seed, epoch=ep + 1, train_loss=float(np.mean(tl)), val_loss=float(np.mean(vl)),
                   minutes=(time.time() - t0) / 60)
        log.append(rec); print(rec)
        if rec['val_loss'] < best: best = rec['val_loss']; best_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_sd); model.eval()
    return model, fm, log

def generate(model, fm, test, bs=16):
    import torch
    dev = next(model.parameters()).device; out = []
    rows = list(zip(test.PROMPT, test.RESPONSE))
    for i in range(0, len(rows), bs):
        chunk = rows[i:i + bs]
        if fm.kind == 't5':
            x = fm.tok([fm.t5_src(q, r) for q, r in chunk], max_length=MAX_SRC, truncation=True, padding=True, return_tensors='pt').to(dev)
            with torch.no_grad(): y = model.generate(**x, max_new_tokens=GEN_NEW, **DECODE)
            out += fm.tok.batch_decode(y, skip_special_tokens=True)
        else:
            ids = [fm.gpt_prompt_ids(q, r) for q, r in chunk]; L = max(map(len, ids)); pad = fm.tok.pad_token_id
            ii = torch.tensor([[pad]*(L-len(s)) + s for s in ids]).to(dev)
            am = torch.tensor([[0]*(L-len(s)) + [1]*len(s) for s in ids]).to(dev)
            with torch.no_grad(): y = model.generate(input_ids=ii, attention_mask=am, max_new_tokens=GEN_NEW,
                                                     pad_token_id=pad, **DECODE)
            out += fm.tok.batch_decode(y[:, L:], skip_special_tokens=True)     # ONLY the newly generated tokens
    return [o.strip() for o in out]

# =============================== MAIN ===============================
def summarize(name, pe):
    row = {'system': name, 'n': len(pe)}
    for c in pe.columns:
        if c in ('len_pred', 'len_ref'): row[c] = float(pe[c].mean()); continue
        m, lo, hi = boot_ci(pe[c].values); row[c] = m; row[c + '_lo'] = lo; row[c + '_hi'] = hi
    return row

def run():
    df = load_data()
    all_rows, all_logs, pairs = [], [], []
    for mode in MODES:
        sp, info = make_split(df, mode)
        tr, va, te = sp[sp.SPLIT == 'train'], sp[sp.SPLIT == 'val'], sp[sp.SPLIT == 'test']
        te = te.sample(n=min(N_TEST, len(te)), random_state=0).reset_index(drop=True)   # same test rows for every system
        refs = list(te.FEEDBACK)
        preds_store = {}
        # ---- baselines (no training) ----
        for nm, pr in [('copy_student_answer', baseline_copy(te)), ('retrieval_tfidf', baseline_retrieval(tr, te))]:
            pe = per_example_metrics(pr, refs); pe.to_csv(f'{OUT_DIR}/perex_{mode}_{nm}.csv', index=False)
            all_rows.append(dict(mode=mode, seed='-', frac=1.0, **summarize(nm, pe))); preds_store[nm] = pr
        # ---- zero-shot ----
        sub = te.iloc[:min(300, len(te))]
        for kind in ['t5', 'gpt2']:
            seed_all(0); fm = Fmt(kind); m = build_model(kind).to('cuda' if _cuda() else 'cpu').eval()
            pr = generate(m, fm, sub); pe = per_example_metrics(pr, list(sub.FEEDBACK))
            all_rows.append(dict(mode=mode, seed='-', frac=0.0, **summarize(f'{kind}_zeroshot', pe))); del m
        # ---- fine-tuned, several seeds / data fractions ----
        for frac in FRACS:
            for seed in (SEEDS if frac == 1.0 else SEEDS[:1]):
                trs = tr.sample(frac=frac, random_state=seed) if frac < 1 else tr
                for kind in ['t5', 'gpt2']:
                    tag = f'{mode}_{kind}_f{frac}_s{seed}'
                    model, fm, log = train_model(kind, trs, va, seed, tag); all_logs += log
                    pr = generate(model, fm, te); pe = per_example_metrics(pr, refs)
                    pe.to_csv(f'{OUT_DIR}/perex_{tag}.csv', index=False)
                    pd.DataFrame({'ROW_ID': te.ROW_ID, 'PROMPT': te.PROMPT, 'RESPONSE': te.RESPONSE,
                                  'REFERENCE': refs, 'GENERATED': pr}).to_csv(f'{OUT_DIR}/predictions_{tag}.csv', index=False)
                    r = dict(mode=mode, seed=seed, frac=frac, params=n_params(model),
                             corpus_bleu=corpus_bleu(pr, refs), **summarize(f'{kind}_finetuned', pe))
                    all_rows.append(r); preds_store[f'{kind}_f{frac}_s{seed}'] = (pr, pe)
                    del model; 
                    try:
                        import torch; torch.cuda.empty_cache()
                    except Exception: pass
                    pd.DataFrame(all_rows).to_csv(f'{OUT_DIR}/results_all.csv', index=False)   # save progress
                    pd.DataFrame(all_logs).to_csv(f'{OUT_DIR}/training_log.csv', index=False)
            # paired comparison T5 vs GPT-2 at this fraction (first seed)
            s0 = SEEDS[0]
            if f't5_f{frac}_s{s0}' in preds_store:
                a, b = preds_store[f't5_f{frac}_s{s0}'][1], preds_store[f'gpt2_f{frac}_s{s0}'][1]
                for c in [c for c in a.columns if c not in ('len_pred', 'len_ref')]:
                    pairs.append(dict(mode=mode, frac=frac, metric=c, **paired_boot(a[c], b[c])))   # T5 - GPT-2
        # ---- blinded human-evaluation sheet (100 items, T5 vs GPT-2 shuffled) ----
        s0 = SEEDS[0]; k = f'_f1.0_s{s0}'
        if True:
            try:
                t5p = pd.read_csv(f'{OUT_DIR}/predictions_{mode}_t5_f1.0_s{s0}.csv'); gp = pd.read_csv(f'{OUT_DIR}/predictions_{mode}_gpt2_f1.0_s{s0}.csv')
                h = t5p.sample(n=min(100, len(t5p)), random_state=1); rng = np.random.RandomState(2); sheet, key = [], []
                for i, r in h.iterrows():
                    flip = rng.rand() < .5; a_, b_ = (r.GENERATED, gp.loc[i, 'GENERATED']); a_, b_ = (b_, a_) if flip else (a_, b_)
                    sheet.append(dict(item=int(r.ROW_ID), question=r.PROMPT, student_answer=r.RESPONSE, reference_feedback=r.REFERENCE,
                                      feedback_A=a_, feedback_B=b_, A_correct_1to5='', A_useful_1to5='', B_correct_1to5='', B_useful_1to5=''))
                    key.append(dict(item=int(r.ROW_ID), A_is='gpt2' if flip else 't5'))
                pd.DataFrame(sheet).to_csv(f'{OUT_DIR}/human_eval_sheet_{mode}.csv', index=False)
                pd.DataFrame(key).to_csv(f'{OUT_DIR}/human_eval_KEY_{mode}.csv', index=False)
            except Exception as e: print('human sheet skipped', e)
    R = pd.DataFrame(all_rows); R.to_csv(f'{OUT_DIR}/results_all.csv', index=False)
    pd.DataFrame(pairs).to_csv(f'{OUT_DIR}/paired_t5_minus_gpt2.csv', index=False)
    # compact table for the paper: mean over seeds of the fine-tuned models, full data
    ft = R[(R.system.str.contains('finetuned')) & (R.frac == 1.0)]
    cols = [c for c in ['bleu', 'rouge1_p', 'rouge1_r', 'rouge1_f', 'rouge2_f', 'rougeL_f', 'meteor', 'bertscore_f1', 'len_pred'] if c in ft.columns]
    summ = ft.groupby(['mode', 'system'])[cols].agg(['mean', 'std']).round(4); summ.to_csv(f'{OUT_DIR}/table_main_mean_std_over_seeds.csv')
    print(summ); print('\nDONE ->', OUT_DIR)

if __name__ == '__main__':
    run()
