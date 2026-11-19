# -*- coding: utf-8 -*-
"""
Audit + human-evaluation tools for the FEED paper.  Run AFTER feed_rerun.py (same folder).
 1) make_audit_sheet()      -> audit_sheet.csv  (200 reference-feedback items for 2 independent raters)
 2) analyze_audit(r1, r2)   -> error rate of the references + agreement (kappa, weighted kappa, alpha)
 3) analyze_human_eval(...) -> blinded T5-vs-GPT-2 ratings, un-blinded with the key, with CIs and agreement
"""
import numpy as np, pandas as pd, json, os

# ------------------------------------------------------------------ helpers
def wilson(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    p = k / n; d = 1 + z*z/n; c = p + z*z/(2*n); h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))
    return ((c - h)/d, (c + h)/d)

def cohen_kappa(a, b, weights=None):
    from sklearn.metrics import cohen_kappa_score
    return float(cohen_kappa_score(a, b, weights=weights))

def krippendorff_alpha(data, level='interval'):
    """data: raters x items array (np.nan = missing). level 'nominal' or 'interval'."""
    D = np.asarray(data, float); vals = np.unique(D[~np.isnan(D)])
    idx = {v: i for i, v in enumerate(vals)}; k = len(vals); o = np.zeros((k, k))
    for u in range(D.shape[1]):
        col = D[:, u]; col = col[~np.isnan(col)]; m = len(col)
        if m < 2: continue
        for i in range(m):
            for j in range(m):
                if i != j: o[idx[col[i]], idx[col[j]]] += 1.0 / (m - 1)
    n_c = o.sum(1); n = n_c.sum()
    if n <= 1: return np.nan
    delta = np.array([[0 if i == j else (1 if level == 'nominal' else (vals[i]-vals[j])**2) for j in range(k)] for i in range(k)], float)
    Do = (o * delta).sum() / n
    De = sum(n_c[i]*n_c[j]*delta[i, j] for i in range(k) for j in range(k)) / (n*(n-1))
    return float(1 - Do/De) if De > 0 else np.nan

def boot_mean_ci(x, n=2000, seed=0):
    x = np.asarray(x, float); rng = np.random.RandomState(seed)
    m = x[rng.randint(0, len(x), (n, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))

# ------------------------------------------------------------------ 1) audit sheet
def make_audit_sheet(n_items=200, out='audit_sheet.csv', seed=7):
    import feed_rerun as F
    df = F.load_data(max_rows=None)
    # spread over questions: at most 3 items per question, then fill randomly
    g = df.sample(frac=1, random_state=seed).groupby('PROMPT').head(3)
    s = g.sample(n=min(n_items, len(g)), random_state=seed).reset_index(drop=True)
    sheet = pd.DataFrame({'item_id': s.ROW_ID, 'question': s.PROMPT, 'student_answer': s.RESPONSE, 'reference_feedback': s.FEEDBACK,
                          'factually_correct_0or1': '', 'pedagogically_useful_1to5': '', 'problem_type': '', 'corrected_feedback_optional': ''})
    sheet.to_csv(out, index=False); print('wrote', out, len(sheet), 'items; give an identical copy to each rater, do NOT let them discuss')
    return sheet

# ------------------------------------------------------------------ 2) analyse audit
def analyze_audit(r1_csv, r2_csv, out_json='audit_results.json'):
    a = pd.read_csv(r1_csv).set_index('item_id'); b = pd.read_csv(r2_csv).set_index('item_id')
    ids = a.index.intersection(b.index)
    a, b = a.loc[ids], b.loc[ids]
    fa, fb = a['factually_correct_0or1'].astype(int), b['factually_correct_0or1'].astype(int)
    ua, ub = a['pedagogically_useful_1to5'].astype(int), b['pedagogically_useful_1to5'].astype(int)
    n = len(ids)
    err_any = int(((fa == 0) | (fb == 0)).sum()); err_both = int(((fa == 0) & (fb == 0)).sum())
    res = dict(n_items=n,
               factual_error_rate_either_rater=err_any / n, ci_either=wilson(err_any, n),
               factual_error_rate_both_raters=err_both / n, ci_both=wilson(err_both, n),
               kappa_factual_correct=cohen_kappa(fa, fb),
               percent_agreement_factual=float((fa == fb).mean()),
               qwk_usefulness=cohen_kappa(ua, ub, 'quadratic'),
               alpha_usefulness_interval=krippendorff_alpha(np.vstack([ua, ub]), 'interval'),
               mean_usefulness_r1=float(ua.mean()), mean_usefulness_r2=float(ub.mean()),
               share_useful_ge4_both=float(((ua >= 4) & (ub >= 4)).mean()))
    pt = pd.concat([a['problem_type'], b['problem_type']]).dropna().astype(str).str.strip().str.lower()
    res['problem_types'] = pt[pt != ''].value_counts().to_dict()
    json.dump(res, open(out_json, 'w'), indent=1, default=float); print(json.dumps(res, indent=1, default=float)); return res

# ------------------------------------------------------------------ 3) analyse blinded human evaluation
def analyze_human_eval(rater_csvs, key_csv, out_csv='human_eval_summary.csv'):
    key = pd.read_csv(key_csv).set_index('item'); long = []
    for ri, f in enumerate(rater_csvs, 1):
        d = pd.read_csv(f)
        for _, r in d.iterrows():
            a_is = key.loc[r['item'], 'A_is']; b_is = 'gpt2' if a_is == 't5' else 't5'
            for side, sysn in (('A', a_is), ('B', b_is)):
                long.append(dict(rater=ri, item=r['item'], system=sysn, correct=r[f'{side}_correct_1to5'], useful=r[f'{side}_useful_1to5']))
    L = pd.DataFrame(long).dropna(); L[['correct', 'useful']] = L[['correct', 'useful']].astype(float)
    rows = []
    for m in ['correct', 'useful']:
        piv = L.groupby(['item', 'system'])[m].mean().unstack()               # mean over raters per item
        for s in ['t5', 'gpt2']:
            mu, lo, hi = boot_mean_ci(piv[s].dropna()); rows.append(dict(metric=m, system=s, mean=mu, ci_lo=lo, ci_hi=hi, n_items=int(piv[s].notna().sum())))
        d = (piv['t5'] - piv['gpt2']).dropna(); mu, lo, hi = boot_mean_ci(d)
        rows.append(dict(metric=m, system='T5 minus GPT-2', mean=mu, ci_lo=lo, ci_hi=hi, n_items=len(d),
                         share_t5_better=float((d > 0).mean()), share_gpt2_better=float((d < 0).mean()), share_tie=float((d == 0).mean())))
        if len(rater_csvs) >= 2:
            M = np.vstack([L[L.rater == r].set_index(['item', 'system'])[m].reindex(L.set_index(['item', 'system']).index.unique()).values for r in sorted(L.rater.unique())])
            rows.append(dict(metric=m, system='inter-rater', krippendorff_alpha_interval=krippendorff_alpha(M, 'interval'),
                             qwk_rater1_vs_2=cohen_kappa(*[pd.Series(M[i]).dropna().astype(int).values for i in range(2)], 'quadratic') if not np.isnan(M[:2]).any() else np.nan))
    R = pd.DataFrame(rows); R.to_csv(out_csv, index=False); print(R.to_string()); return R

if __name__ == '__main__':
    make_audit_sheet()
