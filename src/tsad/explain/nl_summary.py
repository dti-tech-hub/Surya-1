"""6. Rule-based natural-language summary (no LLM dependency)."""
from __future__ import annotations


def summarize(exp: dict) -> str:
    p = exp["parts"]
    ch = ", ".join(f"ch{c['channel']} ({c['importance']:.2f})"
                   for c in p.get("attribution", {}).get("top_channels", []))
    cf = p.get("counterfactual", {})
    cf_txt = (f" A counterfactual ({'valid' if cf.get('valid') else 'INVALID'}) restores normal by "
              f"changing {cf.get('frac_changed', 0) * 100:.1f}% of entries "
              f"(L1={cf.get('delta_l1', 0):.2f}), mostly " +
              ", ".join(f"ch{c}" for c in cf.get("top_channels", [])) + ".") if cf else ""
    attn = ""
    if p.get("attention", {}).get("available"):
        attn = f" Attention peaks at timesteps {p['attention']['top_timesteps']}."
    xd = p.get("cross_domain", {})
    xd_txt = (f" Closest normal: {xd.get('closest_normal')}; "
              f"invariant dims {len(xd.get('invariant_dims', []))}.") if xd else ""
    return (f"Window {exp['t_start']}-{exp['t_end']} flagged with score {exp['score']:.3f} "
            f"(threshold {exp['threshold']:.3f}). Most responsible channels: {ch}."
            f"{attn}{cf_txt}{xd_txt}")
