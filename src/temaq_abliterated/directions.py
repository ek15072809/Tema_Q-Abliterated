# -*- coding: utf-8 -*-
"""残差方向 (residual directions) と層間顕著度 (salience) の計算。

有害/無害プロンプトの最初の応答トークン位置における隠れ状態の平均差から、
各レイヤの「拒否方向」を抽出する。良性方向に平行な成分を除去する
投影補正 (projected abliteration) も実装する。

[v6] ハイパーコネクション (Qwen4Exp: hc_countストリーム) 対応:
  Qwen4ExpTextModel の output_hidden_states は層境界によって幅が異なる
  (埋め込み直後=hidden、層間=hc_count×hidden、最終段=hidden)。このような
  「混合幅」アーキテクチャでは単一の残差方向テンソルが定義できないため、
  全コンポーネントについて**モジュール出力方向** (フォワードフックで
  o_proj / out_proj / 専門家down_proj の出力活性の平均差を取る) を使う。
  出力空間の方向こそ「W'x = Wx − w·(vᵀWx)·v が出力から除去する成分」の
  定義そのものであり、数学的にも摂動対象として正しい空間である。
  次元が一致するモデル (Llama/Qwen/Gemma/Qwen3.5等) は従来どおり
  残差方向を使い、一部コンポーネントだけ次元が合わない場合も
  そのコンポーネントだけモジュール出力方向に切り替える。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from .arch import TargetModule, get_decoder_layers
from .msg import tr


class _ModuleOutputCapture:
    """対象モジュール群の出力活性の最終位置を累積するフック集合。

    sum / sumsq / n を (comp, layer) ごとに蓄積し、あとで平均・分散・
    平均差方向を取り出せるようにする (salience計算にも使う)。
    """

    def __init__(self, targets: dict[str, dict[int, list[TargetModule]]]):
        # comp -> layer -> {"sum": (D,), "sumsq": (D,), "n": int}
        self.acc: dict[str, dict[int, dict]] = {}
        self._handles = []
        self._bt: tuple[int, int] | None = None   # (batch, seq_len) 現在のバッチ配置
        for comp, layers in targets.items():
            for li, tms in layers.items():
                for _tm in tms:
                    self.acc.setdefault(comp, {}).setdefault(li, {"sum": None, "sumsq": None, "n": 0})
                    h = _tm.module.register_forward_hook(self._make_hook(comp, li))
                    self._handles.append(h)

    def set_batch(self, batch: int, seq_len: int) -> None:
        """[v6] 現在処理中のバッチ配置を登録する (2D活性の末尾位置復元用)。

        MoEの専門家・共有専門家は (B*T, D) に平坦化された入力で呼ばれる。
        末尾位置 (各系列の最終トークン) を取り出すには B, T が必要。
        """
        self._bt = (int(batch), int(seq_len))

    def _make_hook(self, comp: str, li: int):
        acc = self.acc

        def hook(_mod, _args, output):
            y = output
            if isinstance(y, tuple):
                for cand in y:
                    if torch.is_tensor(cand) and cand.dim() >= 2:
                        y = cand
                        break
            if not torch.is_tensor(y) or y.dim() < 2:
                return
            if y.dim() >= 3:
                last = y[:, -1, :].float()          # (B, D)
            elif y.dim() == 2 and self._bt is not None:
                b, t = self._bt
                if t <= 0 or y.shape[0] != b * t:
                    return   # 想定外の平坦化: この呼び出しは無視
                last = y.view(b, t, y.shape[-1])[:, -1, :].float()   # (B, D)
            else:
                return
            if last.shape[0] <= 0:
                return
            slot = acc[comp][li]
            s = last.detach().sum(dim=0)
            sq = (last.detach() * last.detach()).sum(dim=0)
            if slot["sum"] is None:
                slot["sum"] = s.clone()
                slot["sumsq"] = sq.clone()
            else:
                slot["sum"] += s
                slot["sumsq"] += sq
            slot["n"] += int(last.shape[0])   # 系列数 (行数ではない)
        return hook

    def remove(self) -> None:
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles.clear()

    def _rows(self, fill_fn):
        """comp -> (L+2, ...) 行列化 (未計測レイヤは最近傍補完)。"""
        out: dict[str, torch.Tensor] = {}
        for comp, layers in self.acc.items():
            lis = sorted(layers.keys())
            if not lis:
                continue
            L = max(lis)
            rows = [None] * (L + 2)
            for li in lis:
                slot = layers[li]
                if slot["sum"] is not None and slot["n"] > 0:
                    rows[li + 1] = fill_fn(slot)
            first = next((r for r in rows if r is not None), None)
            if first is None:
                continue
            prev = first
            for i in range(len(rows)):
                if rows[i] is None:
                    rows[i] = prev
                else:
                    prev = rows[i]
            out[comp] = torch.stack([r for r in rows]).clone()
        return out

    def means(self) -> dict[str, torch.Tensor]:
        """comp -> (mean over probes, D)。"""
        return self._rows(lambda slot: slot["sum"] / slot["n"])

    def variances(self) -> dict[str, torch.Tensor]:
        """comp -> (per-dim variance, D) (クラス内散布の近似に使う)。"""
        def _var(slot):
            n = max(1, slot["n"])
            mean = slot["sum"] / n
            return (slot["sumsq"] / n - mean * mean).clamp(min=0.0)
        return self._rows(_var)


def _fisher_salience(mean_bad, mean_good, var_bad, var_good) -> torch.Tensor:
    """層ごとのFisher的顕著度: ||クラス間平均差|| / sqrt(クラス内散布)。"""
    diff = (mean_bad - mean_good).norm(dim=1)           # (L+2,)
    within = (var_bad + var_good).mean(dim=1)           # (L+2,)
    return diff / (within + 1e-6).sqrt()


@torch.no_grad()
def _sweep_probes(model: nn.Module, texts: list[str], tokenizer, device: str,
                  batch_size: int = 8, system_prompt: str = "You are a helpful assistant.",
                  response_prefix: str = "", progress_kind: str = "",
                  capture: _ModuleOutputCapture | None = None,
                  want_hidden: bool = True) -> torch.Tensor | None:
    """各プロンプトの最終位置 (応答直前) における全レイヤ境界の隠れ状態を返す。

    戻り値: (N, L+1, H) — fp32。インデックス0は埋め込み出力。
    want_hidden=False (混合幅アーキテクチャ) の場合は None を返し、
    capture フックの蓄積だけを行う。
    """
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    old_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    from .ui import Progress
    n_batches = list(range(0, len(texts), batch_size))
    prog = Progress(len(n_batches),
                    prefix=tr("dir_prog_prefix",
                              kind=tr(progress_kind) if progress_kind else ""),
                    quiet=False)
    outs = []
    for i in n_batches:
        chunk = texts[i:i + batch_size]
        chats = [[{"role": "system", "content": system_prompt},
                  {"role": "user", "content": t}] for t in chunk]
        prompts = tokenizer.apply_chat_template(chats, add_generation_prompt=True, tokenize=False)
        if response_prefix:
            prompts = [p + response_prefix for p in prompts]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        if capture is not None:
            try:
                capture.set_batch(len(chunk), int(inputs["input_ids"].shape[1]))
            except Exception:
                pass
        # [v7] logits_to_keep=1: 隠れ状態の計算に全位置のlogitsは不要。
        # 大語彙モデル (24万語彙等) では [B, T, V] のlogitsが数百MBを
        # 占め、低メモリ環境のOOM要因になっていた (数値には影響しない)。
        try:
            out = model(**inputs, use_cache=False, output_hidden_states=want_hidden,
                        logits_to_keep=1)
        except TypeError:
            out = model(**inputs, use_cache=False, output_hidden_states=want_hidden)
        if want_hidden:
            hs = out.hidden_states  # tuple(L+1) of (B, T, H)
            try:
                last = torch.stack([h[:, -1, :] for h in hs], dim=1)  # (B, L+1, H)
                outs.append(last.float().cpu())
            except RuntimeError:
                # 混合幅 (ハイパーコネクション等): 残差スタック不可
                want_hidden = False
                prog.close()
                tokenizer.padding_side = old_side
                return None
        # [v7条件④] 進捗は「完了したバッチ数」で更新する
        # (v6はループ先頭で更新し、測定開始の瞬間に 1/N と表示していた)
        prog.update()
    prog.close()
    tokenizer.padding_side = old_side
    if not want_hidden or not outs:
        return None
    return torch.cat(outs, dim=0)


def _probe_hidden_widths(model, tokenizer, device, system_prompt, response_prefix
                         ) -> list[int]:
    """1プロンプトで output_hidden_states の幅リストを測る (混合幅検出)。"""
    try:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        chats = [[{"role": "system", "content": system_prompt},
                  {"role": "user", "content": "Hello"}]]
        prompt = tokenizer.apply_chat_template(chats, add_generation_prompt=True, tokenize=False)
        inputs = tokenizer([prompt + response_prefix], return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs, use_cache=False, output_hidden_states=True)
        # [v11] 3D (B,T,D) 以外の隠れ状態 (DeepSeek V4系のmHC: (B,T,hc_mult,D)
        # 4Dストリーム) は混合幅として扱う。末元の幅はどちらもDのため幅比較で
        # は検出できず、残差方向を誤った軸で正規化してしまう。この場合は既存の
        # 「モジュール出力方向」経路 (ハイパーコネクション対応) へ切替える。
        if any(h.dim() != 3 for h in out.hidden_states):
            return []
        return [int(h.shape[-1]) for h in out.hidden_states]
    except Exception:
        return []


def compute_directions(
    model: nn.Module,
    tokenizer,
    device: str,
    harmful: list[str],
    harmless: list[str],
    batch_size: int = 8,
    system_prompt: str = "You are a helpful assistant.",
    response_prefix: str = "",
    orthogonalize: bool = True,
    return_residuals: bool = False,
    registry=None,
) -> dict[str, torch.Tensor]:
    """残差方向と層顕著度を計算する。

    戻り値: {"directions": (L+1, D) 単位ベクトル (または代表), "salience": (L+1,),
             "directions_map": {comp: (L+1, D_comp)} — モジュール出力方向を使う
             コンポーネント (混合幅アーキテクチャでは全コンポーネント)}
    """
    # --- 戦略判定: 層境界の隠れ状態幅が均一か? ---
    widths = _probe_hidden_widths(model, tokenizer, device, system_prompt, response_prefix)
    uniform = bool(widths) and len(set(widths)) == 1
    residual_dim = widths[-1] if uniform else 0

    capture_comps: dict[str, dict[int, list[TargetModule]]] = {}
    if registry is not None:
        if not uniform:
            # 混合幅 (ハイパーコネクション等): 全コンポーネントをモジュール出力方向で
            capture_comps = dict(registry.map)
        else:
            for comp, layers in registry.map.items():
                mismatch = False
                for tms in layers.values():
                    for tm in tms:
                        try:
                            if tm.out_features() != residual_dim:
                                mismatch = True
                                break
                        except Exception:
                            pass
                    if mismatch:
                        break
                if mismatch:
                    capture_comps[comp] = layers

    # --- 残差方向 (均一幅の場合のみ) ---
    directions: torch.Tensor | None = None
    salience: torch.Tensor | None = None
    if uniform:
        bad = _sweep_probes(model, harmful, tokenizer, device, batch_size,
                            system_prompt, response_prefix, progress_kind="dir_kind_harmful")
        good = _sweep_probes(model, harmless, tokenizer, device, batch_size,
                             system_prompt, response_prefix, progress_kind="dir_kind_harmless")
        if bad is not None and good is not None:
            bad_mean = bad.mean(dim=0)    # (L+1, H)
            good_mean = good.mean(dim=0)

            directions = bad_mean - good_mean
            directions = F.normalize(directions, p=2, dim=1)

            if orthogonalize:
                g_dir = F.normalize(good_mean, p=2, dim=1)
                proj = (directions * g_dir).sum(dim=1, keepdim=True)
                directions = F.normalize(directions - proj * g_dir, p=2, dim=1)

            within = ((good - good_mean[None]).pow(2).mean(dim=(0, 2))
                      + (bad - bad_mean[None]).pow(2).mean(dim=(0, 2)))
            salience = (bad_mean - good_mean).norm(dim=1) / (within + 1e-6).sqrt()

    # --- モジュール出力方向 (キャプチャが必要なコンポーネント) ---
    dmap: dict[str, torch.Tensor] = {}
    cap_salience: torch.Tensor | None = None
    if capture_comps:
        cap_bad = _ModuleOutputCapture(capture_comps)
        try:
            _sweep_probes(model, harmful, tokenizer, device, batch_size,
                          system_prompt, response_prefix,
                          progress_kind="dir_kind_harmful", capture=cap_bad,
                          want_hidden=False)
            mean_bad = cap_bad.means()
            var_bad = cap_bad.variances()
        finally:
            cap_bad.remove()
        cap_good = _ModuleOutputCapture(capture_comps)
        try:
            _sweep_probes(model, harmless, tokenizer, device, batch_size,
                          system_prompt, response_prefix,
                          progress_kind="dir_kind_harmless", capture=cap_good,
                          want_hidden=False)
            mean_good = cap_good.means()
            var_good = cap_good.variances()
        finally:
            cap_good.remove()

        for comp, mb in mean_bad.items():
            mg = mean_good.get(comp)
            if mg is None or mb.shape != mg.shape:
                continue
            d = mb - mg
            d = F.normalize(d, p=2, dim=1)
            if orthogonalize:
                gd = F.normalize(mg, p=2, dim=1)
                proj = (d * gd).sum(dim=1, keepdim=True)
                d = F.normalize(d - proj * gd, p=2, dim=1)
            dmap[comp] = d
        # 混合幅の場合: salience をモジュール出力のFisherスコアから構成
        if salience is None and mean_bad:
            sal_parts = []
            for comp in mean_bad:
                vb = var_bad.get(comp)
                vg = var_good.get(comp)
                if vb is None or vg is None or vb.shape != mean_bad[comp].shape:
                    continue
                s = _fisher_salience(mean_bad[comp], mean_good[comp], vb, vg)
                sal_parts.append(s[:s.shape[0] - 1])   # 末尾の補完行を落とす
            if sal_parts:
                n_layers = sal_parts[0].shape[0] - 1
                sal = torch.stack(sal_parts).mean(dim=0)   # (L+1,)
                salience = sal[:n_layers + 1]
            else:
                salience = torch.zeros(max(1, len(widths)))

    if directions is None:
        # 混合幅: directions は代表 (最初のコンポーネント方向) で埋める
        directions = next(iter(dmap.values())) if dmap else torch.zeros((1, 1))
    if salience is None:
        salience = torch.zeros(directions.shape[0])

    result: dict[str, torch.Tensor] = {"directions": directions, "salience": salience}
    if dmap:
        result["directions_map"] = dmap

    if return_residuals:
        # 混合幅では残差は返せない (互換用に空)
        result["good_residuals"] = torch.zeros(0)
        result["bad_residuals"] = torch.zeros(0)
    return result


def salience_peak_layer(salience: torch.Tensor, n_layers: int) -> int:
    """顕著度ピークのレイヤ番号 (1..n_layers、embeddings=0を除く) を返す。"""
    s = salience[1:n_layers + 1]
    return int(torch.argmax(s).item()) + 1


def cache_key_for(model_dir: Path, harmful: list[str], harmless: list[str], n_layers: int,
                  components: str = "both") -> str:
    """方向キャッシュのキー (モデル+プローブ+構成のハッシュ)。"""
    h = hashlib.sha256()
    h.update(str(model_dir).encode())
    h.update(str(n_layers).encode())
    h.update(components.encode())
    for t in harmful + harmless:
        h.update(t.encode())
    return h.hexdigest()[:16]


def load_or_compute_directions(
    model, tokenizer, device, model_dir: Path, cache_root: Path,
    harmful: list[str], harmless: list[str], batch_size: int,
    system_prompt: str, response_prefix: str, n_layers: int,
    registry=None, components: str = "both",
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor, bool]:
    """方向のキャッシュがあれば読み、なければ計算して保存する。

    戻り値: (directions[残差 or 代表], directions_map[comp別 or 空], salience, from_cache)
    """
    key = cache_key_for(model_dir, harmful, harmless, n_layers, components)
    path = cache_root / "directions" / f"{key}.pt"
    if path.exists():
        try:
            data = torch.load(path, map_location="cpu", weights_only=True)
            if data["directions"].shape[0] == n_layers + 1:
                dmap = data.get("directions_map", {}) or {}
                print(tr("dir_cached", name=path.name))
                if dmap:
                    print(tr("dir_map_used", n=len(dmap)))
                return data["directions"], dmap, data["salience"], True
        except Exception:
            pass
    print(tr("dir_computing"))
    data = compute_directions(model, tokenizer, device, harmful, harmless,
                              batch_size=batch_size, system_prompt=system_prompt,
                              response_prefix=response_prefix, registry=registry)
    path.parent.mkdir(parents=True, exist_ok=True)
    dmap = data.get("directions_map", {})
    torch.save({"directions": data["directions"], "salience": data["salience"],
                "directions_map": dmap}, path)
    if dmap:
        print(tr("dir_map_used", n=len(dmap)))
    return data["directions"], dmap, data["salience"], False
