### Tema visual do VisioVetor para Gradio.

from __future__ import annotations

from html import escape
from pathlib import Path

import gradio as gr

CSS_PATH = Path(__file__).with_name("ui.css")

# Paleta zinc + marca sky (Tailwind)
Z = {50: "#fafafa", 100: "#f4f4f5", 200: "#e4e4e7", 300: "#d4d4d8", 400: "#a1a1aa", 500: "#71717a",
     600: "#52525b", 700: "#3f3f46", 800: "#27272a", 900: "#18181b", 950: "#09090b"}


def build_theme() -> gr.themes.Base:
    theme = gr.themes.Base(
        primary_hue=gr.themes.colors.sky,
        secondary_hue=gr.themes.colors.zinc,
        neutral_hue=gr.themes.colors.zinc,
        radius_size=gr.themes.sizes.radius_md,
        text_size=gr.themes.sizes.text_md,
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
    )
    return theme.set(
        # fundo e texto
        body_background_fill="#ffffff",
        body_background_fill_dark=Z[950],
        body_text_color=Z[900],
        body_text_color_dark=Z[50],
        body_text_color_subdued=Z[500],
        body_text_color_subdued_dark=Z[400],
        background_fill_primary="#ffffff",
        background_fill_primary_dark=Z[950],
        background_fill_secondary=Z[50],
        background_fill_secondary_dark=Z[900],
        border_color_primary="rgba(228,228,231,.8)",
        border_color_primary_dark=Z[800],
        color_accent="#0284c7",
        color_accent_soft="#f0f9ff",
        color_accent_soft_dark="rgba(14,165,233,.1)",
        # blocos / cards: ring de 1px, sombra mínima, rounded-2xl
        block_background_fill="#ffffff",
        block_background_fill_dark=Z[900],
        block_border_width="1px",
        block_border_color="rgba(9,9,11,.08)",
        block_border_color_dark="rgba(255,255,255,.1)",
        block_radius="16px",
        block_shadow="0 1px 2px rgba(9,9,11,.04)",
        block_shadow_dark="none",
        block_padding="16px",
        block_label_background_fill="transparent",
        block_label_background_fill_dark="transparent",
        block_label_border_width="0px",
        block_label_text_color=Z[500],
        block_label_text_color_dark=Z[400],
        block_label_text_weight="500",
        block_title_text_weight="500",
        block_title_text_color=Z[700],
        block_title_text_color_dark=Z[300],
        container_radius="16px",
        layout_gap="16px",
        # campos: 40px, rounded-lg, foco com borda escura + ring 10%
        input_background_fill="#ffffff",
        input_background_fill_dark=Z[950],
        input_border_color=Z[300],
        input_border_color_dark=Z[700],
        input_border_color_focus=Z[900],
        input_border_color_focus_dark=Z[300],
        input_border_width="1px",
        input_radius="8px",
        input_shadow="0 1px 2px rgba(9,9,11,.04)",
        input_shadow_dark="none",
        input_shadow_focus="0 0 0 3px rgba(9,9,11,.10)",
        input_shadow_focus_dark="0 0 0 3px rgba(255,255,255,.12)",
        input_placeholder_color=Z[400],
        input_placeholder_color_dark=Z[500],
        # botões: primário zinc-900 (invertido no escuro), secundário branco com borda
        button_large_radius="8px",
        button_medium_radius="8px",
        button_small_radius="9999px",
        button_large_padding="10px 18px",
        button_large_text_weight="500",
        button_medium_text_weight="500",
        button_small_text_weight="500",
        button_transition="all .15s cubic-bezier(0.32,0.72,0,1)",
        button_primary_background_fill=Z[900],
        button_primary_background_fill_hover=Z[700],
        button_primary_background_fill_dark="#ffffff",
        button_primary_background_fill_hover_dark=Z[200],
        button_primary_text_color="#ffffff",
        button_primary_text_color_dark=Z[900],
        button_primary_border_color=Z[900],
        button_primary_border_color_dark="#ffffff",
        button_primary_shadow="0 1px 2px rgba(9,9,11,.1)",
        button_secondary_background_fill="#ffffff",
        button_secondary_background_fill_hover=Z[50],
        button_secondary_background_fill_dark=Z[900],
        button_secondary_background_fill_hover_dark=Z[800],
        button_secondary_text_color=Z[900],
        button_secondary_text_color_dark=Z[100],
        button_secondary_border_color=Z[300],
        button_secondary_border_color_dark=Z[700],
        # slider / checkbox na cor da marca, com moderação
        slider_color="#0284c7",
        slider_color_dark="#38bdf8",
        checkbox_label_background_fill_selected="#f0f9ff",
        checkbox_label_background_fill_selected_dark="rgba(14,165,233,.1)",
        table_radius="12px",
        embed_radius="16px",
        shadow_drop="0 1px 2px rgba(9,9,11,.05)",
        shadow_drop_lg="0 10px 30px -10px rgba(9,9,11,.12)",
    )


def load_css() -> str:
    return CSS_PATH.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Ícones (estilo Lucide, stroke 2, currentColor)
# ---------------------------------------------------------------------------
_ICONS = {
    "scan": '<path d="M3 7V5a2 2 0 0 1 2-2h2"/><path d="M17 3h2a2 2 0 0 1 2 2v2"/><path d="M21 17v2a2 2 0 0 1-2 2h-2"/><path d="M7 21H5a2 2 0 0 1-2-2v-2"/><circle cx="12" cy="12" r="3"/>',
    "sun": '<circle cx="12" cy="12" r="4"/><path d="M12 2v2"/><path d="M12 20v2"/><path d="m4.93 4.93 1.41 1.41"/><path d="m17.66 17.66 1.41 1.41"/><path d="M2 12h2"/><path d="M20 12h2"/><path d="m6.34 17.66-1.41 1.41"/><path d="m19.07 4.93-1.41 1.41"/>',
    "moon": '<path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z"/>',
    "arrow-up": '<path d="m5 12 7-7 7 7"/><path d="M12 19V5"/>',
    "monitor": '<rect width="20" height="14" x="2" y="3" rx="2"/><path d="M8 21h8"/><path d="M12 17v4"/>',
}


def icon(name: str, cls: str = "vv-icon") -> str:
    return (f'<svg class="{cls}" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
            f'stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" '
            f'aria-hidden="true">{_ICONS[name]}</svg>')


def num(v: float, dec: int = 1) -> str:
    """Formata número no padrão pt-BR."""
    s = f"{v:,.{dec}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


# ---------------------------------------------------------------------------
# Blocos de HTML
# ---------------------------------------------------------------------------
def hero_html(n_images: int, n_captions: int, n_tags: int, device: str, gain_r1: float | None,
              dim: int, model_name: str) -> str:
    theme_opts = "".join(
        f'<label class="vv-seg" title="{t}"><input type="radio" name="vv-tema" value="{v}">'
        f'{icon(ic)}<span class="vv-sr">{t}</span></label>'
        for v, t, ic in (("claro", "Claro", "sun"), ("escuro", "Escuro", "moon"), ("sistema", "Sistema", "monitor"))
    )
    stats = [("Imagens", num(n_images, 0)), ("Legendas", num(n_captions, 0)), ("Tags", str(n_tags)),
             ("Dispositivo", device.upper())]
    if gain_r1 is not None:
        stats.insert(0, ("Ganho R@1 com fine-tuning", f"{'+' if gain_r1 >= 0 else ''}{num(gain_r1)} pts"))
    stats_html = "".join(f'<div class="vv-stat"><dt>{escape(k)}</dt><dd>{escape(v)}</dd></div>' for k, v in stats)
    return f"""
<header class="vv-hero">
  <div class="vv-topbar">
    <span class="vv-brand"><span class="vv-logo">{icon("scan")}</span>VisioVetor</span>
    <div class="vv-theme" role="radiogroup" aria-label="Tema">{theme_opts}</div>
  </div>
  <p class="vv-eyebrow"><span class="vv-dot vv-sky"></span>Busca semântica multimodal</p>
  <h1>Encontre imagens descrevendo o que você procura.</h1>
  <p class="vv-lead">Imagens e textos no mesmo espaço vetorial de {dim} dimensões, com
  <code class="vv-code">{escape(model_name)}</code> ajustado para português. Sem tags manuais: a busca entende a descrição.</p>
  <dl class="vv-stats">{stats_html}</dl>
  {how_it_works_html()}
</header>"""


_STEPS = (
    ("sky", "Escolha o modo",
     "Use as abas abaixo: buscar por texto, por imagem parecida, descrever uma foto ou comparar modelos."),
    ("violet", "Descreva ou envie",
     "Escreva a cena em português (ex.: “crianças brincando na água”) ou solte uma imagem. Tags refinam o filtro."),
    ("teal", "Veja e converse",
     "Os resultados chegam ordenados por similaridade. O Jeffrey explica o porquê e responde suas perguntas."),
)


def how_it_works_html() -> str:
    """Faixa 'Como funciona': 3 passos numerados ligados por setas."""
    arrow = ('<li class="vv-step-arrow" aria-hidden="true"><svg viewBox="0 0 24 24" fill="none" '
             'stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
             '<path d="M5 12h14"/><path d="m13 6 6 6-6 6"/></svg></li>')
    items = []
    for i, (color, title, desc) in enumerate(_STEPS, 1):
        if i > 1:
            items.append(arrow)
        items.append(f"""
    <li class="vv-step">
      <span class="vv-step-num vv-step-{color}">{i}</span>
      <div><p class="vv-step-title">{escape(title)}</p><p class="vv-step-desc">{escape(desc)}</p></div>
    </li>""")
    return f"""
  <section class="vv-how" aria-labelledby="vv-how-title">
    <p class="vv-eyebrow" id="vv-how-title"><span class="vv-dot vv-violet"></span>Como funciona</p>
    <ol class="vv-steps">{''.join(items)}</ol>
  </section>"""


# Liga o seletor Claro / Escuro / Sistema. "Sistema" é o padrão (nada salvo).
THEME_JS = r"""
const KEY = 'vv-tema';
const mq = window.matchMedia('(prefers-color-scheme: dark)');
const ler = () => { try { return localStorage.getItem(KEY) || 'sistema'; } catch (e) { return 'sistema'; } };
const salvar = (v) => { try { v === 'sistema' ? localStorage.removeItem(KEY) : localStorage.setItem(KEY, v); } catch (e) {} };
const aplicar = (v) => {
  const b = document.body;
  b.classList.add('vv-sem-transicao');
  b.classList.toggle('dark', v === 'escuro' || (v === 'sistema' && mq.matches));
  requestAnimationFrame(() => requestAnimationFrame(() => b.classList.remove('vv-sem-transicao')));
};
const radios = element.querySelectorAll('input[name="vv-tema"]');
const atual = ler();
radios.forEach((r) => {
  r.checked = r.value === atual;
  r.addEventListener('change', () => { salvar(r.value); aplicar(r.value); });
});
mq.addEventListener('change', () => { if (ler() === 'sistema') aplicar('sistema'); });
aplicar(atual);
"""


def section_html(color: str, eyebrow: str, title: str, desc: str) -> str:
    return f"""
<div class="vv-section">
  <p class="vv-eyebrow"><span class="vv-dot vv-{color}"></span>{escape(eyebrow)}</p>
  <h2>{escape(title)}</h2>
  <p class="vv-desc">{escape(desc)}</p>
</div>"""


_METRIC_NAMES = {"t2i": "Texto → Imagem", "i2t": "Imagem → Texto"}


def metrics_html(m: dict | None) -> str:
    if not m:
        return """
<div class="vv-empty">
  <p class="vv-empty-title">Ainda não há modelo ajustado</p>
  <p>Rode <code>python scripts/finetune.py</code> para treinar e comparar com o modelo base.</p>
</div>"""
    rows = []
    for key, base in m["baseline"].items():
        ft = m["best"][key]
        kind, k = key.split("_")
        rows.append(f"""
<div class="vv-metric">
  <div class="vv-metric-head">
    <span>{_METRIC_NAMES.get(kind, kind)} <span class="vv-muted">· {k}</span></span>
    <span class="vv-badge {'vv-badge-teal' if ft >= base else 'vv-badge-rose'}"><span class="vv-dot {'vv-teal' if ft >= base else 'vv-rose'}"></span>{'+' if ft >= base else ''}{num(ft - base)}</span>
  </div>
  <div class="vv-bar-row"><span class="vv-bar-label">Base</span>
    <progress class="vv-bar vv-bar-base" max="100" value="{base:.1f}"></progress><span class="vv-num">{num(base)}%</span></div>
  <div class="vv-bar-row"><span class="vv-bar-label">Ajustado</span>
    <progress class="vv-bar vv-bar-ft" max="100" value="{ft:.1f}"></progress><span class="vv-num">{num(ft)}%</span></div>
</div>""")
    return f"""
<div class="vv-metrics-card">
  <p class="vv-metrics-meta">{num(m['n_train'], 0)} imagens de treino · {num(m['n_val'], 0)} de validação ·
  melhor época {m['best_epoch']} · {num(m['train_seconds'] / 60)} min de treino.
  R@K é a porcentagem de buscas em que o item certo aparece entre os K primeiros.</p>
  <div class="vv-metrics-grid">{''.join(rows)}</div>
</div>"""


# Mesmo avatar dos balões do chat (src/assets/jeffrey.svg), embutido no HTML
JEFF_SVG = (Path(__file__).with_name("assets") / "jeffrey.svg").read_text(encoding="utf-8").strip()

CHAT_HEADER = f"""
<div class="vv-chat-head">
  <span class="vv-chat-avatar" aria-hidden="true">{JEFF_SVG}</span>
  <div><p class="vv-chat-name">Jeffrey</p><p class="vv-chat-status">Assistente do VisioVetor</p></div>
</div>"""

_CHAT_INTRO = {
    "texto": "Pergunte por que uma foto ficou em 1º, o que o score significa ou como refinar a busca.",
    "imagem": "Pergunte por que essas fotos são parecidas ou peça para ele comentar uma posição específica.",
    "descrever": "Pergunte como ler a classificação zero-shot ou por que uma legenda ficou mais perto da foto.",
    "comparar": "Pergunte qual modelo foi melhor nesta busca ou por que os resultados mudaram.",
}


def chat_intro_html(kind: str) -> str:
    return section_html("sky", "Assistente", "Converse com o Jeffrey", _CHAT_INTRO.get(kind, _CHAT_INTRO["texto"]))


_FOOTER_MODES = (
    ("sky", "Texto → Imagem", "Busca por descrição"),
    ("violet", "Imagem → Imagem", "Similaridade visual"),
    ("teal", "Imagem → Texto", "Descrição automática"),
    ("amber", "Base × Ajustado", "Ganho do fine-tuning"),
)


def footer_html(model_name: str, dim: int, device: str, n_images: int) -> str:
    modes = "".join(
        f'<li><button type="button" class="vv-foot-link" data-vv-tab="{escape(tab)}">'
        f'<span class="vv-dot vv-{c}"></span><span>{escape(tab)}</span>'
        f'<span class="vv-muted">· {escape(desc)}</span></button></li>'
        for c, tab, desc in _FOOTER_MODES
    )
    tech = "".join(f"<li>{t}</li>" for t in (
        f'<code class="vv-code">{escape(model_name)}</code>',
        "PyTorch + Hugging Face Transformers",
        "Busca vetorial em NumPy (Q · Vᵀ)",
        "Interface em Gradio",
    ))
    return f"""
<footer class="vv-footer">
  <div class="vv-foot-grid">
    <div class="vv-foot-brand">
      <span class="vv-brand"><span class="vv-logo">{icon("scan")}</span>VisioVetor</span>
      <p>Busca semântica multimodal em português: imagens e textos no mesmo espaço vetorial,
      comparados por similaridade de cosseno.</p>
      <p class="vv-foot-status"><span class="vv-dot vv-teal"></span>Rodando em {escape(device.upper())} ·
      {num(n_images, 0)} imagens · {dim} dimensões</p>
    </div>
    <nav aria-label="Modos de busca">
      <p class="vv-foot-title">Modos</p>
      <ul>{modes}</ul>
    </nav>
    <div>
      <p class="vv-foot-title">Tecnologia</p>
      <ul class="vv-foot-tech">{tech}</ul>
    </div>
  </div>
  <div class="vv-foot-bar">
    <span>Similaridade de cosseno sobre vetores normalizados: <span class="vv-foot-math">sim(q, v) = q̂ · v̂</span></span>
    <button type="button" class="vv-foot-top" data-vv-top>{icon("arrow-up")}Voltar ao topo</button>
  </div>
</footer>"""


# Links do rodapé: troca de aba e volta ao topo.
FOOTER_JS = r"""
element.querySelectorAll('[data-vv-tab]').forEach((b) => b.addEventListener('click', () => {
  const tab = [...document.querySelectorAll('.vv-tabs [role="tab"]')]
    .find((t) => t.textContent.trim() === b.dataset.vvTab);
  if (!tab) return;
  tab.click();
  document.querySelector('.vv-tabs')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
}));
element.querySelector('[data-vv-top]')?.addEventListener('click', () => window.scrollTo({ top: 0, behavior: 'smooth' }));
"""
