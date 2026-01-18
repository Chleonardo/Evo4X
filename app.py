import os
import time
import streamlit as st

from evo4x_engine_v2_6_13_rtsgrid_uiapi import init_new_run, simulate_tick, get_ui_snapshot

APP_TITLE = "Evo4X — Streamlit UI (v2.6.13+)"
RUNS_DIR = "runs"


def fmt(x, nd=3):
    if x is None:
        return "-"
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def list_saves(runs_dir: str) -> list[str]:
    if not os.path.isdir(runs_dir):
        return []
    saves = [os.path.join(runs_dir, f) for f in os.listdir(runs_dir) if f.endswith('.json')]
    saves.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return saves


def species_color(sid: str) -> str:
    pal = [
        "#2E6B4F", "#5FA16A", "#B4C84A", "#8FAE4A", "#D6C15C",
        "#5F7F8F", "#9EC3D1", "#243F2E", "#6C7A63",
    ]
    if sid == "npc":
        return "#C97A3A"
    if sid == "unused":
        return "#3A4248"
    try:
        idx = int(str(sid).replace('sp', ''))
    except Exception:
        idx = 0
    return pal[idx % len(pal)]


def cell_arrow(trend: str | None) -> str:
    if trend == "up":
        return "↑"
    if trend == "down":
        return "↓"
    return "→"


def render_fill_bar(fill: dict | None):
    if not fill or not isinstance(fill, dict):
        return
    segs = fill.get('segments') or []
    if not segs:
        return
    parts = []
    for seg in segs:
        sid = str(seg.get('id'))
        share = float(seg.get('share', 0.0))
        if share <= 1e-6:
            continue
        w = max(0.0, min(100.0, 100.0 * share))
        col = species_color(sid)
        parts.append(f"<div title='{sid}: {share*100:.1f}%' style='width:{w:.2f}%;height:8px;background:{col};'></div>")
    if not parts:
        return
    html = (
        "<div style='display:flex;gap:0px;width:100%;height:8px;"
        "border-radius:6px;overflow:hidden;background:#232629;'>" + "".join(parts) + "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


# -----------------------------
# Session / actions
# -----------------------------

def ensure_state():
    st.session_state.setdefault('save_path', None)
    st.session_state.setdefault('selected_cell', 'c11')
    st.session_state.setdefault('active_species', 'sp0')
    st.session_state.setdefault('last_error', None)


def safe_snapshot():
    sp = st.session_state.get('save_path')
    if not sp:
        return None
    return get_ui_snapshot(sp, st.session_state.get('selected_cell'))


def do_new_game(seed: int | None = None):
    os.makedirs(RUNS_DIR, exist_ok=True)
    base_seed = int(seed) if seed is not None else int(time.time())
    save_path, _hud = init_new_run(
        out_dir=RUNS_DIR,
        world_seed=base_seed,
        npc_seed=base_seed + 101,
        event_seed=base_seed + 202,
        card_seed=base_seed + 303,
    )
    st.session_state['save_path'] = save_path
    st.session_state['selected_cell'] = 'c11'
    st.session_state['active_species'] = 'sp0'
    st.session_state['last_error'] = None


def do_load_game(path: str):
    st.session_state['save_path'] = path
    st.session_state['last_error'] = None


def do_tick(action: dict | None = None):
    action = action or {}
    try:
        save_path = st.session_state['save_path']
        save_path, _res = simulate_tick(save_path, action=action, out_dir=RUNS_DIR)
        st.session_state['save_path'] = save_path
        st.session_state['last_error'] = None
    except Exception as e:
        st.session_state['last_error'] = str(e)


# -----------------------------
# Renderers
# -----------------------------

def render_hud(snapshot: dict):
    hud = snapshot.get('hud', {}) or {}
    econ = snapshot.get('economy', {}) or {}

    st.title(APP_TITLE)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric('Tick', hud.get('tick', '-'))
    c2.metric('Evo', fmt(hud.get('evo_balance'), 3))
    c3.metric('Passive next', fmt(hud.get('passive_preview_next'), 3))
    c4.metric('Player pop', fmt(hud.get('total_population'), 3))

    npc_total = hud.get('npc_total', hud.get('npc_total_population'))
    c5.metric('NPC pop', fmt(npc_total, 3))

    dom = hud.get('dominance', hud.get('dominance_ratio'))
    dom = float(dom or 0.0)
    c6.metric('Dominance', f"{dom*100:.1f}%")
    st.progress(max(0.0, min(1.0, dom)))

    active = st.session_state.get('active_species', 'sp0')
    st.caption(
        f"Trait cost ({active}): {fmt((econ.get('trait_costs') or {}).get(active), 2)} | "
        f"Fork cost: {fmt(econ.get('fork_cost'), 2)}"
    )

    # --- Active effects (global, always visible) ---
    st.markdown("**Active effects (global)**")
    ae = snapshot.get("active_effects") or []
    if ae:
        # компактный список
        for e in ae:
            try:
                cell = e.get("cell")
                res = e.get("res")
                mult = float(e.get("mult"))
                ttl = int(e.get("ttl"))
                st.write(f"- {cell}: {res} ×{mult:.2f} ({ttl}t)")
            except Exception:
                st.write(f"- {e}")
    else:
        st.caption("— none —")



def render_map(snapshot: dict):
    m = snapshot.get('map', {}) or {}
    cells = m.get('cells', {}) or {}
    hud = snapshot.get('hud', {}) or {}
    pop_by_cell = hud.get('player_pop_by_cell') or {}

    grid = [
        ['c00', 'c01', 'c02'],
        ['c10', 'c11', 'c12'],
        ['c20', 'c21', 'c22'],
    ]

    st.subheader('Map')

    for r in range(3):
        cols = st.columns(3)
        for c in range(3):
            cid = grid[r][c]
            info = cells.get(cid, {}) or {}
            is_scouted = bool(info.get('is_scouted', False))
            has_p = 'P' if info.get('has_player') else '·'
            has_n = 'N' if info.get('has_npc') else '·'
            pop = float(pop_by_cell.get(cid, 0.0))
            arrow = cell_arrow(info.get('trend_total'))
            fog = '' if is_scouted else ' ?'
            label = f"{cid} [{has_p}{has_n}] pop {fmt(pop,2)} {arrow}{fog}"

            is_selected = (st.session_state.get('selected_cell') == cid)
            disabled = (not is_scouted) and (not info.get('has_player'))

            if cols[c].button(label, use_container_width=True, type=('primary' if is_selected else 'secondary'), disabled=disabled):
                st.session_state['selected_cell'] = cid

            if is_scouted:
                with cols[c]:
                    render_fill_bar(info.get('fill'))



def render_cell(snapshot: dict):
    selected = snapshot.get('selected_cell') or {}
    inspector = (selected.get('inspector') or {})
    passport = (selected.get('passport') or {})
    consumption = (selected.get('consumption') or {})  # engine кладёт сюда
    cid = selected.get('cell_id', st.session_state.get('selected_cell'))

    st.subheader(f"Selected cell: {cid}")

    st.markdown('**Resources**')
    st.write(inspector.get('resources'))

    st.markdown('**Neighbors**')
    st.write(inspector.get('neighbors'))

    st.markdown('**Active effects (this cell)**')

    detailed = passport.get('active_effects_detailed') or []
    if detailed:
        for e in detailed:
            try:
                res = e.get("res")
                mult = float(e.get("mult"))
                ttl = int(e.get("ttl"))
                st.write(f"- {res} ×{mult:.2f} ({ttl}t)")
            except Exception:
                st.write(f"- {e}")
    else:
        st.caption("— none —")

    # оставляем старое агрегированное как fallback/справку (не обязательно, но полезно)
    agg = passport.get('active_effects')
    if agg:
        st.caption("Effective multipliers (aggregated):")
        st.write(agg)


    st.markdown('**Populations in cell**')
    st.write(inspector.get('populations'))

    # ---- ВОЗВРАЩАЕМ ДЕТАЛИ ВИДОВ В КЛЕТКЕ ----
    st.markdown('**NPC species (passport.npc_species)**')
    npc_species = passport.get('npc_species') or []
    if npc_species:
        for row in npc_species:
            st.code(row, language="json")
    else:
        st.caption("— none —")

    st.markdown('**Player species (passport.player_species)**')
    player_species = passport.get('player_species') or []
    if player_species:
        for row in player_species:
            st.code(row, language="json")
    else:
        st.caption("— none —")

    st.markdown('**Consumption breakdown**')
    if consumption:
        st.write("cell_capacity_total:", consumption.get("cell_capacity_total"))
        st.write("resources_effective:", consumption.get("resources_effective"))
        rows = consumption.get("rows") or []
        if rows:
            for row in rows:
                st.code(row, language="json")
        else:
            st.caption("— no rows —")
    else:
        st.caption("— none —")



def render_events(snapshot: dict):
    st.subheader('Last tick: events & reveals')
    ev = snapshot.get('last_events', []) or []
    rv = snapshot.get('last_reveals', []) or []

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('**Events**')
        if ev:
            for e in ev:
                st.code(e, language='json')
        else:
            st.caption('— none —')

    with col2:
        st.markdown('**Reveals**')
        if rv:
            for r in rv:
                st.code(r, language='json')
        else:
            st.caption('— none —')


# -----------------------------
# UI
# -----------------------------

ensure_state()

st.sidebar.header('Game control')

# Load Save
with st.sidebar.expander('Load Save', expanded=True):
    os.makedirs(RUNS_DIR, exist_ok=True)
    saves = list_saves(RUNS_DIR)
    if saves:
        current = st.session_state.get('save_path')
        idx = saves.index(current) if current in saves else 0
        chosen = st.selectbox('Save file', saves, index=idx, format_func=lambda p: os.path.basename(p))
        if st.button('📂 Load', use_container_width=True):
            do_load_game(chosen)
            st.rerun()
    else:
        st.caption('No saves yet. Create a new game first.')

# New Game
with st.sidebar.expander('New Game', expanded=False):
    seed_str = st.text_input('Seed (optional, int)', value='')
    if st.button('🆕 New Game', use_container_width=True):
        seed = None
        if seed_str.strip():
            try:
                seed = int(seed_str.strip())
            except Exception:
                st.sidebar.error('Seed must be an integer.')
        do_new_game(seed=seed)
        st.rerun()

if not st.session_state.get('save_path'):
    st.info('Нажми **Load** (если есть сейв) или **New Game** слева, чтобы начать.')
    st.stop()

snap = safe_snapshot()
if not snap:
    st.error('Snapshot is empty. Something is wrong with save_path.')
    st.stop()

species_ids = [sp.get('species_id') for sp in (snap.get('species') or []) if sp.get('species_id')]
if species_ids and st.session_state.get('active_species') not in species_ids:
    st.session_state['active_species'] = species_ids[0]

with st.sidebar.expander('Actions', expanded=True):
    st.caption('Покупка трейтов — мгновенно. Миграция/форк — с тиком. Есть пустой тик.')

    active = st.selectbox('Active species', species_ids, index=species_ids.index(st.session_state.get('active_species','sp0')) if species_ids else 0)
    st.session_state['active_species'] = active

    if st.button('⏭️ Next tick (no actions)', use_container_width=True):
        do_tick(action={})
        st.rerun()

    eco = snap.get('economy', {}) or {}
    evo = float(eco.get('evo_balance') or 0.0)

    # Buy trait
    st.divider()
    st.markdown('**Buy trait (instant)**')
    trait_cost = float((eco.get('trait_costs') or {}).get(active) or 0.0)
    cols = st.columns(4)
    for i, key in enumerate(['r', 'b1', 'b2', 'b3']):
        with cols[i]:
            if st.button(f"Buy {key}\n({fmt(trait_cost,2)})", use_container_width=True, disabled=(evo < trait_cost)):
                do_tick(action={'buy_trait': (active, key)})
                st.rerun()

    # Migration
    st.divider()
    st.markdown('**Migration (with tick)**')
    active_cells = list((snap.get('species_cells', {}).get(active, {}) or {}).keys())
    if not active_cells:
        st.caption('No population to migrate.')
    else:
        src = st.selectbox('From', active_cells, key='mig_src')
        nbrs = (snap.get('map', {}).get('cells', {}).get(src, {}) or {}).get('neighbors', []) or []
        dst = st.selectbox('To (neighbor)', nbrs, key='mig_dst')
        max_take = float((snap.get('species_cells', {}).get(active, {}) or {}).get(src, {}).get('population', 0.0))
        amt = st.number_input('Amount', min_value=0.0, max_value=max_take, step=1.0, value=0.0, key='mig_amt')
        action = {}
        if amt and amt > 0:
            action['migrations'] = [(active, src, dst, float(amt))]
        if st.button('⏭️ Next tick (migration)', use_container_width=True):
            do_tick(action=action)
            st.rerun()

    # Fork
    st.divider()
    st.markdown('**Fork (with tick)**')
    fork_cells = list((snap.get('species_cells', {}).get(active, {}) or {}).keys())
    if not fork_cells:
        st.caption('No population to fork from.')
    else:
        fk_src = st.selectbox('Fork from cell', fork_cells, key='fk_src')
        fk_all = st.checkbox('Take ALL from source cell', value=False, key='fk_all')
        fk_starter = st.selectbox('Starter trait', ['r', 'b1', 'b2', 'b3'], key='fk_starter')
        max_take = float((snap.get('species_cells', {}).get(active, {}) or {}).get(fk_src, {}).get('population', 0.0))
        fk_amt = st.number_input('Split amount', min_value=0.0, max_value=max_take, step=1.0, value=min(2.0, max_take), disabled=fk_all, key='fk_amt')
        if st.button('🧬 Fork + Next tick', use_container_width=True):
            split = 'ALL' if fk_all else float(fk_amt)
            do_tick(action={'fork': (active, split, fk_starter, fk_src)})
            st.rerun()

if st.session_state.get('last_error'):
    st.error(st.session_state['last_error'])

# Main
render_hud(snap)
left, right = st.columns([1.15, 0.85])
with left:
    render_map(snap)
    render_events(snap)
with right:
    render_cell(snap)
