import os
import time
import streamlit as st

from evo4x_engine_v2_6_13_rtsgrid_uiapi import (
    init_new_run, simulate_tick, get_ui_snapshot,
    SAVANNA_RESOURCE_UI, SAVANNA_SPECIES, SAVANNA_PLAYER_ICONS,
    SAVANNA_ARCHETYPES,
)

st.set_page_config(page_title="Evo4X", layout="wide", initial_sidebar_state="expanded")

RUNS_DIR = "runs"


def fmt(x, nd=2):
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
    pal = ["#2E6B4F","#5FA16A","#B4C84A","#8FAE4A","#D6C15C","#5F7F8F","#9EC3D1","#243F2E","#6C7A63"]
    if sid == "npc":    return "#C97A3A"
    if sid == "unused": return "#3A4248"
    try:
        idx = int(str(sid).replace('sp', ''))
    except Exception:
        idx = 0
    return pal[idx % len(pal)]


def cell_arrow(trend: str | None) -> str:
    return {"up": "↑", "down": "↓"}.get(trend or "", "→")


def render_fill_bar(fill: dict | None):
    if not fill or not isinstance(fill, dict):
        return
    segs = fill.get('segments') or []
    parts = []
    for seg in segs:
        sid = str(seg.get('id'))
        share = float(seg.get('share', 0.0))
        if share <= 1e-6:
            continue
        w = max(0.0, min(100.0, 100.0 * share))
        col = species_color(sid)
        parts.append(f"<div title='{sid}:{share*100:.0f}%' style='width:{w:.2f}%;height:6px;background:{col};'></div>")
    if not parts:
        return
    st.markdown(
        "<div style='display:flex;width:100%;height:6px;border-radius:4px;overflow:hidden;background:#232629;'>"
        + "".join(parts) + "</div>",
        unsafe_allow_html=True,
    )


# ── Session helpers ──────────────────────────────────────────────────────────

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
    save_path, _ = init_new_run(
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
    try:
        save_path = st.session_state['save_path']
        save_path, _ = simulate_tick(save_path, action=action or {}, out_dir=RUNS_DIR)
        st.session_state['save_path'] = save_path
        st.session_state['last_error'] = None
    except Exception as e:
        st.session_state['last_error'] = str(e)


# ── Renderers ────────────────────────────────────────────────────────────────

def render_hud(snapshot: dict):
    hud  = snapshot.get('hud', {}) or {}
    econ = snapshot.get('economy', {}) or {}
    active = st.session_state.get('active_species', 'sp0')
    icon   = SAVANNA_PLAYER_ICONS.get(active, '🧬')
    dom    = float(hud.get('dominance') or 0.0)
    trait_c = fmt((econ.get('trait_costs') or {}).get(active))
    fork_c  = fmt(econ.get('fork_cost'))

    c1, c2, c3, c4, c5 = st.columns([1, 1.4, 1.4, 1.4, 3])
    c1.metric('Tick', hud.get('tick', '-'))
    c2.metric('Evo',  fmt(hud.get('evo_balance')))
    c3.metric('Pop',  fmt(hud.get('total_population'), 1))
    c4.metric('Dom',  f"{dom*100:.1f}%")
    c5.caption(
        f"{icon} **{active}** · trait {trait_c} · fork {fork_c}  \n"
        f"NPC {fmt(hud.get('npc_total'), 1)} · passive +{fmt(hud.get('passive_preview_next'))}"
    )
    st.progress(max(0.0, min(1.0, dom)))

    ae = snapshot.get('active_effects') or []
    if ae:
        parts = []
        for e in ae:
            em = SAVANNA_RESOURCE_UI.get(e.get('res', ''), {}).get('emoji', '')
            parts.append(f"{e.get('cell')} {em}×{float(e.get('mult',1)):.2f}({e.get('ttl')}t)")
        st.caption('⚡ ' + ' · '.join(parts))


def render_map(snapshot: dict):
    m           = snapshot.get('map', {}) or {}
    cells       = m.get('cells', {}) or {}
    species_cells = snapshot.get('species_cells', {}) or {}
    hud         = snapshot.get('hud', {}) or {}

    grid = [['c00','c01','c02'],['c10','c11','c12'],['c20','c21','c22']]

    for r in range(3):
        cols = st.columns(3)
        for c in range(3):
            cid  = grid[r][c]
            info = cells.get(cid, {}) or {}
            is_scouted = bool(info.get('is_scouted', False))
            has_player = bool(info.get('has_player'))

            if not is_scouted:
                label = f"{cid}\n🌫️ fog of war"
            else:
                arch_key  = info.get('archetype')
                arch_name = SAVANNA_ARCHETYPES.get(arch_key, {}).get('ui_name', cid) if arch_key else cid
                event_flag = ' ⚡' if info.get('has_event') else ''

                # NPC line: emoji + rounded pop for each species
                npc_parts = [
                    f"{n['emoji']}{n['pop']:.0f}"
                    for n in (info.get('npc_summary') or [])
                ]
                npc_line = ' '.join(npc_parts) if npc_parts else '—'

                # Resources line: use effective values during events, base otherwise
                res_base = info.get('resources') or {}
                res = info.get('resources_effective') if info.get('has_event') else res_base
                res = res or res_base
                res_parts = [
                    f"{SAVANNA_RESOURCE_UI.get(k,{}).get('emoji',k)}{int(v)}"
                    for k, v in res.items() if v > 0
                ]
                res_line = ' '.join(res_parts)

                # Player lines: one per species that has pop > 0 on this cell
                player_parts = []
                for sid, sc in species_cells.items():
                    cell_data = (sc or {}).get(cid)
                    if not cell_data:
                        continue
                    pop = float((cell_data or {}).get('population', 0.0))
                    if pop < 0.05:
                        continue
                    icon = SAVANNA_PLAYER_ICONS.get(sid, '🧬')
                    arrow = cell_arrow(cell_data.get('trend', 'flat'))
                    player_parts.append(f"{icon}{pop:.1f}{arrow}")
                player_line = ' '.join(player_parts)

                lines = [arch_name + event_flag, npc_line]
                if res_line:
                    lines.append(res_line)
                if player_line:
                    lines.append(player_line)
                label = '\n'.join(lines)

            is_sel   = (st.session_state.get('selected_cell') == cid)
            disabled = (not is_scouted) and (not has_player)

            if cols[c].button(
                label,
                use_container_width=True,
                type='primary' if is_sel else 'secondary',
                disabled=disabled,
                key=f"map_{cid}",
            ):
                st.session_state['selected_cell'] = cid

            if is_scouted:
                with cols[c]:
                    render_fill_bar(info.get('fill'))


def render_cell(snapshot: dict):
    selected  = snapshot.get('selected_cell') or {}
    inspector = selected.get('inspector') or {}
    passport  = selected.get('passport') or {}
    cid       = selected.get('cell_id', st.session_state.get('selected_cell'))

    arch_key  = passport.get('archetype')
    arch_name = SAVANNA_ARCHETYPES.get(arch_key, {}).get('ui_name', '') if arch_key else ''
    richness  = passport.get('richness', '')

    header = f"**{cid}**"
    if arch_name:
        header += f" — {arch_name}"
    if richness:
        header += f" *({richness})*"
    st.markdown(header)

    # ── Active event ──────────────────────────────
    detailed = passport.get('active_effects_detailed') or []
    if detailed:
        for e in detailed:
            res_em = SAVANNA_RESOURCE_UI.get(e.get('res',''), {}).get('emoji', e.get('res',''))
            res_nm = SAVANNA_RESOURCE_UI.get(e.get('res',''), {}).get('ui_name', e.get('res',''))
            st.warning(
                f"⚡ **Event** — {res_em} {res_nm} ×{float(e.get('mult',1)):.2f} "
                f"({e.get('ttl')} tick(s) left, type: {e.get('type','')})"
            )

    # ── Resources (base vs effective) ─────────────
    raw_res = passport.get('resources') or {}
    eff_res = passport.get('resources_effective') or {}
    if raw_res:
        parts = []
        for k, v in raw_res.items():
            em  = SAVANNA_RESOURCE_UI.get(k, {}).get('emoji', k)
            nm  = SAVANNA_RESOURCE_UI.get(k, {}).get('ui_name', k)
            ev2 = float(eff_res.get(k, v))
            if abs(ev2 - float(v)) > 0.5:
                parts.append(f"{em} {nm}: {float(v):.0f} → **{ev2:.0f}**")
            else:
                parts.append(f"{em} {nm}: {float(v):.0f}")
        st.caption(' · '.join(parts))

    # ── NPC species ───────────────────────────────
    npc_list = [n for n in (passport.get('npc_species') or []) if float((n or {}).get('pop', 0)) >= 0.5]
    if npc_list:
        st.markdown('**NPC species**')
        for npc in npc_list:
            stype = npc.get('species_type') if isinstance(npc, dict) else None
            sd = SAVANNA_SPECIES.get(stype) if stype else None
            pop = float(npc.get('pop', 0)) if isinstance(npc, dict) else 0.0
            r   = float(npc.get('r', 0))   if isinstance(npc, dict) else 0.0
            b   = npc.get('b', {})          if isinstance(npc, dict) else {}
            fk  = npc.get('focus', 'R1')    if isinstance(npc, dict) else 'R1'
            if sd:
                b_str = ' '.join(
                    f"{SAVANNA_RESOURCE_UI.get(rk,{}).get('emoji',rk)}{rv:.1f}"
                    for rk, rv in b.items() if rv > 0
                )
                st.write(f"{sd['emoji']} **{sd['ui_name']}** pop {pop:.1f} · r={r:.2f} · {b_str}")
            else:
                st.write(f"? pop {pop:.1f} · r={r:.2f}")

    # ── Player species ────────────────────────────
    pl_list = [p for p in (passport.get('player_species') or []) if float((p or {}).get('pop', 0)) >= 0.05]
    if pl_list:
        st.markdown('**Player species**')
        for ps in pl_list:
            sid  = ps.get('species_id', '?')
            icon = SAVANNA_PLAYER_ICONS.get(sid, '🧬')
            pop  = float(ps.get('pop', 0))
            r    = float(ps.get('r', 0))
            b    = ps.get('b', {})
            b_str = ' '.join(
                f"{SAVANNA_RESOURCE_UI.get(rk,{}).get('emoji',rk)}{rv:.1f}"
                for rk, rv in b.items() if rv > 0
            )
            st.write(f"{icon} **{sid}** pop {pop:.1f} · r={r:.2f} · {b_str}")

    # ── Debug: last-tick breakdown ────────────────
    breakdown = passport.get('consumption_breakdown')
    ev = snapshot.get('last_events') or []
    rv = snapshot.get('last_reveals') or []
    if breakdown or ev or rv:
        with st.expander('Debug / Last tick', expanded=False):
            if breakdown and isinstance(breakdown, dict):
                cap = fmt(breakdown.get('cell_capacity_total'), 1)
                st.caption(f'**Cell capacity:** {cap}')
                for row in (breakdown.get('rows') or []):
                    rid   = row.get('id', '?')
                    kind  = row.get('kind', '')
                    food  = fmt(row.get('food_eaten'), 1)
                    surv  = fmt(row.get('survivors'), 1)
                    share = row.get('share_of_capacity', 0.0)
                    icon  = SAVANNA_PLAYER_ICONS.get(rid, '🧬') if kind == 'player' else '🐾'
                    st.caption(f"  {icon} {rid} ({kind}): food={food} surv={surv} share={share*100:.1f}%")
            for e in ev:
                st.caption(str(e))
            if rv:
                st.caption(f'{len(rv)} cell(s) revealed')


# ── Sidebar ───────────────────────────────────────────────────────────────────

ensure_state()

with st.sidebar.expander('💾 Game', expanded=True):
    os.makedirs(RUNS_DIR, exist_ok=True)
    saves = list_saves(RUNS_DIR)
    seed_str = ''
    if saves:
        current = st.session_state.get('save_path')
        idx = saves.index(current) if current in saves else 0
        chosen = st.selectbox('Save', saves, index=idx,
                              format_func=lambda p: os.path.basename(p))
        if st.button('📂 Load', use_container_width=True):
            do_load_game(chosen)
            st.rerun()
    else:
        st.caption('No saves yet.')
    seed_str = st.text_input('Seed (optional int)', value='', label_visibility='visible')
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
    st.info('Load a save or start a New Game.')
    st.stop()

snap = safe_snapshot()
if not snap:
    st.error('Snapshot is empty.')
    st.stop()

species_ids = [sp.get('species_id') for sp in (snap.get('species') or []) if sp.get('species_id')]
if species_ids and st.session_state.get('active_species') not in species_ids:
    st.session_state['active_species'] = species_ids[0]

eco       = snap.get('economy', {}) or {}
evo_bal   = float(eco.get('evo_balance') or 0.0)

with st.sidebar.expander('⚙️ Actions', expanded=True):
    def _sp_label(sid):
        return f"{SAVANNA_PLAYER_ICONS.get(sid,'🧬')} {sid}"

    active = st.selectbox(
        'Species', species_ids,
        index=species_ids.index(st.session_state.get('active_species','sp0')) if species_ids else 0,
        format_func=_sp_label,
    )
    st.session_state['active_species'] = active

    if st.button('⏭️ Next tick', use_container_width=True):
        do_tick()
        st.rerun()

    trait_cost = float((eco.get('trait_costs') or {}).get(active) or 0.0)
    st.caption(f'Buy trait — {fmt(trait_cost)} evo')
    tc = st.columns(4)
    for i, (key, lbl) in enumerate([('r','🔴r'),('b1','🌱b1'),('b2','🌿b2'),('b3','🥔b3')]):
        with tc[i]:
            if st.button(lbl, use_container_width=True, disabled=(evo_bal < trait_cost), key=f'trait_{key}'):
                do_tick({'buy_trait': (active, key)})
                st.rerun()

with st.sidebar.expander('↗️ Migration', expanded=False):
    active_cells = list((snap.get('species_cells', {}).get(active, {}) or {}).keys())
    if not active_cells:
        st.caption('No population to migrate.')
    else:
        src = st.selectbox('From', active_cells, key='mig_src')
        nbrs = (snap.get('map',{}).get('cells',{}).get(src,{}) or {}).get('neighbors',[]) or []
        dst  = st.selectbox('To',   nbrs,         key='mig_dst')
        max_take = float((snap.get('species_cells',{}).get(active,{}) or {}).get(src,{}).get('population',0.0))
        amt = st.number_input('Amount', min_value=0.0, max_value=max_take, step=1.0, value=0.0, key='mig_amt')
        action = {'migrations': [(active, src, dst, float(amt))]} if amt > 0 else {}
        if st.button('⏭️ Migrate + tick', use_container_width=True):
            do_tick(action)
            st.rerun()

with st.sidebar.expander('🧬 Fork', expanded=False):
    fork_cells = list((snap.get('species_cells', {}).get(active, {}) or {}).keys())
    if not fork_cells:
        st.caption('No population to fork.')
    else:
        fk_src     = st.selectbox('From cell', fork_cells, key='fk_src')
        fk_all     = st.checkbox('Take ALL', value=False, key='fk_all')
        fk_starter = st.selectbox('Starter trait', ['r','b1','b2','b3'], key='fk_starter')
        max_take   = float((snap.get('species_cells',{}).get(active,{}) or {}).get(fk_src,{}).get('population',0.0))
        fk_amt     = st.number_input('Split amount', min_value=0.0, max_value=max_take,
                                     step=1.0, value=min(2.0, max_take), disabled=fk_all, key='fk_amt')
        if st.button('🧬 Fork + tick', use_container_width=True):
            do_tick({'fork': (active, 'ALL' if fk_all else float(fk_amt), fk_starter, fk_src)})
            st.rerun()

if st.session_state.get('last_error'):
    st.sidebar.error(st.session_state['last_error'])

# ── Main layout ───────────────────────────────────────────────────────────────

render_hud(snap)

left, right = st.columns([1.7, 0.3])
with left:
    render_map(snap)
with right:
    render_cell(snap)
