import json
import os
from pathlib import Path

import streamlit as st
from openai import OpenAI


LEVELS_DIR = Path(__file__).parent / "levels"
ACTION_TYPES = ["Observe", "Talk", "Action", "Think"]
DEFAULT_MODEL = "deepseek-ai/DeepSeek-V3.2"
DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"

GLOBAL_CONTEXT = (
    "你是MindCraft思维训练系统的一部分。"
    "世界观：严谨现实逻辑，无魔法。"
    "核心机制：物理胜利必须由正确的思维驱动。"
    "除非用户真正运用了目标思维模型，否则不要空泛表扬。"
    "所有输出必须使用中文。"
)

MODEL_DEFINITIONS = {
    "Sunk Cost Fallacy": (
        "Focus on future expected value, not past investment. "
        "Cut losses when evidence says a project will not deliver, "
        "even if much has been spent already."
    )
}


def resolve_base_url(model_name: str) -> str | None:
    env_base_url = os.getenv("OPENAI_BASE_URL", "").strip()
    if env_base_url:
        return env_base_url
    if model_name.strip():
        return DEFAULT_BASE_URL
    return DEFAULT_BASE_URL


def load_level_configs() -> list[dict]:
    if not LEVELS_DIR.exists():
        return []

    configs: list[dict] = []
    for path in LEVELS_DIR.glob("*.json"):
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            data["_path"] = str(path)
            configs.append(data)
        except (OSError, json.JSONDecodeError):
            continue

    return sorted(configs, key=lambda item: item.get("title", ""))


def reset_game(level_config: dict) -> None:
    st.session_state.history = [{"role": "assistant", "content": level_config["intro"]}]
    st.session_state.game_log = []
    st.session_state.turn_count = 0
    st.session_state.game_over = False
    st.session_state.status = "active"
    st.session_state.mentor_report = None
    st.session_state.last_logic = None
    st.session_state.selected_level_id = level_config["level_id"]


def ensure_game_state(level_config: dict) -> None:
    if "selected_level_id" not in st.session_state:
        reset_game(level_config)
        return

    if "history" not in st.session_state:
        reset_game(level_config)
        return

    if st.session_state.selected_level_id != level_config["level_id"]:
        reset_game(level_config)


def summarize_history(history: list[dict], limit: int = 6) -> str:
    if not history:
        return ""

    lines = []
    for entry in history[-limit:]:
        if entry["role"] == "user":
            action = entry.get("action_type", "Action")
            lines.append(f"User[{action}]: {entry['content']}")
            continue
        lines.append(f"Narrator: {entry['content']}")
    return "\n".join(lines)


def parse_logic_json(raw_text: str) -> dict | None:
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            data = json.loads(raw_text[start : end + 1])
        except json.JSONDecodeError:
            return None
    if not isinstance(data, dict):
        return None
    return data


def normalize_logic_result(
    result: dict | None,
    turn_count: int,
    max_turns: int,
) -> dict:
    defaults = {
        "outcome": "neutral",
        "narrative_guidance": "Continue the story with realistic consequences.",
        "hidden_score": 0,
        "game_over": False,
        "status": "active",
        "state_change": "",
        "internal_comment": "",
    }
    if not result:
        return defaults

    merged = {**defaults, **result}
    if merged["outcome"] not in {"success", "fail", "neutral"}:
        merged["outcome"] = "neutral"
    if merged["status"] not in {"active", "won", "lost"}:
        merged["status"] = "active"
    if not isinstance(merged["hidden_score"], int):
        try:
            merged["hidden_score"] = int(merged["hidden_score"])
        except (ValueError, TypeError):
            merged["hidden_score"] = 0
    merged["game_over"] = bool(merged["game_over"])

    if turn_count >= max_turns and not merged["game_over"]:
        merged["game_over"] = True
        merged["status"] = "lost"
        merged["narrative_guidance"] = (
            f"{merged['narrative_guidance']} "
            "The game ends because the maximum turns were reached."
        )
        merged["internal_comment"] = (
            f"{merged['internal_comment']} | Forced game over by max turns."
        ).strip(" |")

    if merged["game_over"] and merged["status"] == "active":
        merged["status"] = "won" if merged["outcome"] == "success" else "lost"
    return merged


def call_chat_completion(client: OpenAI, model: str, messages: list[dict], temperature: float) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()


def build_logic_prompt(level_config: dict) -> str:
    model_definition = MODEL_DEFINITIONS.get(
        level_config["target_model"],
        "Apply the specified thinking model carefully.",
    )
    return (
        f"{GLOBAL_CONTEXT}\n"
        "角色：你是逻辑判官。不要写故事，只输出JSON。\n"
        "关卡信息：\n"
        f"- 标题：{level_config['title']}\n"
        f"- 开场：{level_config['intro']}\n"
        f"- 目标模型：{level_config['target_model']}\n"
        f"- 模型定义：{model_definition}\n"
        f"- 胜利条件：{level_config['victory_condition']}\n"
        f"- 最大回合：{level_config['max_turns']}\n"
        "任务：\n"
        "1) 分析用户意图。\n"
        "2) 判断是否符合现实物理规则。\n"
        "3) 判断是否符合目标思维模型，尤其是Think动作。\n"
        "4) 更新游戏状态并给叙事者提示。\n"
        "只返回JSON，字段如下：\n"
        "- outcome: success|fail|neutral\n"
        "- narrative_guidance: 给叙事者的中文提示\n"
        "- hidden_score: 整数 -10..10\n"
        "- game_over: 布尔值\n"
        "- status: active|won|lost\n"
        "- state_change: 简短状态变化\n"
        "- internal_comment: 简短评估备注\n"
        "只输出JSON，不要额外文本。"
    )


def build_narrator_prompt() -> str:
    return (
        f"{GLOBAL_CONTEXT}\n"
        "角色：你是叙事者。把逻辑提示渲染成有代入感的故事。\n"
        "规则：\n"
        "- 使用第二人称。\n"
        "- 1-2段短段落。\n"
        "- 严格遵循逻辑结果和提示。\n"
        "- 可以合理推进时间（如几天或几个月后）。\n"
        "- 必须以明确的下一步困境或问题收尾，引导玩家行动。\n"
        "- 不要暴露JSON或内部评估备注。\n"
        "- 若游戏结束，用收束句收尾，不要继续提问。"
    )


def build_mentor_prompt() -> str:
    return (
        f"{GLOBAL_CONTEXT}\n"
        "角色：你是影子导师。用Markdown写复盘分析。\n"
        "必须包含以下小节：\n"
        "1) 总结\n"
        "2) 模型运用\n"
        "3) 遗漏机会\n"
        "4) 下一步建议\n"
        "要求直率、具体，引用用户的关键决策。"
    )


def process_turn(
    client: OpenAI,
    model: str,
    level_config: dict,
    action_type: str,
    user_input: str,
) -> None:
    next_turn = st.session_state.turn_count + 1
    history_summary = summarize_history(st.session_state.history)
    logic_payload = {
        "action_type": action_type,
        "user_input": user_input,
        "turn": next_turn,
        "max_turns": level_config["max_turns"],
        "current_status": st.session_state.status,
        "victory_condition": level_config["victory_condition"],
        "recent_history": history_summary,
    }

    logic_messages = [
        {"role": "system", "content": build_logic_prompt(level_config)},
        {"role": "user", "content": json.dumps(logic_payload, ensure_ascii=False)},
    ]
    st.session_state.last_debug = {
        "logic": {"messages": logic_messages, "response": None},
        "narrator": {"messages": None, "response": None},
        "mentor": {"messages": None, "response": None},
    }

    try:
        with st.spinner("Logic Engine is thinking..."):
            logic_raw = call_chat_completion(
                client,
                model,
                logic_messages,
                temperature=0.2,
            )
        st.session_state.last_debug["logic"]["response"] = logic_raw
    except Exception:
        st.error("逻辑判官请求失败，请检查API Key、模型或Base URL。")
        st.session_state.last_debug["logic"]["response"] = "ERROR"
        return

    logic_result = normalize_logic_result(
        parse_logic_json(logic_raw),
        turn_count=next_turn,
        max_turns=level_config["max_turns"],
    )
    st.session_state.last_logic = {"raw": logic_raw, "parsed": logic_result}
    st.session_state.game_log.append(
        {
            "turn": next_turn,
            "action_type": action_type,
            "user_input": user_input,
            "logic_raw": logic_raw,
            "logic_result": logic_result,
        }
    )

    st.session_state.history.append(
        {"role": "user", "content": user_input, "action_type": action_type}
    )

    narrator_messages = [
        {"role": "system", "content": build_narrator_prompt()},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "action_type": action_type,
                    "user_input": user_input,
                    "logic_result": logic_result,
                    "turn": next_turn,
                },
                ensure_ascii=False,
            ),
        },
    ]
    st.session_state.last_debug["narrator"]["messages"] = narrator_messages

    try:
        with st.spinner("Narrator is writing..."):
            narrative_text = call_chat_completion(
                client,
                model,
                narrator_messages,
                temperature=0.7,
            )
        st.session_state.last_debug["narrator"]["response"] = narrative_text
    except Exception:
        st.error("叙事者请求失败，请稍后重试。")
        narrative_text = "叙事者暂时不可用。"
        st.session_state.last_debug["narrator"]["response"] = "ERROR"

    st.session_state.history.append({"role": "assistant", "content": narrative_text})
    st.session_state.turn_count = next_turn
    st.session_state.game_over = bool(logic_result["game_over"])
    st.session_state.status = logic_result["status"]

    if not st.session_state.game_over:
        return

    mentor_messages = [
        {"role": "system", "content": build_mentor_prompt()},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "level": level_config,
                    "final_status": st.session_state.status,
                    "history": st.session_state.history,
                    "game_log": st.session_state.game_log,
                },
                ensure_ascii=False,
            ),
        },
    ]
    st.session_state.last_debug["mentor"]["messages"] = mentor_messages

    try:
        with st.spinner("Shadow Mentor is reflecting..."):
            mentor_text = call_chat_completion(
                client,
                model,
                mentor_messages,
                temperature=0.4,
            )
        st.session_state.mentor_report = mentor_text
        st.session_state.last_debug["mentor"]["response"] = mentor_text
    except Exception:
        st.error("影子导师请求失败，请稍后重试。")
        st.session_state.mentor_report = "影子导师暂时不可用。"
        st.session_state.last_debug["mentor"]["response"] = "ERROR"


st.set_page_config(page_title="MindCraft", layout="wide")
st.title("MindCraft - Thinking Model Training Ground")

level_configs = load_level_configs()
if not level_configs:
    st.error("No level configs found. Add JSON files under ./levels to start.")
    st.stop()

with st.sidebar:
    st.header("MindCraft")
    default_api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY", "")
    openai_api_key = st.text_input(
        "API Key",
        type="password",
        value=default_api_key,
    )
    model_name = st.text_input("Model", value=DEFAULT_MODEL)
    st.caption(
        "Base URL uses OPENAI_BASE_URL if set. Default: "
        f"{DEFAULT_BASE_URL}."
    )
    level_titles = [
        f"{config['title']} - {config['target_model']}" for config in level_configs
    ]
    selected_index = st.selectbox(
        "Level Selector",
        options=list(range(len(level_configs))),
        format_func=lambda idx: level_titles[idx],
    )
    debug_mode = st.toggle("Debug Mode", value=False)
    restart_clicked = st.button("Restart Game")

selected_level = level_configs[selected_index]
ensure_game_state(selected_level)
if restart_clicked:
    reset_game(selected_level)
    st.rerun()

if not openai_api_key:
    st.info("Enter your API key to start playing.")

turn_info = f"Turn {st.session_state.turn_count}/{selected_level['max_turns']}"
st.subheader(selected_level["title"])
st.caption(
    f"Target model: {selected_level['target_model']} | "
    f"Victory: {selected_level['victory_condition']} | {turn_info}"
)

if st.session_state.game_over:
    if st.session_state.status == "won":
        st.success("Game Over: Victory")
    else:
        st.error("Game Over: Defeat")

for entry in st.session_state.history:
    if entry["role"] == "user":
        with st.chat_message("user"):
            action_label = entry.get("action_type", "Action")
            st.markdown(f"**{action_label}**\n\n{entry['content']}")
        continue
    with st.chat_message("assistant"):
        st.markdown(entry["content"])

with st.form("action_form", clear_on_submit=True):
    action_type = st.radio("Action Type", options=ACTION_TYPES, horizontal=True)
    user_input = st.text_area(
        "Your instruction",
        placeholder="Describe your action or reasoning...",
        height=120,
    )
    send_disabled = st.session_state.game_over
    submitted = st.form_submit_button("Send", disabled=send_disabled)

if submitted:
    if not openai_api_key:
        st.warning("Please provide an API key before sending.")
        st.stop()
    if not user_input.strip():
        st.warning("Please enter a valid instruction.")
        st.stop()

    normalized_model = model_name.strip() or DEFAULT_MODEL
    client_kwargs = {"api_key": openai_api_key}
    resolved_base_url = resolve_base_url(normalized_model)
    if resolved_base_url:
        client_kwargs["base_url"] = resolved_base_url
    client = OpenAI(**client_kwargs)

    process_turn(
        client=client,
        model=normalized_model,
        level_config=selected_level,
        action_type=action_type,
        user_input=user_input.strip(),
    )
    st.rerun()

if debug_mode:
    if st.session_state.get("last_logic"):
        with st.expander("Logic Engine Output (Debug)", expanded=False):
            st.code(st.session_state.last_logic["raw"], language="json")
            st.json(st.session_state.last_logic["parsed"])

    debug_payloads = st.session_state.get("last_debug")
    if debug_payloads:
        with st.expander("Logic Engine Prompts + Response", expanded=False):
            st.json(debug_payloads["logic"]["messages"])
            st.code(debug_payloads["logic"]["response"] or "", language="text")

        with st.expander("Narrator Prompts + Response", expanded=False):
            st.json(debug_payloads["narrator"]["messages"])
            st.code(debug_payloads["narrator"]["response"] or "", language="text")

        with st.expander("Shadow Mentor Prompts + Response", expanded=False):
            st.json(debug_payloads["mentor"]["messages"])
            st.code(debug_payloads["mentor"]["response"] or "", language="text")

if st.session_state.game_over and st.session_state.mentor_report:
    st.markdown("---")
    st.subheader("Shadow Mentor Report")
    st.markdown(st.session_state.mentor_report)
