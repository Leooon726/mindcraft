import json
import os
import re
from datetime import datetime
from pathlib import Path

import streamlit as st
from openai import OpenAI


LEVELS_DIR = Path(__file__).parent / "levels"
ACTION_TYPES = ["Observe", "Talk", "Action", "Think"]
DEFAULT_MODEL = "deepseek-ai/DeepSeek-V3.2"
DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"
ENTITY_KEYS = ["people", "projects", "locations", "organizations", "assets"]

GLOBAL_CONTEXT = (
    "你是MindCraft思维训练系统的一部分。"
    "世界观：严谨现实逻辑，无魔法。"
    "核心机制：物理胜利必须由正确的思维驱动。"
    "除非用户真正运用了目标思维模型，否则不要空泛表扬。"
    "所有输出必须使用中文。"
)

MODEL_DEFINITIONS = {
    "Sunk Cost Fallacy": {
        "definition": (
            "沉没成本谬误指人在决策时过度受已投入成本影响，"
            "把过去的投入当作继续投入的理由。正确做法是只看未来的"
            "预期价值与风险。"
        ),
        "example": (
            "一款硬件已研发两年却迟迟无法量产，团队提出再投200万"
            "也许能解决问题。你需要评估新增投入是否值得，而不是"
            "因为已投入大量资源就继续加码。"
        ),
        "pitfall": (
            "把“已经花了这么多”当成继续投入的依据；"
            "把止损等同于否定过去努力；"
            "忽视机会成本与替代方案。"
        ),
        "advice": (
            "使用前瞻指标评估后续投入的边际收益；"
            "设立止损线与阶段性验收；"
            "区分情感承诺与商业回报。"
        ),
    }
}


def get_model_explanation(level_config: dict):
    explanation = level_config.get("model_explanation")
    if explanation:
        return explanation
    return MODEL_DEFINITIONS.get(level_config.get("target_model", ""))


def format_model_explanation(explanation) -> str:
    if isinstance(explanation, dict):
        parts = []
        definition = explanation.get("definition", "").strip()
        example = explanation.get("example", "").strip()
        pitfall = explanation.get("pitfall", "").strip()
        advice = explanation.get("advice", "").strip()
        if definition:
            parts.append(f"定义：{definition}")
        if example:
            parts.append(f"简单案例：{example}")
        if pitfall:
            parts.append(f"常见误区：{pitfall}")
        if advice:
            parts.append(f"应用建议：{advice}")
        return "\n".join(parts)
    if isinstance(explanation, str):
        return explanation.strip()
    return ""


def render_model_explanation(explanation) -> None:
    if isinstance(explanation, dict):
        definition = explanation.get("definition", "").strip()
        example = explanation.get("example", "").strip()
        pitfall = explanation.get("pitfall", "").strip()
        advice = explanation.get("advice", "").strip()
        if definition:
            st.markdown(f"**定义**：{definition}")
        if example:
            st.markdown(f"**简单案例**：{example}")
        if pitfall:
            st.markdown(f"**常见误区**：{pitfall}")
        if advice:
            st.markdown(f"**应用建议**：{advice}")
        return
    if isinstance(explanation, str) and explanation.strip():
        st.markdown(explanation)
        return
    st.markdown("暂无说明。")


def format_setting(setting: dict) -> str:
    if not isinstance(setting, dict):
        return ""
    parts = []
    scene = setting.get("scene", "").strip()
    background = setting.get("background", "").strip()
    roles = setting.get("roles", "").strip()
    era = setting.get("era", "").strip()
    extra = setting.get("extra_requirements", "").strip()
    if scene:
        parts.append(f"场景：{scene}")
    if background:
        parts.append(f"背景：{background}")
    if roles:
        parts.append(f"角色：{roles}")
    if era:
        parts.append(f"时代：{era}")
    if extra:
        parts.append(f"要求：{extra}")
    return "；".join(parts)


def slugify_level_id(value: str) -> str:
    base = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    if not base:
        base = "level"
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    return f"{base}_{timestamp}"


def ensure_unique_level_id(base_id: str) -> str:
    candidate = base_id
    counter = 1
    while (LEVELS_DIR / f"{candidate}.json").exists():
        candidate = f"{base_id}_{counter}"
        counter += 1
    return candidate


def build_level_builder_prompt() -> str:
    return (
        f"{GLOBAL_CONTEXT}\n"
        "角色：你是MindCraft的关卡设计师。根据输入的思维模型与设定生成关卡JSON。\n"
        "必须输出严格JSON，不要额外文字。\n"
        "JSON结构：\n"
        "{\n"
        '  "title": "...",\n'
        '  "intro": "...",\n'
        '  "victory_condition": "...",\n'
        '  "max_turns": 10,\n'
        '  "model_explanation": {\n'
        '    "definition": "...",\n'
        '    "example": "...",\n'
        '    "pitfall": "...",\n'
        '    "advice": "..." \n'
        "  },\n"
        '  "setting": {\n'
        '    "scene": "...",\n'
        '    "background": "...",\n'
        '    "roles": "...",\n'
        '    "era": "...",\n'
        '    "extra_requirements": "..." \n'
        "  }\n"
        "}\n"
        "要求：\n"
        "- 全中文输出，命名具体，禁止使用“项目A/B”等抽象代号。\n"
        "- intro使用第二人称，包含具体项目名、人物关系与业务数据。\n"
        "- 引入现实复杂性、认知迷雾与人际摩擦。\n"
        "- model_explanation每项2-4句，包含一个简单案例。\n"
        "- victory_condition要可判定但不要直接照抄用户输入。\n"
    )


def parse_level_json(raw_text: str) -> dict | None:
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


def merge_setting(user_setting: dict, generated_setting: dict) -> dict:
    merged = {}
    for key in ["scene", "background", "roles", "era", "extra_requirements"]:
        value = ""
        if isinstance(generated_setting, dict):
            value = str(generated_setting.get(key, "")).strip()
        if not value:
            value = str(user_setting.get(key, "")).strip()
        merged[key] = value
    return merged


def normalize_level_config(
    raw_level: dict,
    target_model: str,
    title_hint: str,
    max_turns: int,
    user_setting: dict,
) -> dict:
    title = str(raw_level.get("title", "")).strip() or title_hint.strip()
    if not title:
        title = f"{target_model}训练关"

    intro = str(raw_level.get("intro", "")).strip()
    if not intro:
        intro = "你面对一个需要谨慎决策的商业局面。"

    victory_condition = str(raw_level.get("victory_condition", "")).strip()
    if not victory_condition:
        victory_condition = "依据目标思维模型做出关键决策并控制风险。"

    model_explanation = raw_level.get("model_explanation", {})
    if not isinstance(model_explanation, (dict, str)):
        model_explanation = {}

    setting = merge_setting(user_setting, raw_level.get("setting", {}))
    level_id = ensure_unique_level_id(
        slugify_level_id(title or target_model or "level")
    )

    return {
        "level_id": level_id,
        "title": title,
        "intro": intro,
        "target_model": target_model,
        "victory_condition": victory_condition,
        "max_turns": max_turns,
        "model_explanation": model_explanation,
        "setting": setting,
    }


def generate_level_config(
    client: OpenAI,
    model: str,
    target_model: str,
    title_hint: str,
    user_setting: dict,
    max_turns: int,
) -> tuple[dict | None, str]:
    payload = {
        "target_model": target_model,
        "title_hint": title_hint,
        "setting": user_setting,
        "max_turns": max_turns,
    }
    messages = [
        {"role": "system", "content": build_level_builder_prompt()},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
    raw = call_chat_completion(client, model, messages, temperature=0.6)
    parsed = parse_level_json(raw)
    if not parsed:
        return None, raw
    return (
        normalize_level_config(
            parsed,
            target_model=target_model,
            title_hint=title_hint,
            max_turns=max_turns,
            user_setting=user_setting,
        ),
        raw,
    )


def save_level_config(level_config: dict) -> Path:
    LEVELS_DIR.mkdir(parents=True, exist_ok=True)
    path = LEVELS_DIR / f"{level_config['level_id']}.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(level_config, handle, ensure_ascii=False, indent=2)
    return path


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
            if "level_id" not in data:
                data["level_id"] = path.stem
            if "max_turns" not in data:
                data["max_turns"] = 10
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
    st.session_state.entity_bank = normalize_entity_bank(
        level_config.get("entities", {})
    )
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
            content = entry.get("normalized_content", entry.get("content", ""))
            lines.append(f"User[{action}]: {content}")
            continue
        lines.append(f"Narrator: {entry['content']}")
    return "\n".join(lines)


def normalize_entity_list(items) -> list[str]:
    if isinstance(items, str):
        items = [items]
    if not isinstance(items, list):
        return []
    seen = set()
    normalized: list[str] = []
    for item in items:
        if not isinstance(item, str):
            continue
        value = item.strip()
        if not value or value in seen:
            continue
        seen.add(value)
        normalized.append(value)
    return normalized


def normalize_entity_bank(bank) -> dict:
    normalized = {key: [] for key in ENTITY_KEYS}
    if not isinstance(bank, dict):
        return normalized
    for key in ENTITY_KEYS:
        normalized[key] = normalize_entity_list(bank.get(key, []))
    return normalized


def merge_entity_banks(base: dict, updates: dict) -> dict:
    merged = normalize_entity_bank(base)
    normalized_updates = normalize_entity_bank(updates)
    for key in ENTITY_KEYS:
        for item in normalized_updates[key]:
            if item not in merged[key]:
                merged[key].append(item)
    return merged


def format_entity_bank(bank: dict) -> str:
    labels = {
        "people": "人物",
        "projects": "项目",
        "locations": "地点",
        "organizations": "组织",
        "assets": "资产/产品",
    }
    lines = []
    for key in ENTITY_KEYS:
        items = bank.get(key, [])
        if items:
            lines.append(f"{labels[key]}：{', '.join(items)}")
    return "\n".join(lines) if lines else "暂无"


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
        "narrative_guidance": "请用现实后果推进剧情。",
        "hidden_score": 0,
        "game_over": False,
        "status": "active",
        "state_change": "",
        "internal_comment": "",
        "normalized_input": "",
        "corrections": [],
        "entities": {},
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
    model_explanation = format_model_explanation(get_model_explanation(level_config))
    if not model_explanation:
        model_explanation = "请根据目标思维模型做出合理判定。"
    setting_text = format_setting(level_config.get("setting", {}))
    return (
        f"{GLOBAL_CONTEXT}\n"
        "角色：你是逻辑判官。不要写故事，只输出JSON。\n"
        "关卡信息：\n"
        f"- 标题：{level_config['title']}\n"
        f"- 开场：{level_config['intro']}\n"
        f"- 目标模型：{level_config['target_model']}\n"
        f"- 模型解释：{model_explanation}\n"
        f"- 胜利条件：{level_config['victory_condition']}\n"
        f"- 最大回合：{level_config['max_turns']}\n"
        f"- 补充设定：{setting_text}\n"
        "任务：\n"
        "1) 分析用户意图。\n"
        "2) 判断是否符合现实物理规则。\n"
        "3) 判断是否符合目标思维模型，尤其是Think动作。\n"
        "4) 更新游戏状态并给叙事者提示。\n"
        "5) 识别错别字或误拼写，推断用户本意，并给出修正结果。\n"
        "   - 如果不确定，保留原词但给出可能的备选。\n"
        "6) 实体一致性：基于提供的entity_bank复用已有名称，\n"
        "   仅在必要时新增，并在输出的entities中给出最新实体表。\n"
        "HARD_MODE_JUDGMENT（高难度判定）：\n"
        "1) 区分意图与行动：\n"
        "- 用户只说“砍掉项目/转向X”且缺乏沟通策略 -> 结果不应直接成功。\n"
        "- 用户先用[观察]/[对话]收集信息，再用[思考]分析，最后提出处理人际摩擦的行动 -> 可以判定成功。\n"
        "2) 惩罚表面化决策：\n"
        "- 如果用户没有索取关键细节就做重大决策，加入隐藏陷阱或反噬。\n"
        "3) 人际摩擦必须存在：\n"
        "- 任何商业决策都要有人的代价（核心成员、投资人、舆论等）。\n"
        "- 判定时检查用户是否同时处理了这些人际后果。\n"
        "只返回JSON，字段如下：\n"
        "- outcome: success|fail|neutral\n"
        "- narrative_guidance: 给叙事者的中文提示\n"
        "- hidden_score: 整数 -10..10\n"
        "- game_over: 布尔值\n"
        "- status: active|won|lost\n"
        "- state_change: 简短状态变化\n"
        "- internal_comment: 简短评估备注\n"
        "- normalized_input: 纠正错别字后的用户意图文本\n"
        "- corrections: 列表，每项包含wrong/right/note\n"
        "- entities: 实体表，包含people/projects/locations/organizations/assets\n"
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
        "\nREALISM_PROTOCOL（高保真现实感）：\n"
        "1) 禁止抽象代号：不要使用“项目A/B”“某员工”等，必须使用具体项目名、具体业务数据、具体人物关系。\n"
        "2) 认知迷雾：被放弃的项目必须包含至少两个诱人的正面信号；新机会必须包含风险和不确定性。\n"
        "3) 情感钩子：让失败项目绑上情感/承诺/身份。\n"
        "4) 展示而非说教：不要直接说“沉没成本陷阱”，要用细节呈现。\n"
        "5) 人际摩擦：每个关键决策都要带出人际后果或对抗。\n"
        "\nSTRICT_POV_PROTOCOL（严格视角限制）：\n"
        "1) 禁止上帝视角：不要描述主角无法看到/听到/触碰的宏观事实。\n"
        "2) 信息必须有载体：所有关键事实必须通过报表、对话、邮件、截图等媒介传达。\n"
        "3) 不可靠叙述：只写“现象”，把判断留给玩家（员工可能撒谎，报表可能有误）。\n"
        "4) 视角范围受限：如果主角不在现场，不要描写现场。\n"
        "\nTYPO_NORMALIZATION（错别字纠正）：\n"
        "- 优先使用逻辑判官提供的normalized_input与corrections。\n"
        "- 叙事中不要复述用户的错别字或误拼写。\n"
        "- 对名称或术语使用修正后的写法。"
        "\nENTITY_CONSISTENCY（实体一致性）：\n"
        "- 必须复用提供的entity_bank中的名称。\n"
        "- 未经必要不要改名或新增角色/地点/项目。\n"
        "- 如需新增，必须与既有实体风格一致。"
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
    status_placeholder=None,
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
        "entity_bank": st.session_state.entity_bank,
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
        if status_placeholder:
            status_placeholder.empty()
            with status_placeholder.container():
                with st.spinner("逻辑判官正在思考..."):
                    logic_raw = call_chat_completion(
                        client,
                        model,
                        logic_messages,
                        temperature=0.2,
                    )
        else:
            with st.spinner("逻辑判官正在思考..."):
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
    normalized_input = logic_result.get("normalized_input")
    if not isinstance(normalized_input, str) or not normalized_input.strip():
        normalized_input = user_input
    corrections = logic_result.get("corrections")
    if not isinstance(corrections, list):
        corrections = []
    logic_result["normalized_input"] = normalized_input
    logic_result["corrections"] = corrections
    logic_entities = logic_result.get("entities", {})
    if not isinstance(logic_entities, dict):
        logic_entities = {}
    st.session_state.entity_bank = merge_entity_banks(
        st.session_state.entity_bank,
        logic_entities,
    )
    logic_result["entities"] = st.session_state.entity_bank
    st.session_state.last_logic = {"raw": logic_raw, "parsed": logic_result}
    st.session_state.game_log.append(
        {
            "turn": next_turn,
            "action_type": action_type,
            "user_input": user_input,
            "normalized_input": normalized_input,
            "corrections": corrections,
            "entities": st.session_state.entity_bank,
            "logic_raw": logic_raw,
            "logic_result": logic_result,
        }
    )

    st.session_state.history.append(
        {
            "role": "user",
            "content": user_input,
            "normalized_content": normalized_input,
            "action_type": action_type,
        }
    )

    narrator_messages = [
        {"role": "system", "content": build_narrator_prompt()},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "action_type": action_type,
                    "user_input_raw": user_input,
                    "user_input_normalized": normalized_input,
                    "corrections": corrections,
                    "logic_result": logic_result,
                    "turn": next_turn,
                    "entity_bank": st.session_state.entity_bank,
                },
                ensure_ascii=False,
            ),
        },
    ]
    st.session_state.last_debug["narrator"]["messages"] = narrator_messages

    try:
        if status_placeholder:
            status_placeholder.empty()
            with status_placeholder.container():
                with st.spinner("叙事者正在创作..."):
                    narrative_text = call_chat_completion(
                        client,
                        model,
                        narrator_messages,
                        temperature=0.7,
                    )
        else:
            with st.spinner("叙事者正在创作..."):
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
        if status_placeholder:
            status_placeholder.empty()
            with status_placeholder.container():
                with st.spinner("影子导师正在复盘..."):
                    mentor_text = call_chat_completion(
                        client,
                        model,
                        mentor_messages,
                        temperature=0.4,
                    )
        else:
            with st.spinner("影子导师正在复盘..."):
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

create_submitted = False

pending_level_id = st.session_state.pop("pending_level_id", None)
if pending_level_id:
    for idx, config in enumerate(level_configs):
        if config.get("level_id") == pending_level_id:
            st.session_state["level_selector"] = idx
            break

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
        key="level_selector",
    )
    debug_mode = st.toggle("Debug Mode", value=False)
    restart_clicked = st.button("Restart Game")

    with st.expander("Create New Level", expanded=False):
        with st.form("create_level_form"):
            new_title_hint = st.text_input("关卡标题（可选）")
            new_model_name = st.text_input("思维模型名称（必填）")
            new_scene = st.text_input("场景（可选）")
            new_background = st.text_input("背景（可选）")
            new_roles = st.text_input("角色（可选）")
            new_era = st.text_input("时代（可选）")
            new_requirements = st.text_area("附加要求（可选）", height=120)
            new_max_turns = st.number_input(
                "最大回合数",
                min_value=3,
                max_value=30,
                value=10,
                step=1,
            )
            create_submitted = st.form_submit_button("生成并创建关卡")

if create_submitted:
    if not new_model_name.strip():
        st.warning("请填写思维模型名称。")
        st.stop()
    if not openai_api_key:
        st.warning("请先填写API Key，再创建关卡。")
        st.stop()
    create_model = model_name.strip() or DEFAULT_MODEL
    client_kwargs = {"api_key": openai_api_key}
    resolved_base_url = resolve_base_url(create_model)
    if resolved_base_url:
        client_kwargs["base_url"] = resolved_base_url
    client = OpenAI(**client_kwargs)
    setting = {
        "scene": new_scene.strip(),
        "background": new_background.strip(),
        "roles": new_roles.strip(),
        "era": new_era.strip(),
        "extra_requirements": new_requirements.strip(),
    }
    with st.spinner("正在生成关卡..."):
        generated_level, raw_response = generate_level_config(
            client=client,
            model=create_model,
            target_model=new_model_name.strip(),
            title_hint=new_title_hint.strip(),
            user_setting=setting,
            max_turns=int(new_max_turns),
        )
    if not generated_level:
        st.error("关卡生成失败，请稍后重试。")
        if debug_mode:
            with st.expander("Level Builder Output (Debug)", expanded=False):
                st.code(raw_response, language="json")
        st.stop()
    save_level_config(generated_level)
    st.session_state["pending_level_id"] = generated_level["level_id"]
    st.session_state.selected_level_id = generated_level["level_id"]
    st.success("关卡已创建并写入配置。")
    st.rerun()

selected_level = level_configs[selected_index]
ensure_game_state(selected_level)
if restart_clicked:
    reset_game(selected_level)
    st.rerun()

if not openai_api_key:
    st.info("Enter your API key to start playing.")

turn_info = f"Turn {st.session_state.turn_count}/{selected_level['max_turns']}"
st.subheader(
    f"{selected_level['title']} · {selected_level['target_model']}"
)
st.caption(turn_info)
with st.expander(
    f"思维模型解释：{selected_level['target_model']}",
    expanded=False,
):
    render_model_explanation(get_model_explanation(selected_level))

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

if st.session_state.game_over and st.session_state.mentor_report:
    st.markdown("---")
    st.subheader("Shadow Mentor Report")
    st.markdown(st.session_state.mentor_report)

status_placeholder = st.empty()

with st.form("action_form", clear_on_submit=True):
    action_type = st.radio(
        "Action Type",
        options=ACTION_TYPES,
        horizontal=True,
        label_visibility="collapsed",
    )
    user_input = st.text_area(
        "Your instruction",
        placeholder="请输入你的行动、对话或思考...",
        height=120,
        label_visibility="collapsed",
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
        status_placeholder=status_placeholder,
    )
    st.rerun()

if debug_mode:
    if st.session_state.get("last_logic"):
        with st.expander("Logic Engine Output (Debug)", expanded=False):
            st.code(st.session_state.last_logic["raw"], language="json")
            st.json(st.session_state.last_logic["parsed"])

    debug_payloads = st.session_state.get("last_debug")
    if debug_payloads:
        with st.expander("Entity Bank (Debug)", expanded=False):
            st.text(format_entity_bank(st.session_state.entity_bank))

        with st.expander("Logic Engine Prompts + Response", expanded=False):
            st.json(debug_payloads["logic"]["messages"])
            st.code(debug_payloads["logic"]["response"] or "", language="text")

        with st.expander("Narrator Prompts + Response", expanded=False):
            st.json(debug_payloads["narrator"]["messages"])
            st.code(debug_payloads["narrator"]["response"] or "", language="text")

        with st.expander("Shadow Mentor Prompts + Response", expanded=False):
            st.json(debug_payloads["mentor"]["messages"])
            st.code(debug_payloads["mentor"]["response"] or "", language="text")

