import html
import json
import os
import re
from datetime import datetime
from pathlib import Path

import streamlit as st
from openai import OpenAI


LEVELS_DIR = Path(__file__).parent / "levels"
ACTION_TYPES = ["观察", "对话", "动作", "思考", "提示", "提前结束"]
DEFAULT_MODEL = "deepseek-ai/DeepSeek-V3.2"
DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_CHALLENGE_LIMIT = 3
ENTITY_KEYS = ["people", "projects", "locations", "organizations", "assets"]

GLOBAL_CONTEXT = (
    "你是MindCraft能力训练系统的一部分。"
    "世界观：严谨现实逻辑，无魔法。"
    "核心机制：物理胜利必须由正确的能力/策略运用驱动。"
    "除非用户真正运用了目标能力/策略，否则不要空泛表扬。"
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
        "triggers": [
            "已投入大量时间/资金但回报迟迟不见",
            "出现更优替代方案但情感难以割舍",
            "团队或投资人以“已经投了很多”为由坚持继续",
        ],
        "steps": [
            "区分已沉没成本与未来新增投入",
            "用前瞻指标评估未来收益与风险",
            "设定止损线并沟通执行路径",
        ],
        "success_signals": [
            "决策理由基于未来价值而非过去投入",
            "能清楚说明止损依据与替代方案",
            "资源能被转向更高潜力方向",
        ],
    }
}


def get_model_explanation(level_config: dict):
    explanation = level_config.get("model_explanation")
    if explanation:
        return explanation
    return MODEL_DEFINITIONS.get(level_config.get("target_model", ""))


def parse_target_model_input(raw_text: str) -> tuple[str, str]:
    text = raw_text.strip()
    if not text:
        return "", ""
    if "\n" in text:
        name, hint = text.split("\n", 1)
        return name.strip(), hint.strip()
    for sep in ["：", ":", " - ", " — ", "｜", "|"]:
        if sep in text:
            name, hint = text.split(sep, 1)
            name = name.strip()
            hint = hint.strip()
            if name and hint:
                return name, hint
    return text, ""


def normalize_text_list(value) -> list[str]:
    if isinstance(value, list):
        items = [str(item).strip() for item in value]
        return [item for item in items if item]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def format_model_explanation(explanation) -> str:
    if isinstance(explanation, dict):
        parts = []
        definition = explanation.get("definition", "").strip()
        example = explanation.get("example", "").strip()
        pitfall = explanation.get("pitfall", "").strip()
        advice = explanation.get("advice", "").strip()
        triggers = normalize_text_list(explanation.get("triggers"))
        steps = normalize_text_list(explanation.get("steps"))
        success_signals = normalize_text_list(explanation.get("success_signals"))
        if definition:
            parts.append(f"定义：{definition}")
        if example:
            parts.append(f"简单案例：{example}")
        if pitfall:
            parts.append(f"常见误区：{pitfall}")
        if advice:
            parts.append(f"应用建议：{advice}")
        if triggers:
            parts.append(f"触发场景：{'；'.join(triggers)}")
        if steps:
            parts.append(f"核心步骤：{'；'.join(steps)}")
        if success_signals:
            parts.append(f"成功信号：{'；'.join(success_signals)}")
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
        triggers = normalize_text_list(explanation.get("triggers"))
        steps = normalize_text_list(explanation.get("steps"))
        success_signals = normalize_text_list(explanation.get("success_signals"))
        if definition:
            st.markdown(f"**定义**：{definition}")
        if example:
            st.markdown(f"**简单案例**：{example}")
        if pitfall:
            st.markdown(f"**常见误区**：{pitfall}")
        if advice:
            st.markdown(f"**应用建议**：{advice}")
        if triggers:
            st.markdown("**触发场景**：")
            st.markdown("\n".join([f"- {item}" for item in triggers]))
        if steps:
            st.markdown("**核心步骤**：")
            st.markdown("\n".join([f"- {item}" for item in steps]))
        if success_signals:
            st.markdown("**成功信号**：")
            st.markdown("\n".join([f"- {item}" for item in success_signals]))
        return
    if isinstance(explanation, str) and explanation.strip():
        st.markdown(explanation)
        return
    st.markdown("暂无说明。")


def format_model_explanation_markdown(explanation) -> str:
    if isinstance(explanation, dict):
        parts = []
        definition = explanation.get("definition", "").strip()
        example = explanation.get("example", "").strip()
        pitfall = explanation.get("pitfall", "").strip()
        advice = explanation.get("advice", "").strip()
        triggers = normalize_text_list(explanation.get("triggers"))
        steps = normalize_text_list(explanation.get("steps"))
        success_signals = normalize_text_list(explanation.get("success_signals"))
        if definition:
            parts.append(f"- **定义**：{definition}")
        if example:
            parts.append(f"- **简单案例**：{example}")
        if pitfall:
            parts.append(f"- **常见误区**：{pitfall}")
        if advice:
            parts.append(f"- **应用建议**：{advice}")
        if triggers:
            parts.append(f"- **触发场景**：{'；'.join(triggers)}")
        if steps:
            parts.append(f"- **核心步骤**：{'；'.join(steps)}")
        if success_signals:
            parts.append(f"- **成功信号**：{'；'.join(success_signals)}")
        return "\n".join(parts)
    if isinstance(explanation, str) and explanation.strip():
        return explanation.strip()
    return "暂无说明。"


def build_export_markdown(level_config: dict) -> str:
    title = level_config.get("title", "")
    target_model = level_config.get("target_model", "")
    intro = level_config.get("intro", "")
    victory = level_config.get("victory_condition", "")
    setting_text = format_setting(level_config.get("setting", ""))
    model_hint = str(level_config.get("model_hint", "")).strip()
    explanation = format_model_explanation_markdown(
        get_model_explanation(level_config)
    )
    lines = [
        f"# {title}",
        "",
        "## 关卡信息",
        f"- 目标能力/策略：{target_model}",
        f"- 能力补充说明：{model_hint or '无'}",
        f"- 胜利条件：{victory}",
        f"- 最大回合：{level_config.get('max_turns', '')}",
        f"- 挑战额度：{st.session_state.get('challenge_count', '')}"
        f"/{st.session_state.get('challenge_limit', '')}",
        f"- 当前回合：{st.session_state.get('turn_count', '')}",
        f"- 当前状态：{st.session_state.get('status', '')}",
        "",
        "## 背景介绍",
        intro or "暂无",
        "",
    ]
    if setting_text:
        lines.extend(["## 场景设定", setting_text, ""])
    lines.extend(["## 能力/策略解释", explanation, ""])
    lines.append("## 对话记录")
    history = st.session_state.get("history", [])
    if history:
        for entry in history:
            if entry.get("role") == "user":
                action_label = entry.get("action_type", "动作")
                content = entry.get("normalized_content") or entry.get("content", "")
                lines.append(f"- **你（{action_label}）**：{content}")
            else:
                lines.append(f"- **AI**：{entry.get('content', '')}")
    else:
        lines.append("- 暂无对话记录。")
    lines.append("")
    mentor_report = st.session_state.get("mentor_report")
    if mentor_report:
        lines.append("## 复盘报告")
        lines.append(mentor_report)
        lines.append("")
    return "\n".join(lines).strip()


def format_setting(setting) -> str:
    if isinstance(setting, str):
        return setting.strip()
    if not isinstance(setting, dict):
        return ""
    parts = []
    labels = {
        "scene": "场景",
        "background": "背景",
        "roles": "角色",
        "era": "时代",
        "extra_requirements": "要求",
    }
    for key, label in labels.items():
        value = str(setting.get(key, "")).strip()
        if value:
            parts.append(f"{label}：{value}")
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
        "角色：你是MindCraft的关卡设计师。根据输入的能力/策略与设定生成关卡JSON。\n"
        "必须输出严格JSON，不要额外文字。\n"
        "JSON结构：\n"
        "{\n"
        '  "title": "...",\n'
        '  "intro": "...",\n'
        '  "victory_condition": "...",\n'
        '  "max_turns": 10,\n'
        '  "challenge_limit": 3,\n'
        '  "model_explanation": {\n'
        '    "definition": "...",\n'
        '    "example": "...",\n'
        '    "pitfall": "...",\n'
        '    "advice": "...",\n'
        '    "triggers": ["..."],\n'
        '    "steps": ["..."],\n'
        '    "success_signals": ["..."]\n'
        "  },\n"
        '  "setting": "用户设定的场景/背景/角色/时代/要求，合并为一段文字"\n'
        "}\n"
        "要求：\n"
        "- 全中文输出，命名具体，禁止使用“项目A/B”等抽象代号。\n"
        "- intro使用第二人称，包含具体项目名、人物关系与业务数据。\n"
        "- 引入现实复杂性、认知迷雾与人际摩擦。\n"
        "- model_explanation每项2-4句，包含一个简单案例。\n"
        "- triggers/steps/success_signals各给3-5条。\n"
        "- 若提供model_hint，需将其融入解释与案例中。\n"
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


def merge_setting_text(user_text: str, generated_setting) -> str:
    generated_text = format_setting(generated_setting)
    user_text = user_text.strip()
    if generated_text and user_text:
        if user_text in generated_text:
            return generated_text
        return f"{generated_text}；补充：{user_text}"
    return generated_text or user_text


def normalize_level_config(
    raw_level: dict,
    target_model: str,
    title_hint: str,
    max_turns: int,
    user_setting_text: str,
    model_hint: str,
) -> dict:
    title = str(raw_level.get("title", "")).strip() or title_hint.strip()
    if not title:
        title = f"{target_model}训练关"

    intro = str(raw_level.get("intro", "")).strip()
    if not intro:
        intro = "你面对一个需要谨慎决策的商业局面。"

    victory_condition = str(raw_level.get("victory_condition", "")).strip()
    if not victory_condition:
        victory_condition = "依据目标能力/策略做出关键决策并控制风险。"

    challenge_limit = raw_level.get("challenge_limit", DEFAULT_CHALLENGE_LIMIT)
    try:
        challenge_limit = int(challenge_limit)
    except (TypeError, ValueError):
        challenge_limit = DEFAULT_CHALLENGE_LIMIT

    model_explanation = raw_level.get("model_explanation", {})
    if not isinstance(model_explanation, (dict, str)):
        model_explanation = {}
    model_hint = model_hint.strip()

    setting_text = merge_setting_text(user_setting_text, raw_level.get("setting", ""))
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
        "challenge_limit": challenge_limit,
        "model_explanation": model_explanation,
        "model_hint": model_hint,
        "setting": setting_text,
    }


def generate_level_config(
    client: OpenAI,
    model: str,
    target_model: str,
    title_hint: str,
    user_setting_text: str,
    model_hint: str,
    max_turns: int,
) -> tuple[dict | None, str]:
    payload = {
        "target_model": target_model,
        "model_hint": model_hint,
        "title_hint": title_hint,
        "setting": user_setting_text,
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
            user_setting_text=user_setting_text,
            model_hint=model_hint,
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
            if "challenge_limit" not in data:
                data["challenge_limit"] = DEFAULT_CHALLENGE_LIMIT
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
    st.session_state.challenge_count = 0
    st.session_state.challenge_limit = int(
        level_config.get("challenge_limit", DEFAULT_CHALLENGE_LIMIT)
    )
    st.session_state.selected_level_id = level_config["level_id"]


def ensure_game_state(level_config: dict) -> None:
    if "selected_level_id" not in st.session_state:
        reset_game(level_config)
        return

    if "history" not in st.session_state:
        reset_game(level_config)
        return

    if "challenge_count" not in st.session_state:
        st.session_state.challenge_count = 0
    if "challenge_limit" not in st.session_state:
        st.session_state.challenge_limit = int(
            level_config.get("challenge_limit", DEFAULT_CHALLENGE_LIMIT)
        )

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
        "introduce_challenge": False,
        "challenge_note": "",
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
        model_explanation = "请根据目标能力/策略做出合理判定。"
    model_hint = str(level_config.get("model_hint", "")).strip()
    setting_text = format_setting(level_config.get("setting", {}))
    return (
        f"{GLOBAL_CONTEXT}\n"
        "角色：你是逻辑判官。不要写故事，只输出JSON。\n"
        "关卡信息：\n"
        f"- 标题：{level_config['title']}\n"
        f"- 开场：{level_config['intro']}\n"
        f"- 目标能力/策略：{level_config['target_model']}\n"
        f"- 能力说明：{model_explanation}\n"
        f"- 能力补充说明：{model_hint or '无'}\n"
        f"- 胜利条件：{level_config['victory_condition']}\n"
        f"- 最大回合：{level_config['max_turns']}\n"
        f"- 补充设定：{setting_text}\n"
        "任务：\n"
        "1) 分析用户意图。\n"
        "2) 判断是否符合现实物理规则。\n"
        "3) 判断是否符合目标能力/策略（优先看步骤与成功信号）。\n"
        "4) 更新游戏状态并给叙事者提示。\n"
        "5) 识别错别字或误拼写，推断用户本意，并给出修正结果。\n"
        "   - 如果不确定，保留原词但给出可能的备选。\n"
        "6) 实体一致性：基于提供的entity_bank复用已有名称，\n"
        "   仅在必要时新增，并在输出的entities中给出最新实体表。\n"
        "7) 挑战预算：全关最多引入3个新难题。\n"
        "   - 用户payload会提供challenge_count与challenge_limit。\n"
        "   - 当剩余额度<=0时，禁止引入新难题，应推动收束或成功。\n"
        "NON_EXPERT_PROTOCOL（非专业领域优先）：\n"
        "1) 聚焦能力/策略而非行业执行：不要用财务/法律/运营细节否定正确思路。\n"
        "2) 意图通过制：只要用户的意图服务于正确能力方向，允许判定成功。\n"
        "   - 例如“让团队调研可行性”“安抚关键成员”等，若方向正确，应给予正向结果。\n"
        "3) 仅允许建设性失败：只有当用户违背目标能力/策略时，才制造障碍。\n"
        "   - 障碍必须是能力/策略层面的反馈，而不是琐碎的执行问题。\n"
        "HARD_MODE_JUDGMENT（仅在能力违背时才启用）：\n"
        "- 当用户坚持错误思路或明显回避能力要点时，再加入挑战。\n"
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
        "- introduce_challenge: 是否引入新难题（true/false）\n"
        "- challenge_note: 本回合是否新增难题的简短说明\n"
        "只输出JSON，不要额外文本。"
    )


def build_narrator_prompt() -> str:
    return (
        f"{GLOBAL_CONTEXT}\n"
        "角色：你是叙事者。把逻辑提示渲染成有代入感的故事。\n"
        "规则：\n"
        "- 使用第二人称。\n"
        "- 1段为主，最多2段，整体不超过120字。\n"
        "- 严格遵循逻辑结果和提示。\n"
        "- 可以合理推进时间（如几天或几个月后）。\n"
        "- 仅在introduce_challenge为true且仍有挑战额度时，以困境问题收尾。\n"
        "- 当挑战额度用尽时，只能推进、收束或达成结果，不再抛新难题。\n"
        "- 不要暴露JSON或内部评估备注。\n"
        "- 若游戏结束，用收束句收尾，不要继续提问。"
        "- 若logic_result提示意图通过，应给出正向结果并推进剧情。\n"
        "\nREALISM_PROTOCOL（高保真现实感）：\n"
        "1) 禁止抽象代号：不要使用“项目A/B”“某员工”等，必须使用具体项目名、具体业务数据、具体人物关系。\n"
        "2) 认知迷雾：被放弃的项目必须包含至少两个诱人的正面信号；新机会必须包含风险和不确定性。\n"
        "3) 情感钩子：让失败项目绑上情感/承诺/身份。\n"
        "4) 展示而非说教：不要直接说“这是某模型的陷阱”，要用细节呈现。\n"
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
        "聚焦能力/策略，不要陷入行业执行细节。\n"
        "必须包含以下小节：\n"
        "1) 总结\n"
        "2) 能力运用\n"
        "3) 遗漏机会\n"
        "4) 下一步建议\n"
        "5) 参考解法（AI认为的更优选择与行动路径）\n"
        "要求直率、具体，引用用户的关键决策。"
    )


def build_hint_prompt() -> str:
    return (
        f"{GLOBAL_CONTEXT}\n"
        "角色：你是提示官（助教）。给玩家提供可执行的提示，但不要直接给答案。\n"
        "要求：\n"
        "- 输出2到4条要点，使用项目符号。\n"
        "- 不要直接泄露胜利条件或唯一最优解。\n"
        "- 基于现有实体名词（entity_bank）与场景设定给出建议。\n"
        "- 提醒玩家需要收集的信息或可能的风险。\n"
        "- 聚焦能力/策略，不要陷入行业执行细节。\n"
        "- 可以建议使用[观察]/[对话]/[动作]/[思考]等方式推进。\n"
        "- 使用中文。"
    )


def process_turn(
    client: OpenAI,
    model: str,
    level_config: dict,
    action_type: str,
    user_input: str,
    status_placeholder=None,
) -> None:
    if action_type == "提示":
        hint_request = user_input.strip() or "我需要提示。"
        st.session_state.history.append(
            {
                "role": "user",
                "content": hint_request,
                "normalized_content": hint_request,
                "action_type": action_type,
            }
        )
        hint_payload = {
            "action_type": action_type,
            "user_request": hint_request,
            "turn": st.session_state.turn_count,
            "max_turns": level_config["max_turns"],
            "current_status": st.session_state.status,
            "target_model": level_config["target_model"],
            "model_explanation": format_model_explanation(
                get_model_explanation(level_config)
            ),
            "setting": format_setting(level_config.get("setting", "")),
            "recent_history": summarize_history(st.session_state.history),
            "entity_bank": st.session_state.entity_bank,
        }
        hint_messages = [
            {"role": "system", "content": build_hint_prompt()},
            {"role": "user", "content": json.dumps(hint_payload, ensure_ascii=False)},
        ]
        st.session_state.last_debug = {
            "logic": {"messages": None, "response": None},
            "narrator": {"messages": None, "response": None},
            "mentor": {"messages": None, "response": None},
            "hint": {"messages": hint_messages, "response": None},
        }
        try:
            if status_placeholder:
                status_placeholder.empty()
                with status_placeholder.container():
                    with st.spinner("提示生成中..."):
                        hint_text = call_chat_completion(
                            client,
                            model,
                            hint_messages,
                            temperature=0.5,
                        )
            else:
                with st.spinner("提示生成中..."):
                    hint_text = call_chat_completion(
                        client,
                        model,
                        hint_messages,
                        temperature=0.5,
                    )
            st.session_state.last_debug["hint"]["response"] = hint_text
        except Exception:
            st.error("提示生成失败，请稍后重试。")
            hint_text = "提示暂时不可用。"
            st.session_state.last_debug["hint"]["response"] = "ERROR"

        st.session_state.history.append({"role": "assistant", "content": hint_text})
        st.session_state.game_log.append(
            {
                "turn": st.session_state.turn_count,
                "action_type": action_type,
                "user_input": hint_request,
                "hint_response": hint_text,
            }
        )
        return
    if action_type == "提前结束":
        end_request = user_input.strip() or "请求提前结束游戏。"
        st.session_state.history.append(
            {
                "role": "user",
                "content": end_request,
                "normalized_content": end_request,
                "action_type": action_type,
            }
        )
        st.session_state.history.append(
            {
                "role": "assistant",
                "content": "你选择提前结束本次挑战，进入复盘环节。",
            }
        )
        st.session_state.game_log.append(
            {
                "turn": st.session_state.turn_count,
                "action_type": action_type,
                "user_input": end_request,
                "note": "User requested early termination.",
            }
        )
        st.session_state.game_over = True
        st.session_state.status = "lost"
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
                        "reason": "User ended the game early.",
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        st.session_state.last_debug = {
            "logic": {"messages": None, "response": None},
            "narrator": {"messages": None, "response": None},
            "mentor": {"messages": mentor_messages, "response": None},
            "hint": {"messages": None, "response": None},
        }
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
        return

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
        "challenge_count": st.session_state.challenge_count,
        "challenge_limit": st.session_state.challenge_limit,
    }

    logic_messages = [
        {"role": "system", "content": build_logic_prompt(level_config)},
        {"role": "user", "content": json.dumps(logic_payload, ensure_ascii=False)},
    ]
    st.session_state.last_debug = {
        "logic": {"messages": logic_messages, "response": None},
        "narrator": {"messages": None, "response": None},
        "mentor": {"messages": None, "response": None},
        "hint": {"messages": None, "response": None},
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
    introduce_challenge = bool(logic_result.get("introduce_challenge"))
    if st.session_state.challenge_count >= st.session_state.challenge_limit:
        introduce_challenge = False
        logic_result["introduce_challenge"] = False
        if not logic_result.get("challenge_note"):
            logic_result["challenge_note"] = "挑战额度已用尽。"
    if introduce_challenge:
        st.session_state.challenge_count += 1
    challenge_remaining = max(
        0,
        st.session_state.challenge_limit - st.session_state.challenge_count,
    )
    logic_result["challenge_remaining"] = challenge_remaining
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
            "challenge_count": st.session_state.challenge_count,
            "challenge_limit": st.session_state.challenge_limit,
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
                    "challenge_count": st.session_state.challenge_count,
                    "challenge_limit": st.session_state.challenge_limit,
                    "challenge_remaining": challenge_remaining,
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
st.markdown(
    """
<style>
.app-title {
  font-size: 1.6rem;
  font-weight: 700;
  margin-bottom: 0.4rem;
}
.level-title {
  font-size: 1.2rem;
  font-weight: 600;
  margin-top: 0.2rem;
  margin-bottom: 0.25rem;
}
[data-testid="stChatMessageAvatar"],
[data-testid="stChatMessageAvatar"] *,
.stChatMessageAvatar {
  display: none !important;
  width: 0 !important;
  height: 0 !important;
  margin: 0 !important;
  padding: 0 !important;
  visibility: hidden !important;
}
[data-testid="stChatMessage"] > div:first-child {
  display: none !important;
}
[data-testid="stChatMessage"] {
  gap: 0.5rem;
  padding-left: 0;
}
[data-testid="stChatMessageContent"] {
  border-radius: 0.75rem;
  padding: 0.6rem 0.9rem;
  background-color: #f3f5f9;
}
[data-testid="stChatMessage"].stChatMessage--user [data-testid="stChatMessageContent"],
[data-testid="stChatMessage"][data-message-author="user"] [data-testid="stChatMessageContent"],
.stChatMessage.stChatMessage--user .stChatMessageContent {
  background-color: #e8f0ff;
}
[data-testid="stChatMessage"].stChatMessage--assistant [data-testid="stChatMessageContent"],
[data-testid="stChatMessage"][data-message-author="assistant"] [data-testid="stChatMessageContent"],
.stChatMessage.stChatMessage--assistant .stChatMessageContent {
  background-color: #f3f5f9;
}
</style>
""",
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="app-title">MindCraft - 能力训练场</div>',
    unsafe_allow_html=True,
)

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
        type="default",
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
    st.session_state.selected_level_config = level_configs[selected_index]
    debug_mode = st.toggle("Debug Mode", value=False)
    restart_clicked = st.button("Restart Game")

    with st.expander("Advanced Settings", expanded=False):
        challenge_limit_input = st.number_input(
            "挑战额度上限",
            min_value=0,
            max_value=10,
            value=int(st.session_state.get("challenge_limit", DEFAULT_CHALLENGE_LIMIT)),
            step=1,
            key="challenge_limit_input",
        )
        st.session_state.challenge_limit = int(challenge_limit_input)

    with st.expander("Create New Level", expanded=False):
        with st.form("create_level_form"):
            new_title_hint = st.text_input("关卡标题（可选）")
            new_model_name = st.text_area(
                "能力/策略名称（必填）",
                height=120,
                help="可输入“名称”或“名称：简要释义”。也可用换行分隔说明。",
            )
            new_setting_text = st.text_area(
                "场景/背景/角色/时代/要求（可选）",
                height=160,
            )
            new_max_turns = st.number_input(
                "最大回合数",
                min_value=3,
                max_value=30,
                value=10,
                step=1,
            )
            create_submitted = st.form_submit_button("生成并创建关卡")

    with st.expander("Export Session", expanded=False):
        export_clicked = st.button("生成导出内容")
        if export_clicked:
            st.session_state.export_markdown = build_export_markdown(
                st.session_state.get("selected_level_config", {})
            )
        export_markdown = st.session_state.get("export_markdown", "")
        if export_markdown:
            st.text_area(
                "Markdown 内容（可复制）",
                value=export_markdown,
                height=240,
            )
            st.download_button(
                "下载 Markdown 文件",
                data=export_markdown,
                file_name="mindcraft_session.md",
                mime="text/markdown",
            )

if create_submitted:
    model_name_raw = new_model_name.strip()
    model_name_clean, model_hint = parse_target_model_input(model_name_raw)
    if not model_name_clean:
        st.warning("请填写能力/策略名称。")
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
    setting_text = new_setting_text.strip()
    with st.spinner("正在生成关卡..."):
        generated_level, raw_response = generate_level_config(
            client=client,
            model=create_model,
            target_model=model_name_clean,
            title_hint=new_title_hint.strip(),
            user_setting_text=setting_text,
            model_hint=model_hint,
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
    st.success("关卡已创建并写入配置。")
    st.rerun()

selected_level = st.session_state.selected_level_config
ensure_game_state(selected_level)
if restart_clicked:
    reset_game(selected_level)
    st.rerun()

if not openai_api_key:
    st.info("Enter your API key to start playing.")

turn_info = f"Turn {st.session_state.turn_count}/{selected_level['max_turns']}"
level_title = html.escape(selected_level["title"])
level_model = html.escape(selected_level["target_model"])
st.markdown(
    f'<div class="level-title">{level_title} · {level_model}</div>',
    unsafe_allow_html=True,
)
st.caption(turn_info)
with st.expander(
    f"能力/策略解释：{selected_level['target_model']}",
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
    if action_type not in {"提示", "提前结束"} and not user_input.strip():
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
            st.text(
                f"挑战额度：{st.session_state.challenge_count}/"
                f"{st.session_state.challenge_limit}"
            )

        hint_debug = debug_payloads.get("hint")
        if hint_debug and hint_debug.get("messages"):
            with st.expander("Hint Prompts + Response", expanded=False):
                st.json(hint_debug["messages"])
                st.code(hint_debug["response"] or "", language="text")

        logic_debug = debug_payloads.get("logic")
        with st.expander("Logic Engine Prompts + Response", expanded=False):
            if logic_debug and logic_debug.get("messages"):
                st.json(logic_debug["messages"])
                st.code(logic_debug["response"] or "", language="text")
            else:
                st.text("No logic debug data.")

        narrator_debug = debug_payloads.get("narrator")
        with st.expander("Narrator Prompts + Response", expanded=False):
            if narrator_debug and narrator_debug.get("messages"):
                st.json(narrator_debug["messages"])
                st.code(narrator_debug["response"] or "", language="text")
            else:
                st.text("No narrator debug data.")

        mentor_debug = debug_payloads.get("mentor")
        with st.expander("Shadow Mentor Prompts + Response", expanded=False):
            if mentor_debug and mentor_debug.get("messages"):
                st.json(mentor_debug["messages"])
                st.code(mentor_debug["response"] or "", language="text")
            else:
                st.text("No mentor debug data.")

