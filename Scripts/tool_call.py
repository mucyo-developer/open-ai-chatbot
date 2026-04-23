from openai import OpenAI
from cn_mcp import MCPClient
import json
import re

# =====================================================
# CONFIG
# =====================================================

llm = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

mcp = MCPClient(api_key="d53d191f-bcd5-4d6f-b422-ce0489204dc2")


# =====================================================
# LOAD + NORMALIZE TOOLS
# =====================================================

tools_raw = mcp.list_tools()

# ---- normalize any MCP format ----
if isinstance(tools_raw, str):
    tools_raw = json.loads(tools_raw)

if isinstance(tools_raw, dict):
    tools_raw = tools_raw.get("tools", [])

if isinstance(tools_raw, list):
    if len(tools_raw) > 0 and isinstance(tools_raw[0], dict):
        tool_names = [t["name"] for t in tools_raw]
    else:
        tool_names = tools_raw
else:
    raise ValueError("Unknown tool format")

print("Allowed tools:", tool_names)



# =====================================================
# MEMORY
# =====================================================

memory = []

def add(role, content, name=None):
    msg = {"role": role, "content": content}
    if name:
        msg["name"] = name
    memory.append(msg)

def last4():
    return memory[-4:]



# =====================================================
# SYSTEM PROMPT
# =====================================================

SYSTEM = f"""
You are a strict AI assistant with tool access.

Allowed tools:
{", ".join(tool_names)}

RULES:
- Use ONLY tools listed above
- Tool names must match EXACTLY
- NEVER invent tools
- NEVER explain when calling tools

If tool is required respond ONLY JSON:

{{
  "tool": "tool_name",
  "arguments": {{}}
}}

If no tool needed respond normally.
"""



# =====================================================
# JSON EXTRACTION (LLM SAFE)
# =====================================================

def extract_json(text):
    """
    Extract JSON even if model adds explanation text.
    """
    match = re.search(r"\{{.*\}}", text, re.DOTALL)
    if match:
        return json.loads(match.group())
    raise ValueError("No JSON found")



# =====================================================
# CHAT FUNCTION
# =====================================================

def chat(user_input):

    add("user", user_input)

    messages = [{"role": "system", "content": SYSTEM}] + last4()

    response = llm.chat.completions.create(
        model="gpt-oss:120b-cloud",
        messages=messages
    )

    reply = response.choices[0].message.content.strip()

    # =================================================
    # TRY TOOL CALL
    # =================================================
    try:
        data = extract_json(reply)

        if "tool" in data:

            tool_name = data["tool"]
            args = data.get("arguments", {})

            # ---- STRICT VALIDATION ----
            if tool_name not in tool_names:
                raise ValueError(f"Invalid tool requested: {tool_name}")

            print(f"\n[TOOL CALL] {tool_name} -> {args}")

            result = mcp.call_tool(tool_name, args)

            # assistant decided tool
            add("assistant", reply)

            # tool response message
            add("tool", str(result), name=tool_name)

            # followup reasoning
            followup = llm.chat.completions.create(
                model="gpt-oss:120b-cloud",
                messages=[{"role": "system", "content": SYSTEM}] + last4()
            )

            final = followup.choices[0].message.content.strip()

            add("assistant", final)

            return final

    except Exception as e:
        print("Tool parse error:", e)

    # =================================================
    # NORMAL RESPONSE
    # =================================================
    add("assistant", reply)
    return reply



# =====================================================
# RUN LOOP
# =====================================================

if __name__ == "__main__":

    print("\n:rocket: Agent started (type 'exit')\n")

    while True:

        user = input("You: ")

        if user.lower() == "exit":
            break

        output = chat(user)
        print("AI:", output)