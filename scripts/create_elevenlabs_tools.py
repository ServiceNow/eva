"""Create ElevenLabs tools from the airline agent YAML config and add them to an agent.

Reads tool definitions from configs/agents/airline_agent.yaml, converts them to
ElevenLabs client tool format, creates each tool via the API, then attaches all
tool IDs to the specified ElevenLabs agent.

Usage:
    python scripts/create_elevenlabs_tools.py
    python scripts/create_elevenlabs_tools.py --agent-id <agent_id>
    python scripts/create_elevenlabs_tools.py --dry-run
"""

import argparse
from pathlib import Path

import yaml
from elevenlabs.client import ElevenLabs
from elevenlabs.types import ConversationalConfig
from elevenlabs.types.agent_config import AgentConfig
from elevenlabs.types.array_json_schema_property_input import ArrayJsonSchemaPropertyInput
from elevenlabs.types.literal_json_schema_property import LiteralJsonSchemaProperty
from elevenlabs.types.object_json_schema_property_input import ObjectJsonSchemaPropertyInput
from elevenlabs.types.prompt_agent_api_model_output import PromptAgentApiModelOutput
from elevenlabs.types.tool_request_model import ToolRequestModel
from elevenlabs.types.tool_request_model_tool_config import ToolRequestModelToolConfig_Client

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs" / "agents"

# Map YAML parameter types to ElevenLabs LiteralJsonSchemaProperty types
LITERAL_TYPE_MAP: dict[str, str] = {
    "string": "string",
    "integer": "integer",
    "float": "number",
    "boolean": "boolean",
}

PropertyType = LiteralJsonSchemaProperty | ObjectJsonSchemaPropertyInput | ArrayJsonSchemaPropertyInput


def load_tools_from_yaml(config_path: Path) -> list[dict]:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config.get("tools", [])


def _make_property(param: dict) -> PropertyType:
    """Convert a single YAML parameter to an ElevenLabs JSON schema property."""
    param_type = param["type"]

    if param_type == "array":
        items_type = param.get("items", {}).get("type", "string")
        return ArrayJsonSchemaPropertyInput(
            type="array",
            description=param["description"],
            items=LiteralJsonSchemaProperty(
                type=LITERAL_TYPE_MAP.get(items_type, "string"),
                description=param.get("items", {}).get("description", "Item"),
            ),
        )

    kwargs: dict = {
        "type": LITERAL_TYPE_MAP.get(param_type, "string"),
        "description": param["description"],
    }
    if "enum" in param and kwargs["type"] == "string":
        kwargs["enum"] = param["enum"]
    return LiteralJsonSchemaProperty(**kwargs)


def build_parameters(tool: dict) -> ObjectJsonSchemaPropertyInput:
    """Build an ObjectJsonSchemaPropertyInput from YAML tool parameters."""
    properties: dict[str, PropertyType] = {}
    required: list[str] = []

    for param in tool.get("required_parameters", []):
        properties[param["name"]] = _make_property(param)
        required.append(param["name"])

    for param in tool.get("optional_parameters", []):
        properties[param["name"]] = _make_property(param)

    return ObjectJsonSchemaPropertyInput(
        type="object",
        properties=properties,
        required=required or None,
    )


def convert_tool(tool: dict) -> ToolRequestModel:
    """Convert a YAML tool definition to an ElevenLabs ToolRequestModel."""
    client_config = ToolRequestModelToolConfig_Client(
        type="client",
        name=tool["name"],
        description=f"{tool['name']}: {tool['description']}",
        expects_response=True,
        parameters=build_parameters(tool),
    )
    return ToolRequestModel(tool_config=client_config)


## to use run `python scripts/create_elevenlabs_tools.py --domain medical_hr --agent <agent_id>`
def main():
    parser = argparse.ArgumentParser(description="Create ElevenLabs tools from agent config")
    parser.add_argument("--agent-id", default="", help="ElevenLabs agent ID")
    parser.add_argument("--domain", default="airline", help="Agent domain name (e.g. airline, itsm, medical_hr)")
    parser.add_argument("--config", default=None, help="Path to agent YAML config (overrides --domain)")
    parser.add_argument("--dry-run", action="store_true", help="Print tool configs without creating them")
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else CONFIGS_DIR / f"{args.domain}_agent.yaml"
    tools = load_tools_from_yaml(config_path)
    print(f"Loaded {len(tools)} tools from {config_path}")

    if args.dry_run:
        for tool in tools:
            request_model = convert_tool(tool)
            tool_cfg = request_model.tool_config
            print(f"\n--- {tool_cfg.name} ---")
            print(f"  description: {tool_cfg.description}")
            print(f"  expects_response: {tool_cfg.expects_response}")
            print(f"  parameters: {tool_cfg.parameters}")
        return

    client = ElevenLabs()
    created_tool_ids: list[str] = []
    for tool in tools:
        request_model = convert_tool(tool)
        tool_name = request_model.tool_config.name
        print(f"Creating tool: {tool_name}...", end=" ")
        result = client.conversational_ai.tools.create(request=request_model)
        print(f"created (id={result.id})")
        created_tool_ids.append(result.id)

    print(f"\nCreated {len(created_tool_ids)} tools. Adding to agent {args.agent_id}...")

    client.conversational_ai.agents.update(
        agent_id=args.agent_id,
        conversation_config=ConversationalConfig(
            agent=AgentConfig(
                prompt=PromptAgentApiModelOutput(
                    tool_ids=created_tool_ids,
                ),
            ),
        ),
    )

    print(f"Successfully added {len(created_tool_ids)} tools to agent {args.agent_id}")


if __name__ == "__main__":
    main()
