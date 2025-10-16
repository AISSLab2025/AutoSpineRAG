class RoutingAgent:
    def __init__(self, tools):
        self.tools = tools
        self.tool_map = {tool.name: tool for tool in tools}

    def run(self, tool_inputs: dict[str, any], tool_names: list[str] = None) -> str:
        results = []

        selected_tool_names = tool_names if tool_names else tool_inputs.keys()

        for name in selected_tool_names:
            tool = self.tool_map.get(name)
            if not tool:
                results.append(f"[ERROR] Tool '{name}' not found.")
                continue

            tool_input = tool_inputs.get(name)

            # Call the tool with the appropriate input
            try:
                result = tool.run(tool_input)
                results.append(result)
            except Exception as e:
                results.append(f"[ERROR] Tool '{name}' failed with error: {str(e)}")

        return "\n\n".join(results)