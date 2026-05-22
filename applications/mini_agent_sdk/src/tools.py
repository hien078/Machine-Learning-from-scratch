class ToolRegistry:
    def __init__(self):
        # Stores the registered functions (tools)
        self.tools = {}

    def register(self, func):
        """
        Decorator that turns a regular function into a Tool.
        Usage: place @registry.register before a function definition.
        """
        self.tools[func.__name__] = func
        return func

    def get_all_functions(self):
        """Return the list of functions to hand off to the Gemini API."""
        return list(self.tools.values())

    def execute(self, function_call):
        """
        Receive a function-call request from Gemini, run the actual function, and return the result.
        """
        func_name = function_call.name

        # Get the arguments Gemini wants to pass in
        args = function_call.args

        if func_name in self.tools:
            print(f"⚙️ SDK is auto-running function: {func_name}({args})")
            func = self.tools[func_name]

            try:
                # Call the real Python function (unpack dictionary args)
                result = func(**args)
                return str(result)
            except Exception as e:
                return f"Error while running function: {e}"
        else:
            return f"Error: no tool found with name {func_name}"
