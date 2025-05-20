# Router Agent 2 Compatibility Patch

This document outlines the necessary changes to make `router_agent2.py` compatible with the current pydantic-ai SDK.

## Required Changes

### 1. Fix Import Statements

```diff
- from pydantic_ai.messages import ModelMessage, SystemMessage, UserMessage, AssistantMessage
+ from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse
+ from pydantic_ai.messages import SystemPromptPart, UserPromptPart, TextPart
```

### 2. Update Message Handling

```diff
# Replace this code:
-    final_message_history: List[ModelMessage] = list(initial_message_history) if initial_message_history else []
-    if not any(isinstance(m, UserMessage) and m.content == query for m in final_message_history):
-         final_message_history.append(UserMessage(content=query))

# With:
+    final_message_history: List[ModelMessage] = list(initial_message_history) if initial_message_history else []
+    user_query_in_history = False
+    for msg in final_message_history:
+        if isinstance(msg, ModelRequest):
+            for part in msg.parts:
+                if isinstance(part, UserPromptPart) and part.content == query:
+                    user_query_in_history = True
+                    break
+    if not user_query_in_history:
+        final_message_history.append(ModelRequest(parts=[UserPromptPart(content=query)]))
```

### 3. Update Response Creation

```diff
# Replace:
-    final_message_history.append(AssistantMessage(content=aggregated_response.final_answer))

# With:
+    final_message_history.append(
+        ModelResponse(
+            parts=[TextPart(content=aggregated_response.final_answer)],
+            model_name=GEMINI_MODEL_NAME,
+            usage=Usage(),
+        )
+    )
```

### 4. Update Google Gemini Model Integration

```diff
# Replace:
- from pydantic_ai.models.google import GoogleModel
- from pydantic_ai.providers.google import GoogleProvider

# With:
+ from pydantic_ai.models.gemini import GeminiModel
+ from pydantic_ai.providers.google_gla import GoogleGLAProvider
```

### 5. Update Agent Initialization

```diff
# Replace:
-    shared_gemini_model = GoogleModel(GEMINI_MODEL_NAME, provider=google_provider)

# With:
+    shared_gemini_model = GeminiModel(GEMINI_MODEL_NAME, provider=google_provider)
```

### 6. Update Provider Initialization

```diff
# Replace:
-    google_provider = GoogleProvider(api_key=gemini_api_key)

# With:
+    google_provider = GoogleGLAProvider(api_key=gemini_api_key)
```

## Additional Considerations

1. The overall architecture may need further adjustments beyond these basic compatibility fixes.

2. Testing should be done incrementally to ensure each component works before testing the entire system.

3. Error handling may need updates to match the current API's approach.

4. The iterative planning loop design is sound but its implementation details need careful review against the current API.

5. Consider incorporating graph-based routing features from `graph_router.py` which seems to align well with the current pydantic-ai architecture.