package test

import (
	"context"
	"os"
	"testing"

	"github.com/sipeed/picoclaw/pkg/agent"
	"github.com/sipeed/picoclaw/pkg/config"
	"github.com/sipeed/picoclaw/pkg/providers"
	"github.com/sipeed/picoclaw/pkg/skills"
	"github.com/sipeed/picoclaw/pkg/tools"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// mockProviderWithToolCapture captures the tools passed to the LLM for verification
type mockProviderWithToolCapture struct {
	lastMessages []providers.Message
	lastTools    []providers.ToolDefinition
	responseFunc func() *providers.LLMResponse
}

func (m *mockProviderWithToolCapture) Chat(
	ctx context.Context,
	messages []providers.Message,
	tools []providers.ToolDefinition,
	model string,
	options map[string]any,
) (*providers.LLMResponse, error) {
	// Capture for verification
	m.lastMessages = messages
	m.lastTools = tools

	// Use custom response function if set, otherwise return default
	if m.responseFunc != nil {
		return m.responseFunc(), nil
	}

	return &providers.LLMResponse{
		Content:      "pong",
		FinishReason: "stop",
	}, nil
}

func (m *mockProviderWithToolCapture) GetDefaultModel() string {
	return "local"
}

// TestLLMToolSelection_WithContextFiltering verifies that tool filtering works correctly
func TestLLMToolSelection_WithContextFiltering(t *testing.T) {
	workspace := t.TempDir()

	// Create tool registry with different visibility rules
	registry := tools.NewToolRegistry()

	// Public tool - always visible
	publicTool := tools.NewReadFileTool(workspace, true)
	registry.Register(publicTool)

	// Admin-only tool - use RegisterWithFilter
	adminTool := tools.NewExecTool(workspace, true)
	registry.RegisterWithFilter(adminTool, func(ctx tools.ToolVisibilityContext) bool {
		for _, role := range ctx.UserRoles {
			if role == "admin" {
				return true
			}
		}
		return false
	})

	// Telegram-specific tool (no filter for this test - just checking role-based filtering)
	telegramTool := tools.NewWriteFileTool(workspace, true)
	registry.Register(telegramTool) // Use Register instead of RegisterWithFilter

	t.Run("regular user should not see admin tools", func(t *testing.T) {
		mockLLM := &mockProviderWithToolCapture{}

		visibilityCtx := tools.ToolVisibilityContext{
			Channel:   "cli",
			ChatID:    "test-chat",
			UserRoles: []string{"user"}, // Regular user
		}

		// Get filtered tool definitions
		filteredTools := registry.ToProviderDefsForContext(visibilityCtx)

		// Simulate LLM call with filtered tools
		_, err := mockLLM.Chat(context.Background(), nil, filteredTools, "local", nil)
		require.NoError(t, err)

		// Verify admin tool is NOT in the list
		assert.Len(t, mockLLM.lastTools, 2) // read_file + write_file

		// Extract tool names
		toolNames := make([]string, len(mockLLM.lastTools))
		for i, toolDef := range mockLLM.lastTools {
			toolNames[i] = toolDef.Function.Name
		}

		assert.Contains(t, toolNames, "read_file")
		assert.NotContains(t, toolNames, "exec") // Admin tool should be filtered out
	})

	t.Run("admin user should see admin tools", func(t *testing.T) {
		mockLLM := &mockProviderWithToolCapture{}

		visibilityCtx := tools.ToolVisibilityContext{
			Channel:   "cli",
			ChatID:    "test-chat",
			UserRoles: []string{"admin", "user"}, // Admin user
		}

		filteredTools := registry.ToProviderDefsForContext(visibilityCtx)
		_, err := mockLLM.Chat(context.Background(), nil, filteredTools, "local", nil)
		require.NoError(t, err)

		// Verify admin tool IS in the list
		assert.Len(t, mockLLM.lastTools, 3) // read_file + write_file + exec

		toolNames := make([]string, len(mockLLM.lastTools))
		for i, toolDef := range mockLLM.lastTools {
			toolNames[i] = toolDef.Function.Name
		}

		assert.Contains(t, toolNames, "exec") // Admin tool should be visible
	})
}

// TestLLMToolSelection_MockProviderResponse tests LLM tool selection behavior
func TestLLMToolSelection_MockProviderResponse(t *testing.T) {
	workspace := t.TempDir()

	// Setup skills
	createSkill := func(name, desc string) {
		skillDir := workspace + "/skills/" + name
		err := os.MkdirAll(skillDir, 0o755)
		require.NoError(t, err)

		content := `---
name: ` + name + `
description: ` + desc + `
---

# ` + name
		err = os.WriteFile(skillDir+"/SKILL.md", []byte(content), 0o644)
		require.NoError(t, err)
	}

	createSkill("web-search", "Search the web")
	createSkill("file-manager", "Manage files")
	createSkill("code-helper", "Help with coding")

	// Create tool registry
	registry := tools.NewToolRegistry()
	registry.Register(tools.NewReadFileTool(workspace, true))
	registry.Register(tools.NewWriteFileTool(workspace, true))

	t.Run("LLM selects appropriate tool based on context", func(t *testing.T) {
		// Mock LLM that simulates tool selection
		mockLLM := &mockProviderWithToolCapture{
			responseFunc: func() *providers.LLMResponse {
				// Simulate LLM choosing read_file tool
				return &providers.LLMResponse{
					Content: "",
					ToolCalls: []providers.ToolCall{
						{
							ID:   "call_123",
							Type: "function",
							Function: &providers.FunctionCall{
								Name:      "read_file",
								Arguments: `{"path": "test.txt"}`,
							},
						},
					},
					FinishReason: "tool_calls",
				}
			},
		}

		// Build context with tools
		visibilityCtx := tools.ToolVisibilityContext{
			Channel: "cli",
			ChatID:  "test",
		}

		toolDefs := registry.ToProviderDefsForContext(visibilityCtx)

		// Call LLM
		resp, err := mockLLM.Chat(context.Background(), nil, toolDefs, "local", nil)
		require.NoError(t, err)

		// Verify LLM made a tool call
		assert.Len(t, resp.ToolCalls, 1)
		assert.Equal(t, "read_file", resp.ToolCalls[0].Function.Name)

		// Verify only available tools were passed to LLM
		assert.Len(t, mockLLM.lastTools, 2)
	})
}

// TestSkillRecommender_WithRealLLM tests the skill recommender with actual LLM
func TestSkillRecommender_WithRealLLM(t *testing.T) {
	// Skip if no API key
	apiKey := os.Getenv("ZAI_API_KEY")
	if apiKey == "" {
		t.Skip("ZAI_API_KEY not set, skipping real LLM test")
	}

	workspace := t.TempDir()

	// Create test skills
	createSkill := func(name, desc string) {
		skillDir := workspace + "/skills/" + name
		err := os.MkdirAll(skillDir, 0o755)
		require.NoError(t, err)

		content := `---
name: ` + name + `
description: ` + desc + `
---

# ` + name
		err = os.WriteFile(skillDir+"/SKILL.md", []byte(content), 0o644)
		require.NoError(t, err)
	}

	createSkill("web-search", "Search the web for information")
	createSkill("file-manager", "Create, read, update files")
	createSkill("weather-check", "Check weather for a location")
	createSkill("calendar-manage", "Manage calendar events")

	// Load skills
	loader := skills.NewSkillsLoader(workspace, "", "")
	allSkills := loader.ListSkills()
	require.Len(t, allSkills, 4)

	// Create model config for Zhipu (using ZAI_API_KEY)
	modelCfg := &config.ModelConfig{
		ModelName: "zhipu-test",
		Model:     "zhipu/glm-4-flash",
		APIKey:    apiKey,
	}

	provider, _, err := providers.CreateProviderFromConfig(modelCfg)
	require.NoError(t, err)

	// Create recommender
	recommender := agent.NewSkillRecommender(loader, provider, "glm-4-flash")

	t.Run("recommender selects relevant skills for user query", func(t *testing.T) {
		// Test case 1: Weather-related query
		recommendations, err := recommender.RecommendSkillsForContext(
			"cli",
			"user-123",
			"北京今天天气怎么样？需要带伞吗？",
			nil,
		)

		require.NoError(t, err)
		assert.NotEmpty(t, recommendations)

		// Should recommend weather-check skill
		foundWeather := false
		for _, rec := range recommendations {
			if rec.Name == "weather-check" && rec.Score >= 30.0 {
				foundWeather = true
				break
			}
		}
		assert.True(t, foundWeather, "Should recommend weather-check skill")

		t.Log("Recommendations for weather query:")
		for _, rec := range recommendations {
			t.Logf("  - %s (score: %.1f, reason: %s)", rec.Name, rec.Score, rec.Reason)
		}
	})

	t.Run("recommender handles file-related queries", func(t *testing.T) {
		// Test case 2: File-related query
		recommendations, err := recommender.RecommendSkillsForContext(
			"cli",
			"user-123",
			"帮我创建一个文件，内容是 hello world",
			nil,
		)

		require.NoError(t, err)
		assert.NotEmpty(t, recommendations)

		// Should recommend file-manager skill
		foundFile := false
		for _, rec := range recommendations {
			if rec.Name == "file-manager" && rec.Score >= 30.0 {
				foundFile = true
				break
			}
		}
		assert.True(t, foundFile, "Should recommend file-manager skill")

		t.Log("Recommendations for file query:")
		for _, rec := range recommendations {
			t.Logf("  - %s (score: %.1f, reason: %s)", rec.Name, rec.Score, rec.Reason)
		}
	})

	t.Run("recommender considers channel context", func(t *testing.T) {
		// Test case 3: Different channels may affect recommendations
		telegramRecs, err := recommender.RecommendSkillsForContext(
			"telegram",
			"user-456",
			"发个投票给大家",
			nil,
		)

		require.NoError(t, err)
		assert.NotEmpty(t, telegramRecs)

		t.Log("Telegram channel recommendations:")
		for _, rec := range telegramRecs {
			t.Logf("  - %s (score: %.1f)", rec.Name, rec.Score)
		}
	})
}

// TestLLMToolSelection_FilteringInAgentContext tests tool filtering in agent context
func TestLLMToolSelection_FilteringInAgentContext(t *testing.T) {
	workspace := t.TempDir()

	// Setup tool registry with filters
	registry := tools.NewToolRegistry()
	registry.Register(tools.NewReadFileTool(workspace, true))
	registry.Register(tools.NewWriteFileTool(workspace, true))
	registry.Register(tools.NewListDirTool(workspace, true))

	// Admin-only tool
	execTool := tools.NewExecTool(workspace, true)
	registry.RegisterWithFilter(execTool, func(ctx tools.ToolVisibilityContext) bool {
		for _, role := range ctx.UserRoles {
			if role == "admin" {
				return true
			}
		}
		return false
	})

	t.Run("tool selection respects visibility filters", func(t *testing.T) {
		// Manually test tool filtering
		visibilityCtx := tools.ToolVisibilityContext{
			Channel:   "cli",
			ChatID:    "test-user",
			UserRoles: []string{"user"},
		}

		filteredTools := registry.ToProviderDefsForContext(visibilityCtx)

		// Verify exec tool is filtered out
		for _, toolDef := range filteredTools {
			assert.NotEqual(t, "exec", toolDef.Function.Name, "exec tool should be filtered for regular users")
		}

		// Now test with admin user
		adminCtx := tools.ToolVisibilityContext{
			Channel:   "cli",
			ChatID:    "admin-user",
			UserRoles: []string{"admin"},
		}

		adminTools := registry.ToProviderDefsForContext(adminCtx)

		// Verify exec tool is included for admin
		foundExec := false
		for _, toolDef := range adminTools {
			if toolDef.Function.Name == "exec" {
				foundExec = true
				break
			}
		}
		assert.True(t, foundExec, "exec tool should be visible for admin users")
	})
}
