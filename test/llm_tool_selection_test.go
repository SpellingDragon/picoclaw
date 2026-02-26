package test

import (
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

	// Write tool - always visible
	writeTool := tools.NewWriteFileTool(workspace, true)
	registry.Register(writeTool)

	t.Run("regular user should not see admin tools", func(t *testing.T) {
		visibilityCtx := tools.ToolVisibilityContext{
			Channel:   "cli",
			ChatID:    "test-chat",
			UserRoles: []string{"user"}, // Regular user
		}

		// Get filtered tool definitions
		filteredTools := registry.ToProviderDefsForContext(visibilityCtx)

		// Extract tool names
		toolNames := make([]string, len(filteredTools))
		for i, toolDef := range filteredTools {
			toolNames[i] = toolDef.Function.Name
		}

		// Verify admin tool is NOT in the list
		assert.Contains(t, toolNames, "read_file")
		assert.Contains(t, toolNames, "write_file")
		assert.NotContains(t, toolNames, "exec") // Admin tool should be filtered out
	})

	t.Run("admin user should see admin tools", func(t *testing.T) {
		visibilityCtx := tools.ToolVisibilityContext{
			Channel:   "cli",
			ChatID:    "test-chat",
			UserRoles: []string{"admin", "user"}, // Admin user
		}

		filteredTools := registry.ToProviderDefsForContext(visibilityCtx)

		// Extract tool names
		toolNames := make([]string, len(filteredTools))
		for i, toolDef := range filteredTools {
			toolNames[i] = toolDef.Function.Name
		}

		// Verify admin tool IS in the list
		assert.Contains(t, toolNames, "read_file")
		assert.Contains(t, toolNames, "write_file")
		assert.Contains(t, toolNames, "exec") // Admin tool should be visible
	})
}

// TestSkillRecommender_WithRealLLM tests the skill recommender with actual LLM
func TestSkillRecommender_WithRealLLM(t *testing.T) {
	// Load API key from environment (source ~/.bashrc first if needed)
	apiKey := os.Getenv("ZAI_API_KEY")

	// Try to load from .bashrc if not set in current env
	if apiKey == "" {
		t.Log("ZAI_API_KEY not in current env, test will skip")
		t.Skip("ZAI_API_KEY not set. Make sure to run: source ~/.bashrc before running tests")
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

	// Load base config from test/config.json and inject API key
	configPath := testConfigPath(t)
	cfg, err := config.LoadConfig(configPath)
	require.NoError(t, err)

	// Inject API key into the model config
	for i := range cfg.ModelList {
		if cfg.ModelList[i].ModelName == cfg.Agents.Defaults.Model {
			cfg.ModelList[i].APIKey = apiKey
			break
		}
	}

	// Create provider from config (using agent's configured model)
	provider, modelID, err := providers.CreateProvider(cfg)
	require.NoError(t, err)

	// Use the model from config for recommender
	modelToUse := modelID
	if modelToUse == "" {
		modelToUse = cfg.Agents.Defaults.Model
	}

	// Create recommender
	recommender := agent.NewSkillRecommender(loader, provider, modelToUse)

	t.Run("recommender selects relevant skills for weather query", func(t *testing.T) {
		// Test case 1: Weather-related query
		recommendations, err := recommender.RecommendSkillsForContext(
			"cli",
			"user-123",
			"北京今天天气怎么样？需要带伞吗？",
			nil,
		)

		require.NoError(t, err)
		assert.NotEmpty(t, recommendations)

		// Log all recommendations
		t.Logf("Received %d recommendations:", len(recommendations))
		for _, rec := range recommendations {
			t.Logf("  - %s (score: %.1f, reason: %s)", rec.Name, rec.Score, rec.Reason)
		}

		// Should recommend weather-check skill as top recommendation
		assert.NotEmpty(t, recommendations, "Should have at least one recommendation")
		if len(recommendations) > 0 {
			// Check if weather-check is the top recommendation or has reasonable score
			topRec := recommendations[0]
			assert.Equal(t, "weather-check", topRec.Name, "Top recommendation should be weather-check for weather queries")
			assert.Greater(t, topRec.Score, 10.0, "Top recommendation should have score > 10")
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

		// Log all recommendations
		t.Logf("Received %d recommendations:", len(recommendations))
		for _, rec := range recommendations {
			t.Logf("  - %s (score: %.1f, reason: %s)", rec.Name, rec.Score, rec.Reason)
		}

		// Should recommend file-manager skill as top recommendation
		assert.NotEmpty(t, recommendations, "Should have at least one recommendation")
		if len(recommendations) > 0 {
			topRec := recommendations[0]
			assert.Equal(t, "file-manager", topRec.Name, "Top recommendation should be file-manager for file queries")
			assert.Greater(t, topRec.Score, 10.0, "Top recommendation should have score > 10")
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
		// Channel-specific recommendations may be empty if no telegram skills exist
		if len(telegramRecs) > 0 {
			t.Log("Telegram channel recommendations:")
			for _, rec := range telegramRecs {
				t.Logf("  - %s (score: %.1f, reason: %s)", rec.Name, rec.Score, rec.Reason)
			}
		} else {
			t.Log("No specific recommendations for this channel context (expected behavior)")
		}
	})

	t.Run("recommender handles web search query", func(t *testing.T) {
		// Test case 4: Web search query
		recommendations, err := recommender.RecommendSkillsForContext(
			"cli",
			"user-789",
			"帮我搜索一下最近的人工智能新闻",
			nil,
		)

		require.NoError(t, err)
		assert.NotEmpty(t, recommendations)

		// Log all recommendations
		t.Logf("Received %d recommendations:", len(recommendations))
		for _, rec := range recommendations {
			t.Logf("  - %s (score: %.1f, reason: %s)", rec.Name, rec.Score, rec.Reason)
		}

		// Should recommend web-search skill as top recommendation
		assert.NotEmpty(t, recommendations, "Should have at least one recommendation")
		if len(recommendations) > 0 {
			topRec := recommendations[0]
			assert.Equal(t, "web-search", topRec.Name, "Top recommendation should be web-search for search queries")
			assert.Greater(t, topRec.Score, 10.0, "Top recommendation should have score > 10")
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
