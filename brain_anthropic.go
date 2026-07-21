package main

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
)

// anthropicModel keeps the demo snappy and cheap; swap to
// anthropic.ModelClaudeOpus4_8 for maximum capability.
const anthropicModel = anthropic.Model("claude-sonnet-5")

// anthropicBrain is the Claude backend. Each observe call is a single, stateless
// request that FORCES the report_developments tool, so the reply is always the
// structured tool input rather than prose.
type anthropicBrain struct {
	client anthropic.Client
}

// newAnthropicBrain builds the Claude backend. When apiKey is non-empty (from
// -anthropic-api-key) it's passed explicitly; otherwise the SDK falls back to its
// usual ANTHROPIC_API_KEY / ambient profile resolution.
func newAnthropicBrain(apiKey string) *anthropicBrain {
	var opts []option.RequestOption
	if apiKey != "" {
		opts = append(opts, option.WithAPIKey(apiKey))
	}
	return &anthropicBrain{client: anthropic.NewClient(opts...)}
}

func (b *anthropicBrain) label() string { return "anthropic/" + string(anthropicModel) }

func (b *anthropicBrain) observe(ctx context.Context, subject string, known []string, maxItems int) (observations, error) {
	tools := []anthropic.ToolUnionParam{{OfTool: &anthropic.ToolParam{
		Name:        toolName,
		Description: anthropic.String(toolDesc),
		InputSchema: anthropic.ToolInputSchemaParam{
			Properties: map[string]any{
				"items": map[string]any{
					"type":        "array",
					"description": itemsDesc,
					"items": map[string]any{
						"type": "object",
						"properties": map[string]any{
							"headline": map[string]any{"type": "string", "description": hlDesc},
							"detail":   map[string]any{"type": "string", "description": detDesc},
							"entities": map[string]any{
								"type":        "array",
								"description": entsDesc,
								"items": map[string]any{
									"type": "object",
									"properties": map[string]any{
										"name": map[string]any{"type": "string", "description": entNmDesc},
										"type": map[string]any{"type": "string", "description": entTyDesc},
									},
									"required": []string{"name"},
								},
							},
							"relations": map[string]any{
								"type":        "array",
								"description": relsDesc,
								"items": map[string]any{
									"type": "object",
									"properties": map[string]any{
										"subject":      map[string]any{"type": "string", "description": relSbDesc},
										"relationship": map[string]any{"type": "string", "description": relRlDesc},
										"object":       map[string]any{"type": "string", "description": relObDesc},
									},
									"required": []string{"subject", "relationship", "object"},
								},
							},
						},
						"required": []string{"headline"},
					},
				},
			},
			Required: []string{"items"},
		},
	}}}

	system, user := observePrompt(subject, known, maxItems)
	resp, err := b.client.Messages.New(ctx, anthropic.MessageNewParams{
		Model:     anthropicModel,
		MaxTokens: 4096,
		System:    []anthropic.TextBlockParam{{Text: system}},
		Messages:  []anthropic.MessageParam{anthropic.NewUserMessage(anthropic.NewTextBlock(user))},
		Tools:     tools,
		// Force the tool: the observation set is the tool input, never prose.
		ToolChoice: anthropic.ToolChoiceUnionParam{OfTool: &anthropic.ToolChoiceToolParam{Name: toolName}},
	})
	if err != nil {
		return observations{}, err
	}

	for _, blk := range resp.Content {
		if v, ok := blk.AsAny().(anthropic.ToolUseBlock); ok {
			return parseObservations(v.JSON.Input.Raw())
		}
	}
	return observations{}, fmt.Errorf("model did not call %s", toolName)
}

// parseObservations decodes the forced-tool arguments (shared shape across both
// providers) into the observations struct.
func parseObservations(raw string) (observations, error) {
	var in struct {
		Items []observation `json:"items"`
	}
	if err := json.Unmarshal([]byte(raw), &in); err != nil {
		return observations{}, fmt.Errorf("decode tool input: %w", err)
	}
	return observations{Items: in.Items}, nil
}
