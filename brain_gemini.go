package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"google.golang.org/genai"
)

// geminiModel keeps the demo snappy and cheap; swap to "gemini-2.5-pro" for
// maximum capability. The same id works on both the AI Studio (API-key) and
// Vertex AI backends.
const geminiModel = "gemini-3.8-flash"

// geminiBrain is the Google Gemini backend. It talks to either AI Studio (an API
// key in GEMINI_API_KEY / GOOGLE_API_KEY) or Vertex AI (GCP project + location +
// Application Default Credentials). Each observe call is stateless and FORCES the
// report_developments function via ToolConfig mode ANY.
type geminiBrain struct {
	client  *genai.Client
	backend string // for the startup banner, e.g. "vertex:my-proj/us-central1"
	config  *genai.GenerateContentConfig
}

// useVertexAI reports whether to route Gemini through Vertex AI rather than the
// AI Studio API-key path: either explicitly (GOOGLE_GENAI_USE_VERTEXAI=1|true), or
// implicitly when a GCP project is configured and no Studio key is set.
func useVertexAI() bool {
	switch strings.ToLower(os.Getenv("GOOGLE_GENAI_USE_VERTEXAI")) {
	case "1", "true":
		return true
	}
	return os.Getenv("GEMINI_API_KEY") == "" && os.Getenv("GOOGLE_API_KEY") == "" &&
		os.Getenv("GOOGLE_CLOUD_PROJECT") != ""
}

func newGeminiBrain(ctx context.Context) (*geminiBrain, error) {
	cc := &genai.ClientConfig{Backend: genai.BackendGeminiAPI}
	backend := "ai-studio"
	if useVertexAI() {
		// Vertex uses Application Default Credentials (run once:
		//   gcloud auth application-default login
		// or set GOOGLE_APPLICATION_CREDENTIALS to a service-account key) — no API
		// key. Project/location come from the standard GCP env vars.
		project := os.Getenv("GOOGLE_CLOUD_PROJECT")
		if project == "" {
			return nil, fmt.Errorf("Vertex AI selected but GOOGLE_CLOUD_PROJECT is not set (and run: gcloud auth application-default login)")
		}
		location := envOr("GOOGLE_CLOUD_LOCATION", os.Getenv("GOOGLE_CLOUD_REGION"))
		if location == "" {
			location = "global" // Gemini 2.5 is served on the global endpoint
		}
		cc.Backend = genai.BackendVertexAI
		cc.Project = project
		cc.Location = location
		backend = "vertex:" + project + "/" + location
	}
	client, err := genai.NewClient(ctx, cc)
	if err != nil {
		return nil, fmt.Errorf("gemini client: %w", err)
	}

	entitySchema := &genai.Schema{
		Type: genai.TypeArray,
		Items: &genai.Schema{
			Type: genai.TypeObject,
			Properties: map[string]*genai.Schema{
				"name": {Type: genai.TypeString, Description: entNmDesc},
				"type": {Type: genai.TypeString, Description: entTyDesc},
			},
			Required: []string{"name"},
		},
	}
	relationSchema := &genai.Schema{
		Type: genai.TypeArray,
		Items: &genai.Schema{
			Type: genai.TypeObject,
			Properties: map[string]*genai.Schema{
				"subject":      {Type: genai.TypeString, Description: relSbDesc},
				"relationship": {Type: genai.TypeString, Description: relRlDesc},
				"object":       {Type: genai.TypeString, Description: relObDesc},
			},
			Required: []string{"subject", "relationship", "object"},
		},
	}
	config := &genai.GenerateContentConfig{
		MaxOutputTokens: 4096,
		Tools: []*genai.Tool{{
			FunctionDeclarations: []*genai.FunctionDeclaration{{
				Name:        toolName,
				Description: toolDesc,
				Parameters: &genai.Schema{
					Type: genai.TypeObject,
					Properties: map[string]*genai.Schema{
						"items": {
							Type:        genai.TypeArray,
							Description: itemsDesc,
							Items: &genai.Schema{
								Type: genai.TypeObject,
								Properties: map[string]*genai.Schema{
									"headline":  {Type: genai.TypeString, Description: hlDesc},
									"detail":    {Type: genai.TypeString, Description: detDesc},
									"entities":  entitySchema,
									"relations": relationSchema,
								},
								Required: []string{"headline"},
							},
						},
					},
					Required: []string{"items"},
				},
			}},
		}},
		// Force the function call so the observation set is always structured.
		ToolConfig: &genai.ToolConfig{FunctionCallingConfig: &genai.FunctionCallingConfig{
			Mode:                 genai.FunctionCallingConfigModeAny,
			AllowedFunctionNames: []string{toolName},
		}},
	}
	return &geminiBrain{client: client, backend: backend, config: config}, nil
}

func (b *geminiBrain) label() string { return "gemini/" + geminiModel + " (" + b.backend + ")" }

func (b *geminiBrain) observe(ctx context.Context, subject string, known []string, maxItems int) (observations, error) {
	system, user := observePrompt(subject, known, maxItems)
	b.config.SystemInstruction = genai.NewContentFromText(system, genai.RoleUser)

	resp, err := b.client.Models.GenerateContent(ctx, geminiModel,
		[]*genai.Content{genai.NewContentFromText(user, genai.RoleUser)}, b.config)
	if err != nil {
		return observations{}, err
	}
	if len(resp.Candidates) == 0 || resp.Candidates[0].Content == nil {
		return observations{}, fmt.Errorf("empty response")
	}
	for _, part := range resp.Candidates[0].Content.Parts {
		if part.FunctionCall != nil && part.FunctionCall.Name == toolName {
			// Re-marshal the args map through JSON so it decodes into the shared
			// observation structs exactly as the Anthropic path does.
			raw, err := json.Marshal(part.FunctionCall.Args)
			if err != nil {
				return observations{}, fmt.Errorf("encode function args: %w", err)
			}
			return parseObservations(string(raw))
		}
	}
	return observations{}, fmt.Errorf("model did not call %s", toolName)
}
