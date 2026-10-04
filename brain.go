package main

import (
	"context"
	"fmt"
	"os"
	"strings"
)

// brain is the pluggable observing LLM. Given a subject to watch and the entities
// already tracked, it reports notable developments as structured observations.
// It holds no conversation history: each run is a fresh, unattended
// observation. Everything Jennah-facing is identical regardless of which
// brain answers; only the LLM differs, and that's the point of the demo.
type brain interface {
	observe(ctx context.Context, subject string, known []string, maxItems int) (observations, error)
	label() string // short "provider/model" string for the startup banner
}

// observations is the structured result of one observe call.
type observations struct {
	Items []observation
}

// The report_developments tool, described once and mapped into each SDK's own tool
// type so the two backends stay in lockstep. The model is FORCED to call it, so its
// arguments ARE the structured output; there is no free-text path. LLM-as-source
// here keeps the demo standalone; a real deployment would fill these from a
// web-search / RSS / news API instead (see the README).
const (
	toolName = "report_developments"
	toolDesc = "Report the most notable RECENT developments about the watched subject. Each item is one distinct development. Prefer genuinely new or changed things over evergreen background. Name the concrete entities involved (companies, products, people, events) and how they relate."

	itemsDesc = "the notable developments; one entry per distinct development"
	hlDesc    = "one-line headline for the development"
	detDesc   = "1-3 sentences of detail: what happened and why it matters"
	entsDesc  = "the concrete named entities involved in this development"
	entNmDesc = "entity name, e.g. 'Anthropic', 'Claude Opus 4.8', 'Spanner'"
	entTyDesc = "entity kind, e.g. 'company', 'product', 'person', 'event', 'technology'"
	relsDesc  = "how the entities relate, as (subject)-[relationship]->(object) triples"
	relSbDesc = "the subject entity name"
	relRlDesc = "short verb phrase, e.g. 'competes with', 'acquired', 'launched', 'partners with'"
	relObDesc = "the object entity name"
)

// newBrain selects the observing provider. "auto" prefers Anthropic when an
// Anthropic key is present, else Gemini, so someone with only one key set just
// runs `go run .`. anthropicKey, when non-empty, is the Anthropic API key from
// -anthropic-api-key (already defaulted to $ANTHROPIC_API_KEY); it overrides the
// SDK's own env lookup.
//
// "bedrock" is Claude on Amazon Bedrock, using aws (region and profile). It is
// never chosen by "auto": AWS credentials are present in many shells for reasons
// that have nothing to do with this demo, so having them is no sign of intent.
func newBrain(ctx context.Context, provider, anthropicKey string, aws awsTarget) (brain, error) {
	if provider == "auto" {
		switch {
		case anthropicKey != "":
			provider = "anthropic"
		case os.Getenv("GEMINI_API_KEY") != "" || os.Getenv("GOOGLE_API_KEY") != "" || useVertexAI():
			provider = "gemini"
		default:
			return nil, fmt.Errorf("no chat credentials found: pass -anthropic-api-key / set ANTHROPIC_API_KEY (Anthropic), pass -provider bedrock (Claude on Amazon Bedrock, explicit only), or set Vertex AI env / GEMINI_API_KEY (Gemini)")
		}
	}
	switch strings.ToLower(provider) {
	case "anthropic", "claude":
		return newAnthropicBrain(anthropicKey), nil
	case "bedrock":
		if aws.region == "" {
			return nil, fmt.Errorf("-provider bedrock needs an AWS region: pass -aws-region")
		}
		return newBedrockBrain(ctx, aws.region, aws.profile), nil
	case "gemini":
		return newGeminiBrain(ctx)
	default:
		return nil, fmt.Errorf("unknown -provider %q (want auto|anthropic|bedrock|gemini)", provider)
	}
}

// awsTarget is where -provider bedrock sends requests: the AWS region, and the
// named profile whose credentials sign them (empty uses the default chain).
type awsTarget struct {
	region  string
	profile string
}

// observePrompt builds the shared system/user prompt both backends send. Passing
// the already-known entities lets the model bias toward genuinely new developments
// (the client still dedups semantically regardless).
func observePrompt(subject string, known []string, maxItems int) (system, user string) {
	system = "You are a market intelligence watcher with durable long-term memory. " +
		"You run unattended on a schedule. Report the most notable recent developments about the subject below. " +
		"Focus on what is new, changed, or newsworthy; avoid restating stable background. " +
		"Always call the report_developments tool with your findings."

	var b strings.Builder
	fmt.Fprintf(&b, "Subject to watch: %s\n\n", subject)
	if len(known) == 0 {
		b.WriteString("You have not tracked any entities for this subject yet. This is the first run.\n")
	} else {
		b.WriteString("Entities you already track (prefer developments that add to or go beyond these):\n")
		for _, e := range known {
			b.WriteString("- ")
			b.WriteString(e)
			b.WriteString("\n")
		}
	}
	fmt.Fprintf(&b, "\nReport up to %d distinct developments.", maxItems)
	return system, b.String()
}
