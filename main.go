// Command memwatch is a demo agent: an AUTONOMOUS market/competitor WATCHER that
// runs UNATTENDED ON A SCHEDULE (cron) and consumes Jennah's public memory APIs
// exactly the way any external agent would — plain HTTP/JSON through the
// jennah-proxy gateway, authenticated with a jennah_sk_ API key. Like its sibling
// memchat it is a standalone Go module (its own go.mod, not part of the server
// build), so it models a real outside consumer and keeps the LLM SDK dependencies
// out of the server tree.
//
// Where memchat is a REACTIVE chatbot (a human speaks, it recalls, it replies),
// memwatch is PROACTIVE and headless: you give it a subject to watch and walk
// away. Each run it observes what's noteworthy now, DIFFS that against what it has
// already recorded in Jennah, and reports only WHAT'S NEW SINCE THE LAST RUN. The
// entity graph (companies / products / events and how they relate) accretes across
// runs, and the execution log becomes a run-by-run timeline. That is the whole
// pitch: Jennah is the durable brain an unattended agent runs its entire life
// against — kill it, cron it, resume it days later, and it never re-reports what
// it already knows.
//
// Each run does query → observe → diff → commit:
//
//  1. memory:query (log, limit 1)  — when did I last run? (for the banner)
//     memory:query (graph, 1 hop)  — which entities do I already track?
//  2. brain.observe — the LLM reports notable developments about the subject
//     (LLM-as-source here so the demo is standalone; a real deployment swaps in a
//     web-search / RSS / news API at this seam — see the README).
//  3. diff — each observed headline is semantic-queried against past chunks; if the
//     nearest match is closer than -dedup-distance it's KNOWN and skipped, else NEW.
//  4. memory:commit — the NEW items as vector chunks, their entities/relations as
//     graph nodes/edges (plus subject-[MENTIONS]->entity so a traversal from the
//     anchor enumerates the whole watch list), and one run record to the log — all
//     atomically.
//
// Cross-run memory is simply reusing the same agent_instance_id, persisted to a
// small state file. -show prints the accumulated graph and timeline without
// observing.
//
// Setup (Jennah key + one chat provider). Keys come from env or flags:
//
//	export JENNAH_API_KEY=jennah_sk_...      # from POST /v1/apikeys
//	export ANTHROPIC_API_KEY=sk-ant-...      # Anthropic key, OR
//	export GEMINI_API_KEY=...                # Google AI Studio key
//	go run . -subject "the AI agent memory / context platform market"
//
// The agent's home region can be chosen at first launch with -region (or
// $JENNAH_REGION); it's applied only when the workspace is created, since an agent
// is pinned to one home region for its lifetime. Empty uses the platform default.
// List the available regions with 'jnh agents regions'.
package main

import (
	"context"
	"crypto/rand"
	"crypto/sha1"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"os/signal"
	"strings"
	"time"

	agentpb "github.com/alphauslabs/jennah-sdk-go/jennah/agent/v1"
	"google.golang.org/protobuf/encoding/protojson"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/types/known/structpb"
)

// verbose makes the memory activity visible on screen: the observations returned,
// the dedup decisions, and the commit receipt. Set by -verbose.
var verbose bool

// subjectNode is the stable anchor every watched entity hangs off, so a traversal
// from it enumerates the whole watch list. Created once at bootstrap, then never
// re-sent (graph nodes are insert-only server-side).
const subjectNode = "subject"

func main() {
	var (
		endpoint     = flag.String("endpoint", envOr("JENNAH_ENDPOINT", "https://jennah.alphaus.cloud"), "Jennah proxy origin (http/https)")
		statePath    = flag.String("state", "memwatch-state.json", "path to the local state file (agent id)")
		provider     = flag.String("provider", "auto", "chat LLM: auto|gemini|anthropic (auto prefers Anthropic, else Gemini, by which API key is set)")
		region       = flag.String("region", envOr("JENNAH_REGION", ""), "Jennah home region for the agent (e.g. us-central1); empty uses the platform default. Only applied when creating a new agent workspace. List regions with 'jnh agents regions'")
		jennahKey    = flag.String("jennah-api-key", "", "Jennah API key (jennah_sk_...); falls back to $JENNAH_API_KEY")
		anthropicKey = flag.String("anthropic-api-key", "", "Anthropic API key (sk-ant-...); falls back to $ANTHROPIC_API_KEY")
		subject      = flag.String("subject", envOr("MEMWATCH_SUBJECT", ""), "what to watch, e.g. \"the AI agent memory platform market\" (required on first run; remembered thereafter)")
		maxItems     = flag.Int("max-items", 8, "max developments to ask the brain for each run")
		dedupDist    = flag.Float64("dedup-distance", 0.15, "cosine distance below which an observed headline counts as already-known (smaller = stricter)")
		show         = flag.Bool("show", false, "print the accumulated entity graph and run timeline, then exit (no observing)")
	)
	flag.BoolVar(&verbose, "verbose", false, "print observations, dedup decisions, and commit receipts each run")
	flag.Parse()

	// Fall back to the env vars, but keep them OUT of the flag defaults so -help
	// never prints the actual secret. A flag wins over its env var when both are set.
	*jennahKey = envOr2(*jennahKey, "JENNAH_API_KEY")
	*anthropicKey = envOr2(*anthropicKey, "ANTHROPIC_API_KEY")

	apiKey := *jennahKey
	if apiKey == "" {
		fatal("a Jennah API key is required: pass -jennah-api-key or set JENNAH_API_KEY (a jennah_sk_ key for an approved, entitled enterprise)")
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	jc := &jennahClient{
		endpoint: strings.TrimRight(*endpoint, "/"),
		token:    apiKey,
		hc:       &http.Client{Timeout: 60 * time.Second},
	}

	st, err := loadState(*statePath)
	if err != nil {
		fatal("load state: %v", err)
	}

	// One-time bootstrap: create the agent workspace and seed the stable "subject"
	// anchor node, then persist the id. -subject is required to create; afterwards
	// the watched subject is fixed for the agent's life (it's the anchor's Label).
	if st.AgentID == "" {
		if strings.TrimSpace(*subject) == "" {
			fatal("first run needs a subject to watch: pass -subject \"...\" (or set MEMWATCH_SUBJECT)")
		}
		id, err := createAgent(ctx, jc, *region)
		if err != nil {
			fatal("create agent: %v", err)
		}
		if _, err := commit(ctx, jc, id, &agentpb.CommitMemoryRequest{
			AgentInstanceId: id,
			Graph: &agentpb.GraphWrite{Nodes: []*agentpb.GraphNode{{
				NodeId: subjectNode, Label: strings.TrimSpace(*subject),
			}}},
		}); err != nil {
			fatal("seed subject node: %v", err)
		}
		st.AgentID = id
		st.Subject = strings.TrimSpace(*subject)
		save(*statePath, st)
		if *region != "" {
			fmt.Printf("created watcher %s (region %s)\n", id, *region)
		} else {
			fmt.Printf("created watcher %s (platform default region)\n", id)
		}
	}
	// The subject is pinned at creation; ignore any later -subject and use the
	// remembered one (fall back to the flag only if an older state file lacks it).
	watched := st.Subject
	if watched == "" {
		watched = strings.TrimSpace(*subject)
	}

	if *show {
		if err := showState(ctx, jc, st.AgentID, watched); err != nil {
			fatal("show: %v", err)
		}
		return
	}

	// The one part that varies by provider: the observing brain. Everything else is
	// provider-agnostic; the memory APIs don't care which LLM is thinking.
	br, err := newBrain(ctx, *provider, *anthropicKey)
	if err != nil {
		fatal("%v", err)
	}

	last, err := lastRun(ctx, jc, st.AgentID)
	if err != nil {
		fatal("read last run: %v", err)
	}
	lastStr := "never"
	if last != "" {
		lastStr = last
	}
	fmt.Printf("chat model: %s\n", br.label())
	fmt.Printf("watching %q — last run %s\n", watched, lastStr)

	if err := runOnce(ctx, jc, br, st.AgentID, watched, *maxItems, *dedupDist); err != nil {
		if errors.Is(err, context.Canceled) {
			return
		}
		fatal("%v", err)
	}
}

// runOnce performs a single autonomous watch cycle: observe → diff against memory
// → commit what's new → report. It takes no human input; this is what a cron entry
// invokes.
func runOnce(ctx context.Context, jc *jennahClient, br brain, agentID, subject string, maxItems int, dedupDist float64) error {
	known, err := knownEntities(ctx, jc, agentID)
	if err != nil {
		return fmt.Errorf("graph recall: %w", err)
	}
	vlog("already tracking %d entit(ies): %s", len(known), strings.Join(known, ", "))

	obs, err := br.observe(ctx, subject, known, maxItems)
	if err != nil {
		return fmt.Errorf("observe: %w", err)
	}
	vlog("brain reported %d development(s)", len(obs.Items))

	var fresh []observation
	knownCount := 0
	for _, it := range obs.Items {
		if strings.TrimSpace(it.Headline) == "" {
			continue
		}
		dist, ok, err := nearestDistance(ctx, jc, agentID, it.Headline)
		if err != nil {
			return fmt.Errorf("dedup query: %w", err)
		}
		if ok && dist < dedupDist {
			knownCount++
			vlog("known    (d=%.3f): %s", dist, singleLine(it.Headline))
			continue
		}
		vlog("NEW      (d=%s): %s", distStr(ok, dist), singleLine(it.Headline))
		fresh = append(fresh, it)
	}

	if err := commitRun(ctx, jc, agentID, subject, fresh, len(fresh), knownCount); err != nil {
		return fmt.Errorf("commit: %w", err)
	}

	// Report.
	if len(fresh) == 0 {
		fmt.Printf("\nnothing new since last run (%d already known).\n", knownCount)
		return nil
	}
	fmt.Printf("\n%d new since last run:\n", len(fresh))
	for _, it := range fresh {
		fmt.Printf("  • %s\n", singleLine(it.Headline))
		if d := strings.TrimSpace(it.Detail); d != "" {
			fmt.Printf("      %s\n", singleLine(d))
		}
	}
	if knownCount > 0 {
		fmt.Printf("\n(%d already known, skipped)\n", knownCount)
	}
	return nil
}

// ---- Jennah memory API calls (HTTP/JSON through the proxy) ----

// lastRun returns the timestamp (RFC3339) of the most recent execution-log step,
// or "" if the watcher has never observed anything yet.
func lastRun(ctx context.Context, jc *jennahClient, agentID string) (string, error) {
	var resp agentpb.QueryMemoryResponse
	if _, err := jc.do(ctx, http.MethodPost, memoryPath(agentID, "query"), &agentpb.QueryMemoryRequest{
		AgentInstanceId: agentID,
		Log:             &agentpb.LogQuery{Limit: 1},
	}, &resp); err != nil {
		return "", err
	}
	steps := resp.GetLog().GetSteps()
	if len(steps) == 0 {
		return "", nil
	}
	if t := steps[0].GetTimestamp(); t != nil {
		return t.AsTime().UTC().Format(time.RFC3339), nil
	}
	return "", nil
}

// knownEntities lists the labels of every entity already linked to the subject
// anchor (a one-hop MENTIONS traversal). It gives the brain context so it can lean
// toward genuinely new developments.
func knownEntities(ctx context.Context, jc *jennahClient, agentID string) ([]string, error) {
	var resp agentpb.QueryMemoryResponse
	if _, err := jc.do(ctx, http.MethodPost, memoryPath(agentID, "query"), &agentpb.QueryMemoryRequest{
		AgentInstanceId: agentID,
		Graph: &agentpb.GraphQuery{
			Start: &agentpb.GraphNodeMatch{
				Filters: []*agentpb.PropertyFilter{{Key: "NodeId", Value: structpb.NewStringValue(subjectNode)}},
			},
			Steps: []*agentpb.GraphStep{{
				Direction: agentpb.GraphDirection_GRAPH_DIRECTION_OUTGOING,
				Node:      &agentpb.GraphNodeMatch{},
			}},
			Limit: 200,
		},
	}, &resp); err != nil {
		return nil, err
	}
	seen := map[string]bool{}
	var out []string
	for _, row := range resp.GetGraph().GetRows() {
		m := row.AsMap()
		if v := rowStr(m, "n1_label"); v != "" && !seen[v] {
			seen[v] = true
			out = append(out, v)
		}
	}
	return out, nil
}

// nearestDistance runs a semantic query for headline and returns the cosine
// distance of the closest past chunk (smaller = more similar). ok is false when
// memory is empty (nothing to compare against — the item is necessarily new).
func nearestDistance(ctx context.Context, jc *jennahClient, agentID, headline string) (float64, bool, error) {
	var resp agentpb.QueryMemoryResponse
	if _, err := jc.do(ctx, http.MethodPost, memoryPath(agentID, "query"), &agentpb.QueryMemoryRequest{
		AgentInstanceId: agentID,
		Semantic:        &agentpb.SemanticQuery{QueryText: headline, Limit: 3},
	}, &resp); err != nil {
		return 0, false, err
	}
	matches := resp.GetSemantic().GetMatches()
	if len(matches) == 0 {
		return 0, false, nil
	}
	return matches[0].GetDistance(), true, nil
}

// commitRun writes the NEW developments (vector chunks), their entities/relations
// (graph nodes/edges, plus subject-[MENTIONS]->entity so the anchor enumerates the
// watch list), and one run record (log) in a single atomic commit. Graph writes are
// idempotent server-side and use deterministic content-hashed ids, so an entity
// re-seen on a later run just converges instead of fragmenting. A run with no new
// items still writes its log step, so the timeline records that the watcher ran.
func commitRun(ctx context.Context, jc *jennahClient, agentID, subject string, fresh []observation, newCount, knownCount int) error {
	var (
		vectors []*agentpb.VectorChunk
		nodes   []*agentpb.GraphNode
		edges   []*agentpb.GraphEdge
	)
	seenN, seenE := map[string]bool{}, map[string]bool{}

	addNode := func(name, typ string) string {
		nid := "n_" + hash(strings.ToLower(name))
		if !seenN[nid] {
			var props *structpb.Struct
			if strings.TrimSpace(typ) != "" {
				props = &structpb.Struct{Fields: map[string]*structpb.Value{
					"type": structpb.NewStringValue(typ),
				}}
			}
			nodes = append(nodes, &agentpb.GraphNode{NodeId: nid, Label: name, Properties: props})
			seenN[nid] = true
		}
		return nid
	}
	addEdge := func(src, dst, rel string) {
		eid := "e_" + hash(src+"|"+rel+"|"+dst)
		if !seenE[eid] {
			edges = append(edges, &agentpb.GraphEdge{
				EdgeId: eid, SourceNodeId: src, TargetNodeId: dst, RelationshipType: normRel(rel),
			})
			seenE[eid] = true
		}
	}

	for _, it := range fresh {
		vectors = append(vectors, &agentpb.VectorChunk{
			ChunkId:    randID("chunk"),
			RawContent: it.Headline + "\n" + it.Detail,
		})
		// Every named entity is linked to the subject anchor so a traversal from it
		// enumerates the whole watch list.
		for _, e := range it.Entities {
			if strings.TrimSpace(e.Name) == "" {
				continue
			}
			nid := addNode(e.Name, e.Type)
			addEdge(subjectNode, nid, "MENTIONS")
		}
		// Relations connect entities to each other (the reason the graph is worth
		// having: multi-hop "how does X relate to Y" over time).
		for _, r := range it.Relations {
			if strings.TrimSpace(r.Subject) == "" || strings.TrimSpace(r.Object) == "" {
				continue
			}
			sid := addNode(r.Subject, "")
			oid := addNode(r.Object, "")
			addEdge(subjectNode, sid, "MENTIONS")
			addEdge(subjectNode, oid, "MENTIONS")
			addEdge(sid, oid, r.Relationship)
		}
	}

	req := &agentpb.CommitMemoryRequest{
		AgentInstanceId: agentID,
		Log: &agentpb.ExecutionLogStep{
			StepId:         randID("run"),
			ThoughtProcess: "scheduled watch cycle",
			ToolUsed:       "observe",
			ToolInput:      truncate(subject, 500),
			ToolOutput:     fmt.Sprintf("%d new, %d known", newCount, knownCount),
		},
	}
	if len(vectors) > 0 {
		req.Vectors = vectors
	}
	if len(nodes) > 0 || len(edges) > 0 {
		req.Graph = &agentpb.GraphWrite{Nodes: nodes, Edges: edges}
	}

	resp, err := commit(ctx, jc, agentID, req)
	if err != nil {
		return err
	}
	printReceipt(resp)
	return nil
}

// showState prints the accumulated entity graph (a two-hop traversal from the
// subject anchor) and the recent run timeline — a read-only snapshot of what the
// watcher has learned, with no observing.
func showState(ctx context.Context, jc *jennahClient, agentID, subject string) error {
	fmt.Printf("watching: %s\n", subject)

	// Two hops from the anchor: subject -[MENTIONS]-> entity -[rel]-> other. The
	// first hop lists the watch set; the second surfaces entity-to-entity relations.
	var resp agentpb.QueryMemoryResponse
	if _, err := jc.do(ctx, http.MethodPost, memoryPath(agentID, "query"), &agentpb.QueryMemoryRequest{
		AgentInstanceId: agentID,
		Graph: &agentpb.GraphQuery{
			Start: &agentpb.GraphNodeMatch{
				Filters: []*agentpb.PropertyFilter{{Key: "NodeId", Value: structpb.NewStringValue(subjectNode)}},
			},
			Steps: []*agentpb.GraphStep{
				{Direction: agentpb.GraphDirection_GRAPH_DIRECTION_OUTGOING, Node: &agentpb.GraphNodeMatch{}},
				{Direction: agentpb.GraphDirection_GRAPH_DIRECTION_OUTGOING, Node: &agentpb.GraphNodeMatch{}},
			},
			Limit: 500,
		},
	}, &resp); err != nil {
		return err
	}
	entities := map[string]bool{}
	var rels []string
	for _, row := range resp.GetGraph().GetRows() {
		m := row.AsMap()
		if e := rowStr(m, "n1_label"); e != "" {
			entities[e] = true
		}
		// Second hop (present only on rows that traversed it): entity -[rel]-> other.
		a, rel, b := rowStr(m, "n1_label"), prettyRel(rowStr(m, "e1_type")), rowStr(m, "n2_label")
		if a != "" && b != "" && a != b {
			rels = append(rels, fmt.Sprintf("%s %s %s", a, rel, b))
		}
	}
	fmt.Printf("\nentities tracked (%d):\n", len(entities))
	for e := range entities {
		fmt.Printf("  · %s\n", e)
	}
	if len(rels) > 0 {
		fmt.Printf("\nrelations:\n")
		for _, r := range dedupStrings(rels) {
			fmt.Printf("  · %s\n", r)
		}
	}

	// Recent run timeline from the execution log (newest first).
	var lr agentpb.QueryMemoryResponse
	if _, err := jc.do(ctx, http.MethodPost, memoryPath(agentID, "query"), &agentpb.QueryMemoryRequest{
		AgentInstanceId: agentID,
		Log:             &agentpb.LogQuery{Limit: 10},
	}, &lr); err != nil {
		return err
	}
	fmt.Printf("\nrecent runs:\n")
	for _, s := range lr.GetLog().GetSteps() {
		ts := "?"
		if t := s.GetTimestamp(); t != nil {
			ts = t.AsTime().UTC().Format(time.RFC3339)
		}
		fmt.Printf("  · %s  %s\n", ts, s.GetToolOutput())
	}
	return nil
}

// createAgent provisions the agent workspace. region is the optional Jennah home
// region ("" = platform default); it's honored only at creation time because an
// agent instance is pinned to one home region for its lifetime.
func createAgent(ctx context.Context, jc *jennahClient, region string) (string, error) {
	id := randID("agent")
	var resp agentpb.CreateAgentResponse
	if _, err := jc.do(ctx, http.MethodPost, "/v1/agents", &agentpb.CreateAgentRequest{
		AgentInstanceId: id,
		AgentName:       "memwatch-demo",
		Region:          region,
	}, &resp); err != nil {
		return "", err
	}
	if got := resp.GetAgent().GetAgentInstanceId(); got != "" {
		return got, nil
	}
	return id, nil
}

func commit(ctx context.Context, jc *jennahClient, agentID string, req *agentpb.CommitMemoryRequest) (*agentpb.CommitMemoryResponse, error) {
	var resp agentpb.CommitMemoryResponse
	_, err := jc.do(ctx, http.MethodPost, memoryPath(agentID, "commit"), req, &resp)
	return &resp, err
}

func printReceipt(r *agentpb.CommitMemoryResponse) {
	ts := "?"
	if t := r.GetCommitTimestamp(); t != nil {
		ts = t.AsTime().UTC().Format(time.RFC3339)
	}
	vlog("committed: log=%d vec=%d nodes=%d edges=%d @ %s",
		r.GetExecutionLogRows(), r.GetVectorRows(), r.GetGraphNodeRows(), r.GetGraphEdgeRows(), ts)
}

// ---- HTTP client (protojson over the gateway, Bearer auth) ----

type jennahClient struct {
	endpoint string
	token    string
	hc       *http.Client
}

func (c *jennahClient) do(ctx context.Context, method, path string, in, out proto.Message) (int, error) {
	var body io.Reader
	if in != nil {
		b, err := protojson.Marshal(in)
		if err != nil {
			return 0, err
		}
		body = strings.NewReader(string(b))
	}
	req, err := http.NewRequestWithContext(ctx, method, c.endpoint+path, body)
	if err != nil {
		return 0, err
	}
	if in != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	req.Header.Set("Authorization", "Bearer "+c.token)

	resp, err := c.hc.Do(req)
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if resp.StatusCode >= 300 {
		return resp.StatusCode, fmt.Errorf("%s %s: HTTP %d: %s", method, path, resp.StatusCode, gatewayMessage(raw))
	}
	if out != nil {
		if err := (protojson.UnmarshalOptions{DiscardUnknown: true}).Unmarshal(raw, out); err != nil {
			return resp.StatusCode, fmt.Errorf("decode %s: %w", path, err)
		}
	}
	return resp.StatusCode, nil
}

func agentPath(id string) string        { return "/v1/agents/" + url.PathEscape(id) }
func memoryPath(id, verb string) string { return agentPath(id) + "/memory:" + verb }

func gatewayMessage(body []byte) string {
	var gw struct {
		Message string `json:"message"`
	}
	if err := json.Unmarshal(body, &gw); err == nil && strings.TrimSpace(gw.Message) != "" {
		return strings.TrimSpace(gw.Message)
	}
	return strings.TrimSpace(string(body))
}

// ---- local state ----

// state persists the agent id (what makes memory carry across runs) and the pinned
// subject (fixed at creation, so cron entries don't need to repeat -subject). No
// graph id tracking is needed: graph writes are idempotent server-side.
type state struct {
	AgentID string `json:"agent_id"`
	Subject string `json:"subject"`
}

// observation is one development the brain reports; the tool schema mirrors it.
type observation struct {
	Headline  string      `json:"headline"`
	Detail    string      `json:"detail"`
	Entities  []entityRef `json:"entities"`
	Relations []triple    `json:"relations"`
}

type entityRef struct {
	Name string `json:"name"`
	Type string `json:"type"`
}

type triple struct {
	Subject      string `json:"subject"`
	Relationship string `json:"relationship"`
	Object       string `json:"object"`
}

func loadState(path string) (*state, error) {
	st := &state{}
	b, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return st, nil
	}
	if err != nil {
		return nil, err
	}
	if err := json.Unmarshal(b, st); err != nil {
		return nil, err
	}
	return st, nil
}

func save(path string, st *state) {
	b, err := json.MarshalIndent(st, "", "  ")
	if err != nil {
		fmt.Fprintf(os.Stderr, "warning: marshal state: %v\n", err)
		return
	}
	if err := os.WriteFile(path, b, 0o600); err != nil {
		fmt.Fprintf(os.Stderr, "warning: write state %s: %v\n", path, err)
	}
}

// ---- helpers ----

func hash(s string) string {
	sum := sha1.Sum([]byte(s))
	return hex.EncodeToString(sum[:])[:12]
}

func randID(prefix string) string {
	var b [16]byte
	_, _ = rand.Read(b[:])
	return prefix + "_" + hex.EncodeToString(b[:])
}

// normRel turns a verb phrase into an edge RelationshipType, e.g. "competes with" -> "COMPETES_WITH".
func normRel(s string) string {
	var b strings.Builder
	for _, r := range strings.ToUpper(strings.TrimSpace(s)) {
		switch {
		case r >= 'A' && r <= 'Z', r >= '0' && r <= '9':
			b.WriteRune(r)
		default:
			b.WriteRune('_')
		}
	}
	out := strings.Trim(b.String(), "_")
	if out == "" {
		return "RELATED_TO"
	}
	return out
}

// prettyRel renders a stored RelationshipType back as a readable phrase.
func prettyRel(s string) string {
	if s == "" {
		return "→"
	}
	return strings.ToLower(strings.ReplaceAll(s, "_", " "))
}

func rowStr(m map[string]any, key string) string {
	if v, ok := m[key].(string); ok {
		return v
	}
	return ""
}

func singleLine(s string) string {
	return strings.Join(strings.Fields(s), " ")
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n]
}

func dedupStrings(in []string) []string {
	seen := map[string]bool{}
	var out []string
	for _, s := range in {
		if !seen[s] {
			seen[s] = true
			out = append(out, s)
		}
	}
	return out
}

func distStr(ok bool, d float64) string {
	if !ok {
		return "∅"
	}
	return fmt.Sprintf("%.3f", d)
}

func envOr(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

// envOr2 returns val when it's non-empty (a flag was passed), otherwise the value
// of env var key. Unlike envOr, the caller-supplied value wins — so an explicit
// flag overrides the env var, and the env var is never a flag default (keeping
// secrets out of -help).
func envOr2(val, key string) string {
	if strings.TrimSpace(val) != "" {
		return val
	}
	return os.Getenv(key)
}

// vlog prints a dim diagnostic line to stdout, only when -verbose is set.
func vlog(format string, args ...any) {
	if verbose {
		fmt.Printf("  \033[2m%s\033[0m\n", fmt.Sprintf(format, args...))
	}
}

func fatal(format string, args ...any) {
	fmt.Fprintf(os.Stderr, "memwatch: "+format+"\n", args...)
	os.Exit(1)
}
