use inspect_core::types::EntityReview;
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

use crate::prompts;
use crate::state::AppState;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Finding {
    pub issue: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub evidence: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub severity: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file: Option<String>,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
}

#[derive(Deserialize)]
struct Choice {
    message: Message,
}

#[derive(Deserialize)]
struct Message {
    content: Option<String>,
}

#[derive(Deserialize)]
struct IssuesResponse {
    #[serde(default)]
    issues: Vec<serde_json::Value>,
}

// --- Agentic review types ---

pub struct AgentContext {
    pub entity_reviews: Vec<EntityReview>,
    pub repo: String,
    pub base_sha: String,
    pub head_sha: String,
    pub pr_title: String,
    pub triage_section: String,
}

#[derive(Debug, Deserialize)]
struct ResponsesResponse {
    id: String,
    output: Vec<ResponseOutput>,
    #[allow(dead_code)]
    status: String,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum ResponseOutput {
    #[serde(rename = "message")]
    Message { content: Vec<ContentPart> },
    #[serde(rename = "function_call")]
    FunctionCall {
        name: String,
        arguments: String,
        call_id: String,
    },
}

#[derive(Debug, Deserialize)]
struct ContentPart {
    #[allow(dead_code)]
    text: String,
}

enum AgentStep {
    ToolCalls(Vec<(String, String, String)>), // (name, arguments, call_id)
    Finished(Vec<Finding>),
    Evidence(Vec<EvidenceReport>),
    KeptIndices(Vec<usize>),
    Empty,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct EvidenceReport {
    finding_index: usize,
    finding_issue: String,
    evidence_summary: String,
    code_snippets: Vec<String>,
    verdict: String, // "rescue" or "reject"
}

/// Call OpenAI chat completions API.
async fn call_openai(
    state: &AppState,
    system: &str,
    prompt: &str,
    temperature: f64,
    seed: Option<u64>,
) -> Result<String, String> {
    let mut body = serde_json::json!({
        "model": state.openai_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
    });
    if let Some(s) = seed {
        body["seed"] = serde_json::json!(s);
    }

    let resp = state
        .http
        .post("https://api.openai.com/v1/chat/completions")
        .header("Authorization", format!("Bearer {}", state.openai_api_key))
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("request failed: {e}"))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        return Err(format!("OpenAI API error {status}: {text}"));
    }

    let chat: ChatResponse = resp.json().await.map_err(|e| format!("parse failed: {e}"))?;
    let content = chat
        .choices
        .first()
        .and_then(|c| c.message.content.clone())
        .unwrap_or_default();

    Ok(content)
}

/// Call Anthropic Messages API.
async fn call_anthropic(
    state: &AppState,
    system: &str,
    prompt: &str,
    temperature: f64,
) -> Result<String, String> {
    let api_key = state
        .anthropic_api_key
        .as_ref()
        .ok_or_else(|| "ANTHROPIC_API_KEY not set".to_string())?;

    let body = serde_json::json!({
        "model": state.anthropic_model,
        "max_tokens": 4096,
        "system": system,
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
    });

    let resp = state
        .http
        .post("https://api.anthropic.com/v1/messages")
        .header("x-api-key", api_key)
        .header("anthropic-version", "2023-06-01")
        .header("content-type", "application/json")
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("Anthropic request failed: {e}"))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        return Err(format!("Anthropic API error {status}: {text}"));
    }

    let body: serde_json::Value = resp.json().await.map_err(|e| format!("parse failed: {e}"))?;
    let content = body
        .get("content")
        .and_then(|c| c.as_array())
        .and_then(|arr| arr.first())
        .and_then(|block| block.get("text"))
        .and_then(|t| t.as_str())
        .unwrap_or_default()
        .to_string();

    Ok(content)
}

/// Strip markdown code fences and parse JSON issues.
fn parse_issues(text: &str) -> Vec<Finding> {
    let cleaned = strip_code_fences(text);

    // Try direct parse first
    if let Ok(resp) = serde_json::from_str::<IssuesResponse>(&cleaned) {
        return extract_findings(resp);
    }

    // Fallback: find JSON object anywhere in the text (handles prose before/after JSON)
    if let Some(start) = cleaned.find('{') {
        // Find matching closing brace
        let mut depth = 0i32;
        let mut end = start;
        for (i, ch) in cleaned[start..].char_indices() {
            match ch {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        end = start + i + 1;
                        break;
                    }
                }
                _ => {}
            }
        }
        if depth == 0 {
            let json_str = &cleaned[start..end];
            if let Ok(resp) = serde_json::from_str::<IssuesResponse>(json_str) {
                info!("Parsed JSON from embedded response (offset {})", start);
                return extract_findings(resp);
            }
        }
    }

    warn!("Failed to parse LLM response as JSON, text starts with: {}", &cleaned[..cleaned.len().min(100)]);
    Vec::new()
}

fn extract_findings(resp: IssuesResponse) -> Vec<Finding> {
    resp.issues
        .into_iter()
        .filter_map(|v| match v {
            serde_json::Value::String(s) => Some(Finding {
                issue: s,
                evidence: None,
                severity: None,
                file: None,
            }),
            serde_json::Value::Object(map) => {
                let issue = map
                    .get("issue")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                if issue.is_empty() {
                    return None;
                }
                Some(Finding {
                    issue,
                    evidence: map.get("evidence").and_then(|v| v.as_str()).map(String::from),
                    severity: map.get("severity").and_then(|v| v.as_str()).map(String::from),
                    file: map.get("file").and_then(|v| v.as_str()).map(String::from),
                })
            }
            _ => None,
        })
        .collect()
}

fn strip_code_fences(text: &str) -> String {
    let trimmed = text.trim();
    if trimmed.starts_with("```") {
        let after_fence = &trimmed[3..];
        // Skip optional language tag
        let content = if after_fence.starts_with("json") {
            &after_fence[4..]
        } else {
            after_fence
        };
        // Find closing fence
        if let Some(end) = content.rfind("```") {
            return content[..end].trim().to_string();
        }
        return content.trim().to_string();
    }
    trimmed.to_string()
}

/// Extract file paths and basenames from a unified diff.
fn extract_diff_files(diff: &str) -> std::collections::HashSet<String> {
    let mut files = std::collections::HashSet::new();
    for line in diff.lines() {
        if line.starts_with("+++ b/") || line.starts_with("--- a/") {
            let path = &line[6..];
            if path != "/dev/null" && !path.is_empty() {
                files.insert(path.to_string());
                if let Some(basename) = path.rsplit('/').next() {
                    files.insert(basename.to_string());
                }
            }
        }
    }
    files
}

const CODE_EXTENSIONS: &[&str] = &[
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".go", ".rs", ".rb",
    ".c", ".cpp", ".cs", ".swift", ".kt", ".scala", ".hbs", ".erb",
    ".ex", ".exs", ".hcl", ".dart",
];

/// Drop findings that reference code files not present in the diff.
fn structural_file_filter(
    findings: Vec<Finding>,
    diff_files: &std::collections::HashSet<String>,
) -> Vec<Finding> {
    if diff_files.is_empty() {
        return findings;
    }
    let lower_basenames: std::collections::HashSet<String> = diff_files
        .iter()
        .map(|f| f.rsplit('/').next().unwrap_or(f).to_lowercase())
        .collect();

    findings
        .into_iter()
        .filter(|f| {
            let text = f.issue.to_lowercase();
            for word in text.replace('/', " / ").split_whitespace() {
                if CODE_EXTENSIONS.iter().any(|ext| word.ends_with(ext)) {
                    let base = word.rsplit('/').next().unwrap_or(word);
                    if !lower_basenames.contains(base) {
                        return false;
                    }
                }
            }
            true
        })
        .collect()
}

/// Validation with seed support.
async fn validate_findings_seeded(
    state: &AppState,
    pr_title: &str,
    diff: &str,
    candidates: &[Finding],
    seed: Option<u64>,
) -> Result<Vec<Finding>, String> {
    let candidates_text: String = candidates
        .iter()
        .enumerate()
        .map(|(i, f)| {
            let mut line = format!("{}. {}", i + 1, f.issue);
            if let Some(ref ev) = f.evidence {
                line.push_str(&format!("\n   Evidence: {ev}"));
            }
            line
        })
        .collect::<Vec<_>>()
        .join("\n");

    let prompt = prompts::format_validate_prompt(pr_title, diff, &candidates_text);
    let text = call_openai(state, prompts::SYSTEM_VALIDATE, &prompt, 0.0, seed).await?;
    Ok(parse_issues(&text))
}

pub async fn review_raw_lenses(
    state: &AppState,
    pr_title: &str,
    diff: &str,
    triage_section: &str,
    max_findings: usize,
) -> Vec<Finding> {
    let truncated = prompts::truncate_diff(diff, 65_000);
    let diff_files = extract_diff_files(diff);

    let p_data = prompts::format_lens_prompt(prompts::PROMPT_LENS_DATA, pr_title, triage_section, &truncated);
    let p_conc = prompts::format_lens_prompt(prompts::PROMPT_LENS_CONCURRENCY, pr_title, triage_section, &truncated);
    let p_cont = prompts::format_lens_prompt(prompts::PROMPT_LENS_CONTRACTS, pr_title, triage_section, &truncated);
    let p_sec = prompts::format_lens_prompt(prompts::PROMPT_LENS_SECURITY, pr_title, triage_section, &truncated);
    let p_typo = prompts::format_lens_prompt(prompts::PROMPT_LENS_TYPOS, pr_title, triage_section, &truncated);
    let p_rt = prompts::format_lens_prompt(prompts::PROMPT_LENS_RUNTIME, pr_title, triage_section, &truncated);
    let p_gen = prompts::format_deep_prompt(pr_title, triage_section, &truncated);

    let (r1, r2, r3, r4, r5, r6, r7, r8, r9) = tokio::join!(
        call_openai(state, prompts::SYSTEM_DATA, &p_data, 0.0, Some(42)),
        call_openai(state, prompts::SYSTEM_CONCURRENCY, &p_conc, 0.0, Some(42)),
        call_openai(state, prompts::SYSTEM_CONTRACTS, &p_cont, 0.0, Some(42)),
        call_openai(state, prompts::SYSTEM_SECURITY, &p_sec, 0.0, Some(42)),
        call_openai(state, prompts::SYSTEM_TYPOS, &p_typo, 0.0, Some(42)),
        call_openai(state, prompts::SYSTEM_RUNTIME, &p_rt, 0.0, Some(42)),
        call_openai(state, prompts::SYSTEM_REVIEW, &p_gen, 0.0, Some(42)),
        call_anthropic(state, prompts::SYSTEM_REVIEW, &p_gen, 0.0),
        call_anthropic(state, prompts::SYSTEM_REVIEW, &p_gen, 0.1),
    );

    let mut all_findings: Vec<Finding> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();

    for result in [r1, r2, r3, r4, r5, r6, r7, r8, r9] {
        if let Ok(text) = result {
            for f in parse_issues(&text) {
                let key: String = f.issue.to_lowercase().chars().take(80).collect();
                if seen.insert(key) {
                    all_findings.push(f);
                }
            }
        }
    }

    let pre_filter = all_findings.len();
    all_findings = structural_file_filter(all_findings, &diff_files);
    info!(
        "RAW LENSES: {} after dedup, {} after structural filter (no validation)",
        pre_filter, all_findings.len()
    );

    all_findings.into_iter().take(max_findings).collect()
}

/// 9 parallel lenses + structural filter + blind validation.
async fn review_hybrid_inner(
    state: &AppState,
    pr_title: &str,
    diff: &str,
    triage_section: &str,
    max_findings: usize,
) -> Vec<Finding> {
    let truncated = prompts::truncate_diff(diff, 65_000);
    let diff_files = extract_diff_files(diff);

    // Build 6 specialized lens prompts
    let p_data = prompts::format_lens_prompt(prompts::PROMPT_LENS_DATA, pr_title, triage_section, &truncated);
    let p_conc = prompts::format_lens_prompt(prompts::PROMPT_LENS_CONCURRENCY, pr_title, triage_section, &truncated);
    let p_cont = prompts::format_lens_prompt(prompts::PROMPT_LENS_CONTRACTS, pr_title, triage_section, &truncated);
    let p_sec = prompts::format_lens_prompt(prompts::PROMPT_LENS_SECURITY, pr_title, triage_section, &truncated);
    let p_typo = prompts::format_lens_prompt(prompts::PROMPT_LENS_TYPOS, pr_title, triage_section, &truncated);
    let p_rt = prompts::format_lens_prompt(prompts::PROMPT_LENS_RUNTIME, pr_title, triage_section, &truncated);

    // 3 general lens prompts
    let p_gen = prompts::format_deep_prompt(pr_title, triage_section, &truncated);

    // 9 lenses in parallel: 7 GPT + 2 Sonnet (cross-model diversity)
    let (r1, r2, r3, r4, r5, r6, r7, r8, r9) = tokio::join!(
        call_openai(state, prompts::SYSTEM_DATA, &p_data, 0.0, Some(42)),
        call_openai(state, prompts::SYSTEM_CONCURRENCY, &p_conc, 0.0, Some(42)),
        call_openai(state, prompts::SYSTEM_CONTRACTS, &p_cont, 0.0, Some(42)),
        call_openai(state, prompts::SYSTEM_SECURITY, &p_sec, 0.0, Some(42)),
        call_openai(state, prompts::SYSTEM_TYPOS, &p_typo, 0.0, Some(42)),
        call_openai(state, prompts::SYSTEM_RUNTIME, &p_rt, 0.0, Some(42)),
        call_openai(state, prompts::SYSTEM_REVIEW, &p_gen, 0.0, Some(42)),
        call_anthropic(state, prompts::SYSTEM_REVIEW, &p_gen, 0.0),
        call_anthropic(state, prompts::SYSTEM_REVIEW, &p_gen, 0.1),
    );

    // Merge + dedup
    let mut all_findings: Vec<Finding> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();

    for result in [r1, r2, r3, r4, r5, r6, r7, r8, r9] {
        if let Ok(text) = result {
            for f in parse_issues(&text) {
                let key: String = f.issue.to_lowercase().chars().take(80).collect();
                if seen.insert(key) {
                    all_findings.push(f);
                }
            }
        } else {
            warn!("Lens failed: {:?}", result.err());
        }
    }

    info!("FUNNEL merge: {} findings from 9 lenses", all_findings.len());

    if all_findings.is_empty() {
        return Vec::new();
    }

    // Structural file filter
    let pre_filter = all_findings.len();
    all_findings = structural_file_filter(all_findings, &diff_files);
    info!(
        "FUNNEL structural filter: {} -> {} ({} dropped)",
        pre_filter, all_findings.len(), pre_filter - all_findings.len()
    );

    if all_findings.is_empty() {
        return Vec::new();
    }

    // Skip validation if few findings
    if all_findings.len() <= 2 {
        info!("FUNNEL skip validation (<= 2 findings), returning {} as-is", all_findings.len());
        return all_findings;
    }

    // Validation pass with seed=42
    let pre_validation = all_findings.len();
    match validate_findings_seeded(state, pr_title, &truncated, &all_findings, Some(42)).await {
        Ok(validated) => {
            let post_validation = validated.len();
            let final_count = post_validation.min(max_findings);
            info!(
                "FUNNEL validation: {} -> {} ({} dropped), top-{} = {}",
                pre_validation, post_validation, pre_validation - post_validation,
                max_findings, final_count
            );
            // Log which findings survived
            for f in &validated {
                info!("FUNNEL survived: {}", &f.issue[..f.issue.len().min(100)]);
            }
            validated.into_iter().take(max_findings).collect()
        }
        Err(e) => {
            warn!("Validation failed: {e}");
            all_findings.into_iter().take(max_findings).collect()
        }
    }
}

// --- Responses API for agentic review ---

/// Call OpenAI Responses API. Initial call uses instructions + input.
/// Continuation calls use previous_response_id + tool results.
async fn call_responses_api(
    state: &AppState,
    instructions: &str,
    input: Option<&str>,
    tools: &[serde_json::Value],
    previous_response_id: Option<&str>,
    tool_outputs: Option<Vec<serde_json::Value>>,
) -> Result<ResponsesResponse, String> {
    let mut body = serde_json::json!({
        "model": state.openai_model,
        "instructions": instructions,
        "tools": tools,
        "temperature": 0.0,
    });

    if let Some(prev_id) = previous_response_id {
        body["previous_response_id"] = serde_json::json!(prev_id);
        if let Some(outputs) = tool_outputs {
            body["input"] = serde_json::Value::Array(outputs);
        }
    } else if let Some(inp) = input {
        body["input"] = serde_json::json!(inp);
    }

    let resp = state
        .http
        .post("https://api.openai.com/v1/responses")
        .header("Authorization", format!("Bearer {}", state.openai_api_key))
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("responses API request failed: {e}"))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        return Err(format!("Responses API error {status}: {text}"));
    }

    resp.json::<ResponsesResponse>()
        .await
        .map_err(|e| format!("parse responses failed: {e}"))
}

/// Parse agent response into tool calls, finished findings, or empty.
fn parse_agent_step(output: &[ResponseOutput]) -> AgentStep {
    let mut tool_calls = Vec::new();

    for item in output {
        match item {
            ResponseOutput::FunctionCall {
                name,
                arguments,
                call_id,
            } => {
                if name == "submit_kept_indices" {
                    if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(arguments) {
                        let indices = parsed
                            .get("kept_indices")
                            .and_then(|f| f.as_array())
                            .map(|arr| {
                                arr.iter()
                                    .filter_map(|v| v.as_u64().map(|n| n as usize))
                                    .collect()
                            })
                            .unwrap_or_default();
                        return AgentStep::KeptIndices(indices);
                    }
                    return AgentStep::KeptIndices(Vec::new());
                }
                if name == "submit_findings" {
                    // Parse findings from arguments
                    if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(arguments) {
                        let findings = parsed
                            .get("findings")
                            .and_then(|f| f.as_array())
                            .map(|arr| {
                                arr.iter()
                                    .filter_map(|v| {
                                        let issue = v.get("issue")?.as_str()?.to_string();
                                        Some(Finding {
                                            issue,
                                            evidence: v
                                                .get("evidence")
                                                .and_then(|e| e.as_str())
                                                .map(String::from),
                                            severity: v
                                                .get("severity")
                                                .and_then(|s| s.as_str())
                                                .map(String::from),
                                            file: v
                                                .get("file")
                                                .and_then(|f| f.as_str())
                                                .map(String::from),
                                        })
                                    })
                                    .collect()
                            })
                            .unwrap_or_default();
                        return AgentStep::Finished(findings);
                    }
                    return AgentStep::Finished(Vec::new());
                }
                if name == "submit_evidence" {
                    // Parse evidence reports from arguments
                    if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(arguments) {
                        let reports = parsed
                            .get("reports")
                            .and_then(|r| r.as_array())
                            .map(|arr| {
                                arr.iter()
                                    .filter_map(|v| {
                                        Some(EvidenceReport {
                                            finding_index: v.get("finding_index")?.as_u64()? as usize,
                                            finding_issue: v.get("finding_issue")?.as_str()?.to_string(),
                                            evidence_summary: v.get("evidence_summary")?.as_str()?.to_string(),
                                            code_snippets: v.get("code_snippets")
                                                .and_then(|s| s.as_array())
                                                .map(|a| a.iter().filter_map(|s| s.as_str().map(String::from)).collect())
                                                .unwrap_or_default(),
                                            verdict: v.get("verdict")?.as_str()?.to_string(),
                                        })
                                    })
                                    .collect()
                            })
                            .unwrap_or_default();
                        return AgentStep::Evidence(reports);
                    }
                    return AgentStep::Evidence(Vec::new());
                }
                tool_calls.push((name.clone(), arguments.clone(), call_id.clone()));
            }
            ResponseOutput::Message { .. } => {}
        }
    }

    if tool_calls.is_empty() {
        AgentStep::Empty
    } else {
        AgentStep::ToolCalls(tool_calls)
    }
}

/// Build tool definitions for challenge agent: same tools but submit_kept_indices instead of submit_findings.
/// The agent returns which findings to KEEP by their 1-indexed number, preserving original text.
fn build_challenge_tool_definitions() -> Vec<serde_json::Value> {
    vec![
        serde_json::json!({
            "type": "function",
            "name": "get_entity",
            "description": "Get full details of a changed entity from triage. Returns before/after code, risk score, blast radius, dependents, dependencies.",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_name": { "type": "string", "description": "Name of the entity" },
                    "file_path": { "type": "string", "description": "Optional file path to disambiguate" }
                },
                "required": ["entity_name"]
            }
        }),
        serde_json::json!({
            "type": "function",
            "name": "read_file",
            "description": "Read a file from the repo at head or base commit. Use for full context, imports, type definitions, surrounding code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "File path relative to repo root" },
                    "ref": { "type": "string", "enum": ["head", "base"], "description": "Which commit to read from" },
                    "start_line": { "type": "integer", "description": "Optional 1-indexed start line" },
                    "end_line": { "type": "integer", "description": "Optional end line, max 200 lines from start" }
                },
                "required": ["path", "ref"]
            }
        }),
        serde_json::json!({
            "type": "function",
            "name": "search_code",
            "description": "Search the repo for code matching a query. Find callers, implementations, usages of a function or type.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "Search query (function name, type name, etc.)" },
                    "path_prefix": { "type": "string", "description": "Optional path filter like 'src/' or 'lib/'" }
                },
                "required": ["query"]
            }
        }),
        serde_json::json!({
            "type": "function",
            "name": "get_dependents",
            "description": "Get entities that depend on (call/reference) a given entity. Shows blast radius.",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_name": { "type": "string", "description": "Name of the entity" },
                    "file_path": { "type": "string", "description": "Optional file path to disambiguate" }
                },
                "required": ["entity_name"]
            }
        }),
        serde_json::json!({
            "type": "function",
            "name": "get_dependencies",
            "description": "Get entities that a given entity depends on (calls/imports). Shows what this entity relies on.",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_name": { "type": "string", "description": "Name of the entity" },
                    "file_path": { "type": "string", "description": "Optional file path to disambiguate" }
                },
                "required": ["entity_name"]
            }
        }),
        serde_json::json!({
            "type": "function",
            "name": "submit_kept_indices",
            "description": "Submit the indices (1-indexed numbers) of findings you want to KEEP. Only call this once when done investigating. Do NOT rewrite findings - just return their numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "kept_indices": {
                        "type": "array",
                        "items": { "type": "integer" },
                        "description": "List of 1-indexed finding numbers to keep (e.g. [1, 3, 5] keeps findings #1, #3, #5)"
                    }
                },
                "required": ["kept_indices"]
            }
        }),
    ]
}

fn execute_get_entity(ctx: &AgentContext, args: &str) -> String {
    let parsed: serde_json::Value = match serde_json::from_str(args) {
        Ok(v) => v,
        Err(_) => return r#"{"error": "invalid arguments"}"#.to_string(),
    };

    let name = parsed
        .get("entity_name")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let file_filter = parsed.get("file_path").and_then(|v| v.as_str());

    let entity = ctx.entity_reviews.iter().find(|e| {
        let name_match = e.entity_name == name
            || e.entity_name.ends_with(&format!("::{name}"))
            || e.entity_name.ends_with(&format!(".{name}"));
        let file_match = file_filter.map_or(true, |f| e.file_path.contains(f));
        name_match && file_match
    });

    match entity {
        Some(e) => {
            let before: String = e
                .before_content
                .as_ref()
                .map(|s| s.chars().take(3000).collect())
                .unwrap_or_default();
            let after: String = e
                .after_content
                .as_ref()
                .map(|s| s.chars().take(3000).collect())
                .unwrap_or_default();

            let dependents: Vec<serde_json::Value> = e
                .dependent_names
                .iter()
                .take(15)
                .map(|(n, f)| serde_json::json!({"name": n, "file": f}))
                .collect();
            let dependencies: Vec<serde_json::Value> = e
                .dependency_names
                .iter()
                .take(15)
                .map(|(n, f)| serde_json::json!({"name": n, "file": f}))
                .collect();

            serde_json::json!({
                "entity_name": e.entity_name,
                "entity_type": e.entity_type,
                "file_path": e.file_path,
                "change_type": format!("{:?}", e.change_type),
                "risk_score": e.risk_score,
                "blast_radius": e.blast_radius,
                "is_public_api": e.is_public_api,
                "before_content": before,
                "after_content": after,
                "dependents": dependents,
                "dependencies": dependencies,
            })
            .to_string()
        }
        None => {
            // List available entities to help the agent
            let available: Vec<String> = ctx
                .entity_reviews
                .iter()
                .take(30)
                .map(|e| format!("{} ({})", e.entity_name, e.file_path))
                .collect();
            serde_json::json!({
                "error": "entity not found",
                "available": available,
            })
            .to_string()
        }
    }
}

async fn execute_read_file(
    state: &AppState,
    ctx: &AgentContext,
    args: &str,
) -> String {
    let parsed: serde_json::Value = match serde_json::from_str(args) {
        Ok(v) => v,
        Err(_) => return r#"{"error": "invalid arguments"}"#.to_string(),
    };

    let path = match parsed.get("path").and_then(|v| v.as_str()) {
        Some(p) => p,
        None => return r#"{"error": "path is required"}"#.to_string(),
    };

    let sha = match parsed.get("ref").and_then(|v| v.as_str()) {
        Some("base") => &ctx.base_sha,
        _ => &ctx.head_sha,
    };

    let start_line = parsed
        .get("start_line")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize);
    let end_line = parsed
        .get("end_line")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize);

    let encoded_path = urlencoding::encode(path);
    let url = format!(
        "https://api.github.com/repos/{}/contents/{}?ref={}",
        ctx.repo, encoded_path, sha
    );

    let resp = match state
        .http
        .get(&url)
        .header("Authorization", format!("token {}", state.github_token))
        .header("Accept", "application/vnd.github.v3+json")
        .header("User-Agent", "inspect-api")
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => return format!(r#"{{"error": "request failed: {e}"}}"#),
    };

    if !resp.status().is_success() {
        let status = resp.status();
        return format!(r#"{{"error": "GitHub API {status} for {path}"}}"#);
    }

    let body: serde_json::Value = match resp.json().await {
        Ok(v) => v,
        Err(e) => return format!(r#"{{"error": "parse failed: {e}"}}"#),
    };

    let content_b64 = body
        .get("content")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    // GitHub returns base64 with newlines
    let cleaned: String = content_b64.chars().filter(|c| !c.is_whitespace()).collect();
    let decoded = match base64::Engine::decode(&base64::engine::general_purpose::STANDARD, &cleaned)
    {
        Ok(bytes) => String::from_utf8_lossy(&bytes).to_string(),
        Err(_) => return r#"{"error": "base64 decode failed"}"#.to_string(),
    };

    // Apply line range
    let lines: Vec<&str> = decoded.lines().collect();
    let start = start_line.unwrap_or(1).saturating_sub(1).min(lines.len());
    let end = end_line
        .unwrap_or(lines.len())
        .min(start + 200)
        .min(lines.len())
        .max(start);

    let selected: String = lines[start..end].join("\n");

    serde_json::json!({
        "path": path,
        "lines": format!("{}-{}", start + 1, end),
        "content": selected,
    })
    .to_string()
}

async fn execute_search_code(
    state: &AppState,
    ctx: &AgentContext,
    args: &str,
) -> String {
    let parsed: serde_json::Value = match serde_json::from_str(args) {
        Ok(v) => v,
        Err(_) => return r#"{"error": "invalid arguments"}"#.to_string(),
    };

    let query = match parsed.get("query").and_then(|v| v.as_str()) {
        Some(q) => q,
        None => return r#"{"error": "query is required"}"#.to_string(),
    };

    let path_prefix = parsed.get("path_prefix").and_then(|v| v.as_str());

    let mut search_q = format!("{query} repo:{}", ctx.repo);
    if let Some(prefix) = path_prefix {
        search_q.push_str(&format!(" path:{prefix}"));
    }

    let url = format!(
        "https://api.github.com/search/code?q={}&per_page=20",
        urlencoding::encode(&search_q)
    );

    let resp = match state
        .http
        .get(&url)
        .header("Authorization", format!("token {}", state.github_token))
        .header("Accept", "application/vnd.github.v3+json")
        .header("User-Agent", "inspect-api")
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => return format!(r#"{{"error": "search failed: {e}"}}"#),
    };

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        return format!(r#"{{"error": "GitHub search API {status}: {text}"}}"#);
    }

    let body: serde_json::Value = match resp.json().await {
        Ok(v) => v,
        Err(e) => return format!(r#"{{"error": "parse failed: {e}"}}"#),
    };

    let items = body
        .get("items")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();

    let results: Vec<serde_json::Value> = items
        .iter()
        .take(20)
        .filter_map(|item| {
            let path = item.get("path")?.as_str()?;
            let name = item.get("name")?.as_str()?;
            Some(serde_json::json!({
                "path": path,
                "name": name,
            }))
        })
        .collect();

    serde_json::json!({
        "total_count": body.get("total_count").and_then(|v| v.as_u64()).unwrap_or(0),
        "results": results,
    })
    .to_string()
}

fn execute_get_dependents(ctx: &AgentContext, args: &str) -> String {
    let parsed: serde_json::Value = match serde_json::from_str(args) {
        Ok(v) => v,
        Err(_) => return r#"{"error": "invalid arguments"}"#.to_string(),
    };

    let name = parsed
        .get("entity_name")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let file_filter = parsed.get("file_path").and_then(|v| v.as_str());

    let entity = ctx.entity_reviews.iter().find(|e| {
        let name_match = e.entity_name == name
            || e.entity_name.ends_with(&format!("::{name}"))
            || e.entity_name.ends_with(&format!(".{name}"));
        let file_match = file_filter.map_or(true, |f| e.file_path.contains(f));
        name_match && file_match
    });

    match entity {
        Some(e) => {
            let dependents: Vec<serde_json::Value> = e
                .dependent_names
                .iter()
                .take(20)
                .map(|(n, f)| serde_json::json!({"name": n, "file": f}))
                .collect();
            serde_json::json!({
                "entity": e.entity_name,
                "dependent_count": e.dependent_count,
                "blast_radius": e.blast_radius,
                "dependents": dependents,
            })
            .to_string()
        }
        None => r#"{"error": "entity not found in triage"}"#.to_string(),
    }
}

fn execute_get_dependencies(ctx: &AgentContext, args: &str) -> String {
    let parsed: serde_json::Value = match serde_json::from_str(args) {
        Ok(v) => v,
        Err(_) => return r#"{"error": "invalid arguments"}"#.to_string(),
    };

    let name = parsed
        .get("entity_name")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let file_filter = parsed.get("file_path").and_then(|v| v.as_str());

    let entity = ctx.entity_reviews.iter().find(|e| {
        let name_match = e.entity_name == name
            || e.entity_name.ends_with(&format!("::{name}"))
            || e.entity_name.ends_with(&format!(".{name}"));
        let file_match = file_filter.map_or(true, |f| e.file_path.contains(f));
        name_match && file_match
    });

    match entity {
        Some(e) => {
            let dependencies: Vec<serde_json::Value> = e
                .dependency_names
                .iter()
                .take(20)
                .map(|(n, f)| serde_json::json!({"name": n, "file": f}))
                .collect();
            serde_json::json!({
                "entity": e.entity_name,
                "dependency_count": e.dependency_count,
                "dependencies": dependencies,
            })
            .to_string()
        }
        None => r#"{"error": "entity not found in triage"}"#.to_string(),
    }
}

/// Agentic challenge: agent tries to DISPROVE each candidate finding using tools.
/// Starts skeptical, only keeps what it can't disprove.
async fn challenge_agentic(
    state: &AppState,
    ctx: &AgentContext,
    candidates: &[Finding],
    max_findings: usize,
) -> (Vec<Finding>, usize, usize) {
    let tools = build_challenge_tool_definitions();
    let instructions = prompts::SYSTEM_AGENT_CHALLENGE;
    let initial_prompt = prompts::build_agent_challenge_prompt(ctx, candidates);

    let mut resp = match call_responses_api(
        state,
        instructions,
        Some(&initial_prompt),
        &tools,
        None,
        None,
    )
    .await
    {
        Ok(r) => r,
        Err(e) => {
            warn!("Agentic challenge initial call failed: {e}");
            return (candidates.iter().take(max_findings).cloned().collect(), 0, 0);
        }
    };

    // Scale iterations with candidate count (more findings = more tool calls needed)
    let max_iterations = if candidates.len() > 20 { 50 } else { 25 };
    let mut total_tool_calls = 0usize;
    let mut iterations = 0usize;

    loop {
        iterations += 1;
        if iterations > max_iterations {
            warn!("Agentic challenge hit max iterations ({max_iterations})");
            break;
        }

        match parse_agent_step(&resp.output) {
            AgentStep::KeptIndices(indices) => {
                // Map 1-indexed indices back to original candidates
                let kept: Vec<Finding> = indices
                    .iter()
                    .filter_map(|&i| candidates.get(i.wrapping_sub(1)).cloned())
                    .collect();
                info!(
                    "Agentic challenge finished (index-based): kept {}/{} (indices: {:?}) after {} iterations, {} tool calls",
                    kept.len(), candidates.len(), indices, iterations, total_tool_calls
                );
                return (kept.into_iter().take(max_findings).collect(), iterations, total_tool_calls);
            }
            AgentStep::Finished(findings) => {
                // Fallback: agent called submit_findings instead of submit_kept_indices
                warn!("Agentic challenge used submit_findings instead of submit_kept_indices, using text match fallback");
                return (findings.into_iter().take(max_findings).collect(), iterations, total_tool_calls);
            }
            AgentStep::ToolCalls(calls) => {
                total_tool_calls += calls.len();
                info!(
                    "Agentic challenge iter {}: {} tool calls (total: {})",
                    iterations, calls.len(), total_tool_calls
                );

                let mut tool_outputs = Vec::new();
                for (name, arguments, call_id) in &calls {
                    let result = match name.as_str() {
                        "get_entity" => execute_get_entity(ctx, arguments),
                        "read_file" => execute_read_file(state, ctx, arguments).await,
                        "search_code" => execute_search_code(state, ctx, arguments).await,
                        "get_dependents" => execute_get_dependents(ctx, arguments),
                        "get_dependencies" => execute_get_dependencies(ctx, arguments),
                        _ => format!(r#"{{"error": "unknown tool: {name}"}}"#),
                    };

                    tool_outputs.push(serde_json::json!({
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": result,
                    }));
                }

                let prev_id = resp.id.clone();
                resp = match call_responses_api(
                    state,
                    instructions,
                    None,
                    &tools,
                    Some(&prev_id),
                    Some(tool_outputs),
                )
                .await
                {
                    Ok(r) => r,
                    Err(e) => {
                        warn!("Agentic challenge continuation failed: {e}");
                        break;
                    }
                };
            }
            AgentStep::Evidence(_) => {
                info!("Agentic challenge got evidence instead of findings");
                break;
            }
            AgentStep::Empty => {
                info!("Agentic challenge returned empty");
                break;
            }
        }
    }

    warn!("Agentic challenge did not submit findings, returning candidates as fallback");
    (candidates.iter().take(max_findings).cloned().collect(), iterations, total_tool_calls)
}

/// Hybrid v20: v10 pipeline + agentic challenge pass.
/// Runs the standard 9-lens + blind validation (v10), then passes survivors
/// through an agent with tools that tries to disprove each finding.
pub async fn review_hybrid_v20(
    state: &AppState,
    pr_title: &str,
    diff: &str,
    triage_section: &str,
    max_findings: usize,
    ctx: &AgentContext,
) -> (Vec<Finding>, usize, usize) {
    // Step 1: Run v10 pipeline
    let v10_results = review_hybrid_inner(state, pr_title, diff, triage_section, max_findings).await;

    if v10_results.is_empty() {
        return (v10_results, 0, 0);
    }

    info!("v20: v10 produced {} findings, sending to agentic challenge", v10_results.len());

    // Step 2: Agentic challenge - try to disprove each finding
    let (challenged, iterations, tool_calls) = challenge_agentic(state, ctx, &v10_results, max_findings).await;

    info!(
        "v20: agentic challenge {} -> {} ({} dropped, {} iterations, {} tool calls)",
        v10_results.len(), challenged.len(), v10_results.len() - challenged.len(), iterations, tool_calls
    );

    (challenged, iterations, tool_calls)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_issues_string_array() {
        let input = r#"{"issues": ["bug 1", "bug 2"]}"#;
        let findings = parse_issues(input);
        assert_eq!(findings.len(), 2);
        assert_eq!(findings[0].issue, "bug 1");
    }

    #[test]
    fn test_parse_issues_object_array() {
        let input = r#"{"issues": [{"issue": "null check missing", "evidence": "if (x)"}]}"#;
        let findings = parse_issues(input);
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].issue, "null check missing");
        assert_eq!(findings[0].evidence.as_deref(), Some("if (x)"));
    }

    #[test]
    fn test_parse_issues_with_code_fence() {
        let input = "```json\n{\"issues\": [\"bug\"]}\n```";
        let findings = parse_issues(input);
        assert_eq!(findings.len(), 1);
    }

    #[test]
    fn test_strip_code_fences() {
        assert_eq!(strip_code_fences("```json\n{}\n```"), "{}");
        assert_eq!(strip_code_fences("```\n{}\n```"), "{}");
        assert_eq!(strip_code_fences("{}"), "{}");
    }
}
