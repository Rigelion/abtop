use super::{context_window_for_model, process};
use crate::model::{
    AgentSession, ChildProcess, ChatMessage, ChatRole, FileAccess, FileOp, LaunchSurface,
    SessionStatus, ToolCall, MAX_CHAT_MESSAGES, MAX_FILE_ACCESSES,
};
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::{BufRead, Read, Seek, SeekFrom};
#[cfg(unix)]
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// Maximum transcript line size accepted by the parser.
const MAX_LINE_BYTES: usize = 10 * 1024 * 1024;

/// Maximum absolute difference between a pi process start time and a session
/// file's creation timestamp for PID → session matching.
const MATCH_WINDOW_MS: u64 = 15 * 60 * 1000;

/// pi session header (first line of every session JSONL):
/// `{"type":"session","version":3,"id":"...","timestamp":"...","cwd":"..."}`
#[derive(Debug, Clone)]
struct PiSessionHeader {
    id: String,
    cwd: String,
    started_at: u64,
}

/// Result of parsing a pi transcript file from an offset.
#[derive(Debug, Clone)]
struct PiTranscriptResult {
    /// Model display string ("provider/modelId" when provider is known).
    model: String,
    model_id: String,
    total_input: u64,
    total_output: u64,
    total_cache_read: u64,
    total_cache_create: u64,
    /// Last assistant turn's input context size (input + cacheRead).
    last_context_tokens: u64,
    /// High-water mark: largest context seen in any turn.
    max_context_tokens: u64,
    /// cacheRead from the previous turn (for compaction detection).
    prev_cache_read: u64,
    context_history: Vec<u64>,
    compaction_count: u32,
    turn_count: u32,
    /// "name arg" of the latest toolCall in the latest assistant turn,
    /// cleared when a toolResult closes the turn.
    current_task: String,
    last_activity: SystemTime,
    new_offset: u64,
    /// File identity: (inode, mtime_ns). Used to detect file replacement.
    file_identity: (u64, u64),
    token_history: Vec<u64>,
    initial_prompt: String,
    first_assistant_text: String,
    chat_messages: Vec<ChatMessage>,
    tool_calls: Vec<ToolCall>,
    /// Timestamp (epoch ms) of the last assistant turn whose toolCalls are
    /// still awaiting a toolResult. Zero when closed.
    last_assistant_ts_ms: u64,
    /// Timestamp (epoch ms) of the most recent user prompt not yet followed
    /// by an assistant turn. Zero when the latest entry was assistant.
    last_user_ts_ms: u64,
    /// True when this parse observed at least one user/assistant line.
    saw_turn: bool,
    file_accesses: Vec<FileAccess>,
}

/// Collector for pi coding agent sessions.
///
/// Discovery strategy:
/// 1. `ps` (shared process data) to find running `pi` processes
/// 2. Match each PID to a session JSONL under `~/.pi/agent/sessions/<encoded-cwd>/`
///    by comparing the process start time with the session file's creation
///    timestamp (embedded in the file name), with a single-candidate
///    fallback. pi does not embed a PID anywhere in its session files, so
///    lsof/open-fd matching (the Claude/Codex approach) is not possible —
///    pi opens the file per append (`appendFileSync`) and holds no
///    persistent fd.
/// 3. Parse the JSONL transcript with offset-based incremental reads, the
///    same strategy as the Claude collector.
///
/// pi exposes no rate-limit data, so the quota panel stays Claude + Codex
/// only. Context windows come from `~/.pi/agent/models.json` (pi is
/// model-agnostic, including local providers), falling back to the generic
/// 200K/1M heuristic.
pub struct PiCollector {
    /// pi agent directory (e.g. `~/.pi/agent`, from `PI_CODING_AGENT_DIR`).
    agent_dir: PathBuf,
    /// Model id → context window from `models.json`.
    model_windows: HashMap<String, u64>,
    model_windows_loaded: bool,
    /// Incremental transcript cache, keyed by session id.
    transcript_cache: HashMap<String, PiTranscriptResult>,
}

impl PiCollector {
    pub fn new() -> Self {
        let agent_dir = std::env::var("PI_CODING_AGENT_DIR")
            .ok()
            .map(PathBuf::from)
            .filter(|p| p.is_dir())
            .unwrap_or_else(|| dirs::home_dir().unwrap_or_default().join(".pi").join("agent"));
        Self {
            agent_dir,
            model_windows: HashMap::new(),
            model_windows_loaded: false,
            transcript_cache: HashMap::new(),
        }
    }

    fn sessions_dir(&self) -> PathBuf {
        self.agent_dir.join("sessions")
    }

    fn load_model_windows(&mut self) {
        if self.model_windows_loaded {
            return;
        }
        self.model_windows_loaded = true;
        let path = self.agent_dir.join("models.json");
        let Ok(content) = fs::read_to_string(&path) else {
            return;
        };
        let Ok(root) = serde_json::from_str::<Value>(&content) else {
            return;
        };
        if let Some(providers) = root.get("providers").and_then(|p| p.as_object()) {
            for provider in providers.values() {
                if let Some(models) = provider.get("models").and_then(|m| m.as_array()) {
                    for model in models {
                        let Some(id) = model.get("id").and_then(|v| v.as_str()) else {
                            continue;
                        };
                        let Some(window) = model
                            .get("contextWindow")
                            .and_then(|v| v.as_u64())
                        else {
                            continue;
                        };
                        if window > 0 {
                            self.model_windows.insert(id.to_string(), window);
                        }
                    }
                }
            }
        }
    }

    fn collect_sessions(&mut self, shared: &super::SharedProcessData) -> Vec<AgentSession> {
        let self_pid = std::process::id();
        let pids = Self::find_pi_pids(&shared.process_info, self_pid);
        if pids.is_empty() {
            // No live pi processes: drop all cached state so counters from
            // finished sessions don't survive into future runs.
            self.transcript_cache.clear();
            return Vec::new();
        }

        // models.json changes rarely; reload only on slow ticks.
        if shared.slow_tick || !self.model_windows_loaded {
            self.load_model_windows();
        }

        let sessions_dir = self.sessions_dir();
        let mut claimed: HashSet<String> = HashSet::new();
        let mut sessions = Vec::new();

        for &pid in &pids {
            let Some(cwd) = get_process_cwd(pid) else {
                continue;
            };
            let process_start = process_start_ms(pid);
            let Some(session_file) = match_session_file(
                &sessions_dir,
                &cwd,
                process_start,
                &mut claimed,
            )
            else {
                continue;
            };

            if let Some(session) = self.load_session(
                pid,
                &session_file,
                &cwd,
                &shared.process_info,
                &shared.children_map,
                &shared.ports,
            ) {
                sessions.push(session);
            }
        }

        // Evict cache entries for session ids that are no longer live.
        let active_ids: HashSet<&str> = sessions.iter().map(|s| s.session_id.as_str()).collect();
        self.transcript_cache
            .retain(|sid, _| active_ids.contains(sid.as_str()));

        sessions.sort_by_key(|s| std::cmp::Reverse(s.started_at));
        sessions
    }

    fn load_session(
        &mut self,
        pid: u32,
        session_file: &Path,
        proc_cwd: &str,
        process_info: &HashMap<u32, process::ProcInfo>,
        children_map: &HashMap<u32, Vec<u32>>,
        ports: &HashMap<u32, Vec<u16>>,
    ) -> Option<AgentSession> {
        let header = read_session_header(session_file)?;

        // The header's cwd is the source of truth (matches Claude's
        // sessions/{PID}.json convention); process cwd is the fallback.
        let cwd = if header.cwd.is_empty() {
            proc_cwd.to_string()
        } else {
            header.cwd.clone()
        };

        let proc = process_info.get(&pid);
        let mem_mb = proc.map(|p| p.rss_kb / 1024).unwrap_or(0);

        // Incremental transcript parse with the same cache-merge strategy
        // as the Claude collector.
        let cached = self.transcript_cache.remove(&header.id);
        let identity_changed = cached
            .as_ref()
            .map(|c| c.file_identity != file_identity(session_file))
            .unwrap_or(false);
        let from_offset = if identity_changed {
            0
        } else {
            cached.as_ref().map(|c| c.new_offset).unwrap_or(0)
        };
        let (initial_context_tokens, initial_cache_read) = if from_offset > 0 {
            cached
                .as_ref()
                .map(|c| (c.last_context_tokens, c.prev_cache_read))
                .unwrap_or((0, 0))
        } else {
            (0, 0)
        };
        let delta = parse_pi_transcript(
            session_file,
            from_offset,
            initial_context_tokens,
            initial_cache_read,
        );

        let cached_result = if let Some(mut prev) = cached {
            if identity_changed || from_offset == 0 || delta.new_offset < from_offset {
                // File replaced, shrank, or first parse — replace entirely.
                prev = delta;
            } else {
                // Merge delta into cached result.
                if !delta.model.is_empty() {
                    prev.model = delta.model.clone();
                }
                if !delta.model_id.is_empty() {
                    prev.model_id = delta.model_id.clone();
                }
                prev.total_input += delta.total_input;
                prev.total_output += delta.total_output;
                prev.total_cache_read += delta.total_cache_read;
                prev.total_cache_create += delta.total_cache_create;
                if delta.last_context_tokens > 0 {
                    prev.last_context_tokens = delta.last_context_tokens;
                }
                if delta.max_context_tokens > prev.max_context_tokens {
                    prev.max_context_tokens = delta.max_context_tokens;
                }
                if delta.last_context_tokens > 0 {
                    prev.prev_cache_read = delta.prev_cache_read;
                }
                prev.compaction_count += delta.compaction_count;
                prev.turn_count += delta.turn_count;
                // Always update current_task from delta — empty means the
                // latest toolResult closed the pending tool.
                if delta.saw_turn {
                    prev.current_task = delta.current_task;
                }
                if delta.last_activity > prev.last_activity {
                    prev.last_activity = delta.last_activity;
                }
                prev.token_history.extend(delta.token_history);
                if prev.tool_calls.len() < 500 {
                    let remaining = 500 - prev.tool_calls.len();
                    prev.tool_calls.extend(delta.tool_calls.into_iter().take(remaining));
                }
                prev.chat_messages.extend(delta.chat_messages);
                let len = prev.chat_messages.len();
                if len > MAX_CHAT_MESSAGES {
                    prev.chat_messages.drain(..len - MAX_CHAT_MESSAGES);
                }
                if delta.saw_turn {
                    prev.last_assistant_ts_ms = delta.last_assistant_ts_ms;
                    prev.last_user_ts_ms = delta.last_user_ts_ms;
                }
                if prev.initial_prompt.is_empty() && !delta.initial_prompt.is_empty() {
                    prev.initial_prompt = delta.initial_prompt;
                }
                if prev.first_assistant_text.is_empty() && !delta.first_assistant_text.is_empty() {
                    prev.first_assistant_text = delta.first_assistant_text;
                }
                prev.file_accesses.extend(delta.file_accesses);
                let len = prev.file_accesses.len();
                if len > MAX_FILE_ACCESSES {
                    prev.file_accesses.drain(..len - MAX_FILE_ACCESSES);
                }
                prev.new_offset = delta.new_offset;
            }
            prev
        } else {
            delta
        };
        self.transcript_cache.insert(header.id.clone(), cached_result.clone());

        let model_window = self
            .model_windows
            .get(&cached_result.model_id)
            .copied()
            .unwrap_or_else(|| {
                context_window_for_model(&cached_result.model, "", cached_result.max_context_tokens)
            });
        let context_percent = if model_window > 0 {
            (cached_result.last_context_tokens as f64 / model_window as f64) * 100.0
        } else {
            0.0
        };

        // Status is best-effort, same signals as the Claude collector:
        // 1. Active descendant CPU → tool running.
        // 2. Pending tool (toolCall without toolResult yet) → executing.
        // 3. Trailing user prompt with no assistant reply → thinking.
        let has_active_descendant =
            process::has_active_descendant(pid, children_map, process_info, 5.0);
        let pending_tool = cached_result.last_assistant_ts_ms > 0;
        let model_generating = cached_result.last_user_ts_ms > 0;
        let status = if has_active_descendant || pending_tool {
            SessionStatus::Executing
        } else if model_generating {
            SessionStatus::Thinking
        } else {
            SessionStatus::Waiting
        };

        let current_tasks = if !cached_result.current_task.is_empty() {
            vec![cached_result.current_task.clone()]
        } else if matches!(status, SessionStatus::Waiting) {
            vec!["waiting for input".to_string()]
        } else {
            vec!["thinking...".to_string()]
        };

        // All descendants (not just direct children) so port-listening
        // grandchildren are caught, same as the Claude collector.
        let mut children = Vec::new();
        let mut stack: Vec<u32> = children_map.get(&pid).cloned().unwrap_or_default();
        let mut visited = HashSet::new();
        while let Some(cpid) = stack.pop() {
            if !visited.insert(cpid) {
                continue;
            }
            if let Some(cproc) = process_info.get(&cpid) {
                let port = ports.get(&cpid).and_then(|v| v.first().copied());
                children.push(ChildProcess {
                    pid: cpid,
                    command: cproc.command.clone(),
                    mem_kb: cproc.rss_kb,
                    port,
                });
            }
            if let Some(grandchildren) = children_map.get(&cpid) {
                stack.extend(grandchildren);
            }
        }

        let project_name = process::last_path_segment(&cwd).unwrap_or("?").to_string();

        Some(AgentSession {
            agent_cli: "pi",
            launch_surface: LaunchSurface::Cli,
            pid,
            session_id: header.id,
            cwd,
            project_name,
            started_at: header.started_at,
            status,
            model: cached_result.model,
            effort: String::new(),
            context_percent,
            total_input_tokens: cached_result.total_input,
            total_output_tokens: cached_result.total_output,
            total_cache_read: cached_result.total_cache_read,
            total_cache_create: cached_result.total_cache_create,
            turn_count: cached_result.turn_count,
            current_tasks,
            mem_mb,
            version: pi_version(&self.agent_dir),
            git_branch: String::new(),
            git_added: 0,
            git_modified: 0,
            token_history: cached_result.token_history,
            context_history: cached_result.context_history,
            compaction_count: cached_result.compaction_count,
            context_window: model_window,
            subagents: Vec::new(),
            mem_file_count: 0,
            mem_line_count: 0,
            children,
            initial_prompt: cached_result.initial_prompt,
            first_assistant_text: cached_result.first_assistant_text,
            chat_messages: cached_result.chat_messages,
            tool_calls: cached_result.tool_calls,
            pending_since_ms: cached_result.last_assistant_ts_ms,
            thinking_since_ms: cached_result.last_user_ts_ms,
            file_accesses: cached_result.file_accesses,
            config_root: super::abbrev_path(&self.agent_dir),
        })
    }

    /// Find live `pi` processes. The pi TUI runs as a process literally
    /// named `pi` (the launcher execs the release binary, so `ps` shows
    /// `pi`, not `node .../pi`).
    fn find_pi_pids(
        process_info: &HashMap<u32, process::ProcInfo>,
        self_pid: u32,
    ) -> Vec<u32> {
        let mut pids = Vec::new();
        for (pid, info) in process_info {
            if !process::cmd_has_binary(&info.command, "pi") {
                continue;
            }
            // Exclude abtop's own children (defensive: abtop does not spawn
            // pi processes today, but this keeps the collector consistent
            // with the Claude collector's self-filtering.
            if process::is_descendant_of(*pid, self_pid, process_info) {
                continue;
            }
            pids.push(*pid);
        }
        pids
    }
}

impl Default for PiCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl super::AgentCollector for PiCollector {
    fn collect(&mut self, shared: &super::SharedProcessData) -> Vec<AgentSession> {
        self.collect_sessions(shared)
    }
}

/// Read and parse the first line (session header) of a pi session file.
fn read_session_header(path: &Path) -> Option<PiSessionHeader> {
    let mut file = fs::File::open(path).ok()?;
    let mut buf = Vec::new();
    file.read_to_end(&mut buf).ok()?;
    let line = buf.split(|&b| b == b'\n').next()?.to_vec();
    let val = serde_json::from_slice::<Value>(&line).ok()?;
    if val.get("type").and_then(|t| t.as_str()) != Some("session") {
        return None;
    }
    let id = val
        .get("id")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let cwd = val
        .get("cwd")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let started_at = val
        .get("timestamp")
        .and_then(|t| t.as_str())
        .and_then(|ts| chrono::DateTime::parse_from_rfc3339(ts).ok())
        .map(|dt| dt.timestamp_millis().max(0) as u64)
        .unwrap_or(0);
    Some(PiSessionHeader { id, cwd, started_at })
}

/// Encode a cwd the way pi names session subdirectories:
/// strip one leading `/` or `\`, replace every `/`, `\`, `:` with `-`,
/// and wrap in `--...--`. Matches `getDefaultSessionDirPath` in
/// pi's `session-manager.js`.
fn encode_cwd(cwd: &str) -> String {
    let stripped = cwd.strip_prefix('/').or_else(|| cwd.strip_prefix('\\'));
    let stripped = stripped.unwrap_or(cwd);
    let encoded: String = stripped
        .chars()
        .map(|c| match c {
            '/' | '\\' | ':' => '-',
            c => c,
        })
        .collect();
    format!("--{encoded}--")
}

/// Parse a session file name's creation timestamp.
/// File names look like `2026-11-11T18-59-38-725Z_<uuid>.jsonl`.
fn filename_ts_ms(stem: &str) -> Option<u64> {
    let bytes = stem.as_bytes();
    if bytes.len() < 24 || bytes[10] != b'T' || bytes[23] != b'Z' {
        return None;
    }
    let num = |a: &[u8]| -> Option<u64> {
        a.iter().try_fold(0u64, |acc, &b| b.checked_sub(b'0').map(|d| acc * 10 + d as u64))
    };
    let year = num(&bytes[0..4])?;
    let month = num(&bytes[5..7])?;
    let day = num(&bytes[8..10])?;
    let hour = num(&bytes[11..13])?;
    let minute = num(&bytes[14..16])?;
    let second = num(&bytes[17..19])?;
    let millis = num(&bytes[20..23])?;
    // Days since epoch via chrono (handles leap years correctly).
    let date = chrono::NaiveDate::from_ymd_opt(year as i32, month as u32, day as u32)?;
    let days = (date - chrono::NaiveDate::from_ymd_opt(1970, 1, 1).unwrap())
        .num_days() as u64;
    Some(
        days * 86_400_000
            + (hour * 3600 + minute * 60 + second) * 1000
            + millis,
    )
}

/// Match a live pi PID to its session file.
///
/// Primary: the candidate whose creation timestamp is closest to the
/// process start time, within `MATCH_WINDOW_MS`. This covers normal
/// sessions (pi is spawned per session) as well as `/clear` forks whose
/// new file is minted minutes after the process started.
///
/// Fallback: when exactly one unclaimed candidate exists in the encoded
/// cwd directory, take it (the common single-session-per-dir case).
fn match_session_file(
    sessions_dir: &Path,
    cwd: &str,
    process_start: Option<u64>,
    claimed: &mut HashSet<String>,
) -> Option<PathBuf> {
    let dir = sessions_dir.join(encode_cwd(cwd));
    let entries = fs::read_dir(&dir).ok()?;
    let mut candidates: Vec<(PathBuf, u64)> = entries
        .flatten()
        .filter_map(|entry| {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("jsonl") {
                return None;
            }
            if is_symlink(&path) {
                return None;
            }
            // Prefer the file name's creation timestamp; fall back to mtime.
            let ts = path
                .file_stem()
                .and_then(|s| s.to_str())
                .and_then(filename_ts_ms)
                .or_else(|| {
                    entry
                        .metadata()
                        .ok()
                        .and_then(|m| m.modified().ok())
                        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                        .map(|d| d.as_millis() as u64)
                })?;
            Some((path, ts))
        })
        .collect();
    candidates.retain(|(path, _)| !claimed.contains(&path.to_string_lossy().into_owned()));
    candidates.sort_by_key(|a| std::cmp::Reverse(a.1)); // newest first

    if let Some(start) = process_start {
        let best = candidates
            .iter()
            .filter(|(_, ts)| {
                ts.saturating_sub(start).abs_diff(start.saturating_sub(*ts)) <= MATCH_WINDOW_MS
            })
            .min_by_key(|(_, ts)| {
                ts.saturating_sub(start).abs_diff(start.saturating_sub(*ts))
            });
        if let Some((path, _)) = best {
            claimed.insert(path.to_string_lossy().into_owned());
            return Some(path.clone());
        }
    }

    if candidates.len() == 1 {
        let (path, _) = candidates.remove(0);
        claimed.insert(path.to_string_lossy().into_owned());
        return Some(path);
    }

    None
}

/// pi's version string from the managed install (`install/current-version`).
fn pi_version(agent_dir: &Path) -> String {
    fs::read_to_string(agent_dir.join("install").join("current-version"))
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_default()
}

/// Check if a path is a symlink without following it (fail-closed).
fn is_symlink(path: &Path) -> bool {
    fs::symlink_metadata(path)
        .map(|m| m.file_type().is_symlink())
        .unwrap_or(true)
}

/// Get file identity as (device, inode) for detecting file replacement.
#[cfg(unix)]
fn file_identity(path: &Path) -> (u64, u64) {
    fs::metadata(path)
        .ok()
        .map(|m| (m.dev(), m.ino()))
        .unwrap_or((0, 0))
}

#[cfg(windows)]
fn file_identity(path: &Path) -> (u64, u64) {
    fs::metadata(path)
        .ok()
        .map(|m| {
            let size = m.len();
            let mtime_ns = m
                .modified()
                .ok()
                .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0);
            (size, mtime_ns)
        })
        .unwrap_or((0, 0))
}

/// The process's current working directory.
#[cfg(target_os = "linux")]
fn get_process_cwd(pid: u32) -> Option<String> {
    fs::read_link(format!("/proc/{pid}/cwd"))
        .ok()
        .map(|p| p.to_string_lossy().into_owned())
}

#[cfg(target_os = "windows")]
fn get_process_cwd(pid: u32) -> Option<String> {
    use sysinfo::{Pid, ProcessRefreshKind, ProcessesToUpdate, System, UpdateKind};
    let mut sys = System::new();
    let pid = Pid::from_u32(pid);
    sys.refresh_processes_specifics(
        ProcessesToUpdate::Some(&[pid]),
        false,
        ProcessRefreshKind::new().with_cwd(UpdateKind::Always),
    );
    sys.process(pid)
        .and_then(|p| p.cwd())
        .map(|p| p.to_string_lossy().into_owned())
}

#[cfg(all(not(target_os = "linux"), not(target_os = "windows")))]
fn get_process_cwd(pid: u32) -> Option<String> {
    // -a ANDs the selection terms; without it, lsof ORs `-p <pid>` with
    // `-d cwd` and returns cwd entries for unrelated processes too.
    let output = std::process::Command::new("lsof")
        .args(["-a", "-p", &pid.to_string(), "-d", "cwd", "-Fn"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    stdout
        .lines()
        .find(|l| l.starts_with('n') && l.len() > 1)
        .map(|l| l[1..].to_string())
}

/// When the process started (epoch ms).
#[cfg(target_os = "linux")]
fn process_start_ms(pid: u32) -> Option<u64> {
    // The /proc/{pid} inode's mtime is the process creation time.
    fs::metadata(format!("/proc/{pid}"))
        .ok()?
        .modified()
        .ok()?
        .duration_since(UNIX_EPOCH)
        .ok()
        .map(|d| d.as_millis() as u64)
}

/// Parse a pi transcript from `from_offset`, carrying the previous turn's
/// context size (for compaction detection) the same way
/// `claude::parse_transcript_with_previous` does.
fn parse_pi_transcript(
    path: &Path,
    from_offset: u64,
    initial_context_tokens: u64,
    initial_cache_read: u64,
) -> PiTranscriptResult {
    let identity = file_identity(path);
    let mut result = PiTranscriptResult {
        model: String::new(),
        model_id: String::new(),
        total_input: 0,
        total_output: 0,
        total_cache_read: 0,
        total_cache_create: 0,
        last_context_tokens: 0,
        max_context_tokens: 0,
        prev_cache_read: 0,
        context_history: Vec::new(),
        compaction_count: 0,
        turn_count: 0,
        current_task: String::new(),
        last_activity: UNIX_EPOCH,
        new_offset: from_offset,
        file_identity: identity,
        token_history: Vec::new(),
        initial_prompt: String::new(),
        first_assistant_text: String::new(),
        chat_messages: Vec::new(),
        tool_calls: Vec::new(),
        last_assistant_ts_ms: 0,
        last_user_ts_ms: 0,
        saw_turn: false,
        file_accesses: Vec::new(),
    };

    let file = match fs::File::open(path) {
        Ok(f) => f,
        Err(_) => return result,
    };

    let file_len = file.metadata().map(|m| m.len()).unwrap_or(0);
    if file_len == from_offset {
        // No new data.
        result.new_offset = file_len;
        return result;
    }
    // File shrank (truncation/rotation) — reset and reparse from start.
    let from_offset = if file_len < from_offset { 0 } else { from_offset };
    // Seed the previous-turn state for compaction detection when resuming
    // from a cache offset.
    if from_offset > 0 {
        result.prev_cache_read = initial_cache_read;
        if initial_context_tokens > 0 {
            result.context_history.push(initial_context_tokens);
        }
    }

    let mut reader = std::io::BufReader::new(file);
    if from_offset > 0 {
        let _ = reader.seek(SeekFrom::Start(from_offset));
    }

    result.last_activity = fs::metadata(path)
        .ok()
        .and_then(|m| m.modified().ok())
        .unwrap_or(UNIX_EPOCH);

    let mut bytes_read = from_offset;
    let mut line_buf = String::new();
    loop {
        line_buf.clear();
        // Bounded read so a malformed or hostile transcript with an
        // unbounded line cannot OOM the process.
        match reader
            .by_ref()
            .take(MAX_LINE_BYTES as u64 + 1)
            .read_line(&mut line_buf)
        {
            Ok(0) => break,
            Ok(n) => {
                if line_buf.len() > MAX_LINE_BYTES && !line_buf.ends_with('\n') {
                    bytes_read = file_len;
                    break;
                }
                let has_newline = line_buf.ends_with('\n');
                let line = line_buf.trim();
                if line.is_empty() {
                    if has_newline {
                        bytes_read += n as u64;
                    }
                    continue;
                }
                let val = match serde_json::from_str::<Value>(line) {
                    Ok(v) => v,
                    Err(_) => {
                        if has_newline {
                            // Complete line but invalid JSON — skip it.
                            bytes_read += n as u64;
                        }
                        // Incomplete line with parse error — defer to next poll.
                        if !has_newline {
                            break;
                        }
                        continue;
                    }
                };
                bytes_read += n as u64;
                handle_pi_line(&mut result, &val);
            }
            Err(_) => break,
        }
    }

    let len = result.file_accesses.len();
    if len > MAX_FILE_ACCESSES {
        result.file_accesses.drain(..len - MAX_FILE_ACCESSES);
    }

    result.new_offset = bytes_read;
    result
}

/// Process one parsed JSONL line of a pi transcript.
fn handle_pi_line(result: &mut PiTranscriptResult, val: &Value) {
    let entry_ts_ms = val
        .get("timestamp")
        .and_then(|t| t.as_str())
        .and_then(|ts| chrono::DateTime::parse_from_rfc3339(ts).ok())
        .map(|dt| dt.timestamp_millis().max(0) as u64)
        .unwrap_or(0);

    match val.get("type").and_then(|t| t.as_str()) {
        Some("model_change") => {
            let Some(model_id) = val.get("modelId").and_then(|v| v.as_str()) else {
                return;
            };
            result.model_id = model_id.to_string();
            result.model = model_display(val, model_id);
        }
        Some("message") => {
            let Some(message) = val.get("message") else {
                return;
            };
            let role = message.get("role").and_then(|r| r.as_str()).unwrap_or("");
            match role {
                "assistant" => handle_pi_assistant(result, message, entry_ts_ms),
                "user" => handle_pi_user(result, message, entry_ts_ms),
                "toolResult" => handle_pi_tool_result(result, entry_ts_ms),
                // "system" and anything unknown: ignore.
                _ => {}
            }
        }
        _ => {}
    }
}

/// Build the display string "provider/modelId" (or just the model id
/// when no provider is known). `obj` is either the `message` object or
/// the top-level `model_change` entry — both carry `provider`.
fn model_display(obj: &Value, model_id: &str) -> String {
    let provider = obj.get("provider").and_then(|p| p.as_str()).unwrap_or("");
    if !provider.is_empty() {
        format!("{provider}/{model_id}")
    } else {
        model_id.to_string()
    }
}

fn handle_pi_assistant(
    result: &mut PiTranscriptResult,
    message: &Value,
    entry_ts_ms: u64,
) {
    result.turn_count += 1;
    // Clear previous task on each new turn so stale tasks don't persist.
    result.current_task = String::new();

    // pi stores model/provider/usage inside the `message` object.
    let model_id = message
        .get("model")
        .and_then(|m| m.as_str())
        .unwrap_or("");
    if !model_id.is_empty() {
        result.model_id = model_id.to_string();
        result.model = model_display(message, model_id);
    }

    if let Some(usage) = message.get("usage") {
        let inp = usage.get("input").and_then(|v| v.as_u64()).unwrap_or(0);
        let out = usage.get("output").and_then(|v| v.as_u64()).unwrap_or(0);
        let cr = usage.get("cacheRead").and_then(|v| v.as_u64()).unwrap_or(0);
        let cc = usage.get("cacheWrite").and_then(|v| v.as_u64()).unwrap_or(0);
        result.total_input += inp;
        result.total_output += out;
        result.total_cache_read += cr;
        result.total_cache_create += cc;
        // Context = input + cacheRead (excludes cacheWrite; same rule as
        // the Claude collector, see #54). Exception: a fresh session
        // creating cache for the first time reports the whole context as
        // cacheWrite, so count it there.
        let current_context = if cr == 0 && cc > 0 { inp + cc } else { inp + cr };
        result.last_context_tokens = current_context;
        if current_context > result.max_context_tokens {
            result.max_context_tokens = current_context;
        }
        // Compaction: context dropped > 30% between turns AND cacheRead
        // collapsed (old cache invalidated).
        if context_dropped(result, current_context, cr) {
            result.compaction_count += 1;
        }
        result.prev_cache_read = cr;
        if result.context_history.len() < 10_000 {
            result.context_history.push(current_context);
        }
        if result.token_history.len() < 10_000 {
            result.token_history.push(inp + out + cr + cc);
        }
    }

    let mut has_tool_call = false;
    if let Some(content) = message.get("content").and_then(|c| c.as_array()) {
        for block in content {
            match block.get("type").and_then(|t| t.as_str()) {
                Some("text") => {
                    let Some(text) = block.get("text").and_then(|t| t.as_str()) else {
                        continue;
                    };
                    if result.first_assistant_text.is_empty() {
                        result.first_assistant_text = truncate(&normalize_text(text), 200);
                    }
                    let chat = truncate(&normalize_text(text), 500);
                    if !chat.is_empty() {
                        push_chat_message(
                            &mut result.chat_messages,
                            ChatRole::Assistant,
                            chat,
                        );
                    }
                }
                Some("toolCall") => {
                    has_tool_call = true;
                    let name = block
                        .get("name")
                        .and_then(|n| n.as_str())
                        .unwrap_or("?");
                    let arg = extract_pi_tool_arg(block);
                    // Last toolCall in forward order wins current_task.
                    result.current_task = format!("{name} {arg}");
                    if result.tool_calls.len() < 500 {
                        result.tool_calls.push(ToolCall {
                            name: name.to_string(),
                            arg: truncate(&arg, 40),
                            duration_ms: 0, // filled when the toolResult arrives
                        });
                    }
                    // File access audit: read/edit/write carry a `path`.
                    if let Some(file_path) = block
                        .get("arguments")
                        .and_then(|a| a.get("path"))
                        .and_then(|f| f.as_str())
                    {
                        let op = match name {
                            "read" => Some(FileOp::Read),
                            "edit" => Some(FileOp::Edit),
                            "write" => Some(FileOp::Write),
                            _ => None,
                        };
                        if let Some(op) = op {
                            result.file_accesses.push(FileAccess {
                                path: file_path.to_string(),
                                operation: op,
                                turn_index: result.turn_count,
                            });
                        }
                    }
                }
                // "thinking" blocks are internal model output; skip.
                _ => {}
            }
        }
    }

    // Save timestamp only when toolCalls are present, so "tools pending"
    // can be detected; a text-only assistant turn closes the window.
    if entry_ts_ms > 0 && has_tool_call {
        result.last_assistant_ts_ms = entry_ts_ms;
    } else {
        result.last_assistant_ts_ms = 0;
    }
    // Any assistant turn closes the prior "thinking" window.
    result.last_user_ts_ms = 0;
    result.saw_turn = true;
}

fn handle_pi_user(result: &mut PiTranscriptResult, message: &Value, entry_ts_ms: u64) {
    let text = extract_pi_text(message);
    if text.is_empty() {
        return; // empty prompt — not a thinking window
    }
    if entry_ts_ms > 0 {
        result.last_user_ts_ms = entry_ts_ms;
    }
    result.saw_turn = true;
    if result.initial_prompt.is_empty() {
        result.initial_prompt = truncate(&text, 160);
    }
    let chat = truncate(&text, 500);
    if !chat.is_empty() {
        push_chat_message(&mut result.chat_messages, ChatRole::User, chat);
    }
}

/// A `toolResult` entry closes the pending tool round: it stamps tool
/// durations, clears `last_assistant_ts_ms`, and never opens a thinking
/// window (it is not a user prompt).
fn handle_pi_tool_result(result: &mut PiTranscriptResult, entry_ts_ms: u64) {
    if entry_ts_ms > 0 && result.last_assistant_ts_ms > 0 {
        let duration = entry_ts_ms.saturating_sub(result.last_assistant_ts_ms);
        let pending: Vec<usize> = result
            .tool_calls
            .iter()
            .enumerate()
            .rev()
            .take_while(|(_, tc)| tc.duration_ms == 0)
            .map(|(i, _)| i)
            .collect();
        if !pending.is_empty() {
            let per_call = duration / pending.len() as u64;
            for idx in pending {
                result.tool_calls[idx].duration_ms = per_call;
            }
        }
        result.last_assistant_ts_ms = 0;
    }
}

/// True when context dropped > 30% between consecutive assistant turns
/// AND cacheRead collapsed (old cache invalidated). `prev_cache_read`
/// must hold the previous turn's value at call time.
fn context_dropped(result: &PiTranscriptResult, current: u64, cr: u64) -> bool {
    let prev_context = result.context_history.last().copied().unwrap_or(0);
    prev_context > 0
        && current < prev_context * 7 / 10
        && result.prev_cache_read > 1000
        && cr < result.prev_cache_read / 5
}

/// Extract the short first-argument for display (file path, command,
/// pattern, ...). Tries a fixed priority of known pi tool argument keys.
fn extract_pi_tool_arg(block: &Value) -> String {
    let Some(args) = block.get("arguments").and_then(|a| a.as_object()) else {
        return String::new();
    };
    for key in ["path", "file_path", "command", "pattern", "url", "query", "prompt"] {
        if let Some(v) = args.get(key) {
            let s = match v {
                Value::String(s) => s.clone(),
                other => other.to_string(),
            };
            if !s.trim().is_empty() {
                return s;
            }
        }
    }
    String::new()
}

/// Collect text from a pi message content field (string or text blocks).
fn extract_pi_text(message: &Value) -> String {
    match message.get("content") {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(arr)) => arr
            .iter()
            .filter_map(|block| {
                if block.get("type").and_then(|t| t.as_str()) == Some("text") {
                    block.get("text").and_then(|t| t.as_str()).map(|s| s.to_string())
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join(" "),
        _ => String::new(),
    }
}

fn normalize_text(raw: &str) -> String {
    let cleaned = super::sanitize_terminal_text(raw);
    cleaned
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

fn truncate(s: &str, max_chars: usize) -> String {
    let mut out: String = s.chars().take(max_chars).collect();
    if s.chars().count() > max_chars {
        out.push('…');
    }
    out
}

fn push_chat_message(messages: &mut Vec<ChatMessage>, role: ChatRole, text: String) {
    if text.is_empty() {
        return;
    }
    messages.push(ChatMessage { role, text });
    let len = messages.len();
    if len > MAX_CHAT_MESSAGES {
        messages.drain(..len - MAX_CHAT_MESSAGES);
    }
}

#[cfg(target_os = "windows")]
fn process_start_ms(pid: u32) -> Option<u64> {
    use sysinfo::{Pid, ProcessRefreshKind, ProcessesToUpdate, System};
    let mut sys = System::new();
    let pid = Pid::from_u32(pid);
    sys.refresh_processes_specifics(
        ProcessesToUpdate::Some(&[pid]),
        true,
        ProcessRefreshKind::default(),
    );
    sys.process(pid)
        .and_then(|p| p.start_time())
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_millis() as u64)
}

#[cfg(all(not(target_os = "linux"), not(target_os = "windows")))]
fn process_start_ms(pid: u32) -> Option<u64> {
    let output = std::process::Command::new("ps")
        .args(["-o", "lstart=", "-p", &pid.to_string()])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&output.stdout).trim();
    // Example: "Wed Nov 11 19:59:38 2026" (local time)
    let naive = chrono::NaiveDateTime::parse_from_str(s, "%a %b %d %H:%M:%S %Y").ok()?;
    Some(naive.and_utc().timestamp_millis().max(0) as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_lines(file: &std::path::Path, lines: &[&str]) {
        let mut content = String::new();
        for line in lines {
            content.push_str(line);
            content.push('\n');
        }
        fs::write(file, content).unwrap();
    }

    #[test]
    fn test_encode_cwd() {
        assert_eq!(
            encode_cwd("/home/rigel/src/bench"),
            "--home-rigel-src-bench--"
        );
        assert_eq!(encode_cwd("/polybar-scripts"), "--polybar-scripts--");
        assert_eq!(encode_cwd("/home/rigel/src/abtop"), "--home-rigel-src-abtop--");
        // Windows-style drive path: both the colon and the backslashes
        // become dashes (pi does not special-case drive letters).
        assert_eq!(encode_cwd("C:\\foo\\bar"), "--C--foo-bar--");
    }

    #[test]
    fn test_filename_ts_ms() {
        let expected = chrono::DateTime::parse_from_rfc3339("2026-11-11T18:59:38.725Z")
            .unwrap()
            .timestamp_millis() as u64;
        assert_eq!(
            filename_ts_ms("2026-11-11T18-59-38-725Z_01a1c5fa-e5e1-72f3-952b-4837640dbf7f"),
            Some(expected)
        );
        assert_eq!(filename_ts_ms("not-a-timestamp_abc"), None);
    }

    #[test]
    fn test_parse_pi_transcript_basic() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        write_lines(
            tmp.path(),
            &[
                r#"{"type":"session","version":3,"id":"sid-1","timestamp":"2026-11-11T18:59:38.725Z","cwd":"/home/rigel/src/bench"}"#,
                r#"{"type":"model_change","id":"mc1","parentId":null,"timestamp":"2026-11-11T18:59:38.730Z","provider":"llamacpp-localV2","modelId":"swift"}"#,
                r#"{"type":"message","id":"u1","parentId":"mc1","timestamp":"2026-11-11T18:59:40.000Z","message":{"role":"user","content":[{"type":"text","text":"fix the build"}]}}"#,
                r#"{"type":"message","id":"a1","parentId":"u1","timestamp":"2026-11-11T18:59:42.000Z","message":{"role":"assistant","content":[{"type":"toolCall","id":"t1","name":"edit","arguments":{"path":"/home/rigel/src/bench/src/main.rs"}}],"model":"swift","provider":"llamacpp-localV2","usage":{"input":100,"output":20,"cacheRead":5000,"cacheWrite":0,"reasoning":0,"totalTokens":5120}}}"#,
            ],
        );
        let r = parse_pi_transcript(tmp.path(), 0, 0, 0);
        assert_eq!(r.total_input, 100);
        assert_eq!(r.total_output, 20);
        assert_eq!(r.total_cache_read, 5000);
        assert_eq!(r.turn_count, 1);
        assert_eq!(r.model, "llamacpp-localV2/swift");
        assert_eq!(r.model_id, "swift");
        assert_eq!(r.current_task, "edit /home/rigel/src/bench/src/main.rs");
        // Trailing assistant with an open toolCall: pending, not thinking.
        assert!(r.last_assistant_ts_ms > 0);
        assert_eq!(r.last_user_ts_ms, 0);
        assert_eq!(r.initial_prompt, "fix the build");
        assert_eq!(r.context_history, vec![5100]); // 100 + 5000
        assert_eq!(r.file_accesses.len(), 1);
        assert_eq!(r.file_accesses[0].path, "/home/rigel/src/bench/src/main.rs");
        assert_eq!(r.file_accesses[0].operation, FileOp::Edit);
    }

    #[test]
    fn test_parse_pi_transcript_tool_result_closes_pending() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        write_lines(
            tmp.path(),
            &[
                r#"{"type":"session","version":3,"id":"sid-2","timestamp":"2026-11-11T19:00:00.000Z","cwd":"/p"}"#,
                r#"{"type":"message","id":"u1","timestamp":"2026-11-11T19:00:01.000Z","message":{"role":"user","content":"hello"}}"#,
                r#"{"type":"message","id":"a1","timestamp":"2026-11-11T19:00:02.000Z","message":{"role":"assistant","content":[{"type":"toolCall","id":"t1","name":"bash","arguments":{"command":"cargo build"}}],"usage":{"input":10,"output":5,"cacheRead":0,"cacheWrite":0,"totalTokens":15}}}"#,
                r#"{"type":"message","id":"tr1","timestamp":"2026-11-11T19:00:07.000Z","message":{"role":"toolResult","toolCallId":"t1","toolName":"bash","content":[{"type":"text","text":"ok"}]}}"#,
            ],
        );
        let r = parse_pi_transcript(tmp.path(), 0, 0, 0);
        // toolResult closed the pending tool window.
        assert_eq!(r.last_assistant_ts_ms, 0);
        assert_eq!(r.last_user_ts_ms, 0);
        // Duration stamped on the tool call: 5s.
        assert_eq!(r.tool_calls[0].name, "bash");
        assert_eq!(r.tool_calls[0].duration_ms, 5000);
        // Only the user prompt is in the chat tail (the assistant turn was
        // tool-only; the toolResult text must not appear).
        assert_eq!(r.chat_messages.len(), 1);
        assert_eq!(r.chat_messages[0].text, "hello");
    }

    #[test]
    fn test_parse_pi_transcript_trailing_user_marks_thinking() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        write_lines(
            tmp.path(),
            &[
                r#"{"type":"session","version":3,"id":"sid-3","timestamp":"2026-11-11T19:01:00.000Z","cwd":"/p"}"#,
                r#"{"type":"message","id":"u1","timestamp":"2026-11-11T19:01:01.000Z","message":{"role":"user","content":"now summarize"}}"#,
            ],
        );
        let r = parse_pi_transcript(tmp.path(), 0, 0, 0);
        assert!(r.last_user_ts_ms > 0);
        assert_eq!(r.last_assistant_ts_ms, 0);
    }

    #[test]
    fn test_parse_pi_transcript_incremental_offset() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        write_lines(
            tmp.path(),
            &[
                r#"{"type":"session","version":3,"id":"sid-4","timestamp":"2026-11-11T19:02:00.000Z","cwd":"/p"}"#,
                r#"{"type":"message","id":"u1","timestamp":"2026-11-11T19:02:01.000Z","message":{"role":"user","content":"one"}}"#,
                r#"{"type":"message","id":"a1","timestamp":"2026-11-11T19:02:02.000Z","message":{"role":"assistant","content":[{"type":"text","text":"done"}],"usage":{"input":100,"output":10,"cacheRead":0,"cacheWrite":0,"totalTokens":110}}}"#,
            ],
        );
        let first = parse_pi_transcript(tmp.path(), 0, 0, 0);
        assert_eq!(first.new_offset, fs::metadata(tmp.path()).unwrap().len());

        // Append a second assistant turn and parse only the new bytes.
        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .open(tmp.path())
            .unwrap();
        std::io::Write::write_all(
            &mut f,
            r#"{"type":"message","id":"a2","timestamp":"2026-11-11T19:02:05.000Z","message":{"role":"assistant","content":[{"type":"text","text":"more"}],"usage":{"input":200,"output":20,"cacheRead":1000,"cacheWrite":0,"totalTokens":1220}}}"#
                .as_bytes(),
        )
        .unwrap();
        let delta = parse_pi_transcript(tmp.path(), first.new_offset, first.last_context_tokens, first.prev_cache_read);
        assert_eq!(delta.total_input, 200);
        assert_eq!(delta.total_output, 20);
        assert_eq!(delta.total_cache_read, 1000);
        assert_eq!(delta.turn_count, 1);
        assert!(delta.saw_turn);
    }

    #[test]
    fn test_parse_pi_transcript_compaction_detected() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        write_lines(
            tmp.path(),
            &[
                r#"{"type":"session","version":3,"id":"sid-5","timestamp":"2026-11-11T19:03:00.000Z","cwd":"/p"}"#,
                r#"{"type":"message","id":"u1","timestamp":"2026-11-11T19:03:01.000Z","message":{"role":"user","content":"one"}}"#,
                r#"{"type":"message","id":"a1","timestamp":"2026-11-11T19:03:02.000Z","message":{"role":"assistant","content":[{"type":"text","text":"ok"}],"usage":{"input":100000,"output":10,"cacheRead":50000,"cacheWrite":0,"totalTokens":150100}}}"#,
                r#"{"type":"message","id":"a2","timestamp":"2026-11-11T19:03:05.000Z","message":{"role":"assistant","content":[{"type":"text","text":"post-compaction"}],"usage":{"input":5000,"output":10,"cacheRead":0,"cacheWrite":0,"totalTokens":5010}}}"#,
            ],
        );
        let r = parse_pi_transcript(tmp.path(), 0, 0, 0);
        assert_eq!(r.compaction_count, 1);
    }

    #[test]
    fn test_match_session_file_primary_timestamp_match() {
        let tmp = tempfile::TempDir::new().unwrap();
        let sessions_dir = tmp.path();
        let cwd = "/home/rigel/src/bench";
        let dir = sessions_dir.join(encode_cwd(cwd));
        fs::create_dir_all(&dir).unwrap();
        // Two sessions in the same cwd; process started at file A's time.
        let file_a = dir.join("2026-11-11T18-00-00-000Z_aaa.jsonl");
        let file_b = dir.join("2026-11-11T19-00-00-000Z_bbb.jsonl");
        fs::write(&file_a, b"{}").unwrap();
        fs::write(&file_b, b"{}").unwrap();

        let start = filename_ts_ms("2026-11-11T18-00-00-000Z").unwrap();
        let mut claimed = HashSet::new();
        let matched = match_session_file(sessions_dir, cwd, Some(start), &mut claimed).unwrap();
        assert_eq!(matched, file_a);

        // A second process started at file B's time picks B, not A.
        let start_b = filename_ts_ms("2026-11-11T19-00-00-000Z").unwrap();
        let matched_b = match_session_file(sessions_dir, cwd, Some(start_b), &mut claimed).unwrap();
        assert_eq!(matched_b, file_b);

        // No candidates left for a third process.
        assert_eq!(
            match_session_file(sessions_dir, cwd, Some(start_b), &mut claimed),
            None
        );
    }

    #[test]
    fn test_match_session_file_single_candidate_fallback() {
        let tmp = tempfile::TempDir::new().unwrap();
        let sessions_dir = tmp.path();
        let cwd = "/home/rigel/src/bench";
        let dir = sessions_dir.join(encode_cwd(cwd));
        fs::create_dir_all(&dir).unwrap();
        let file_a = dir.join("2026-11-11T18-00-00-000Z_aaa.jsonl");
        fs::write(&file_a, b"{}").unwrap();

        // Process start far from the file timestamp (e.g. resumed session):
        // falls back to the single unclaimed candidate.
        let start = filename_ts_ms("2026-11-11T18-00-00-000Z").unwrap() + 10 * 3600 * 1000;
        let mut claimed = HashSet::new();
        let matched = match_session_file(sessions_dir, cwd, Some(start), &mut claimed).unwrap();
        assert_eq!(matched, file_a);
    }

    #[test]
    fn test_find_pi_pids() {
        let mut info = HashMap::new();
        info.insert(
            100,
            process::ProcInfo {
                pid: 100,
                ppid: 1,
                rss_kb: 1000,
                cpu_pct: 0.0,
                command: "pi".to_string(),
            },
        );
        info.insert(
            200,
            process::ProcInfo {
                pid: 200,
                ppid: 1,
                rss_kb: 500,
                cpu_pct: 0.0,
                command: "/usr/bin/pip install x".to_string(),
            },
        );
        info.insert(
            300,
            process::ProcInfo {
                pid: 300,
                ppid: 1,
                rss_kb: 800,
                cpu_pct: 0.0,
                command: "/home/u/.pi/agent/bin/pi-node".to_string(),
            },
        );
        let pids = PiCollector::find_pi_pids(&info, 99_999);
        assert_eq!(pids, vec![100]);
    }

    #[test]
    fn test_read_session_header() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        write_lines(
            tmp.path(),
            &[
                r#"{"type":"session","version":3,"id":"sid-9","timestamp":"2026-11-11T18:59:38.725Z","cwd":"/home/rigel/src/bench"}"#,
                r#"{"type":"message","id":"u1","timestamp":"2026-11-11T18:59:40.000Z","message":{"role":"user","content":"hi"}}"#,
            ],
        );
        let h = read_session_header(tmp.path()).unwrap();
        assert_eq!(h.id, "sid-9");
        assert_eq!(h.cwd, "/home/rigel/src/bench");
        assert!(h.started_at > 0);
    }
}
