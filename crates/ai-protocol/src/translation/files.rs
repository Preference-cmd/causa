//! File-source helpers for wire translation.
//!
//! The translation layer is transport-free: it never performs network
//! IO. Workspace-relative file references (`FileSource::Url`) are
//! resolved to inline base64 by the concrete adapter *before* the pure
//! translation functions run; this module supplies the single
//! resolution primitive so every adapter shares the same size limit and
//! path-traversal policy.

use std::path::{Component, Path};

use crate::error::ProviderAdapterError;

/// Maximum size of a workspace file referenced by a file block, in
/// bytes. Mirrors the daemon's inline-base64 limit (10MB decoded).
pub const MAX_WORKSPACE_FILE_BYTES: u64 = 10 * 1024 * 1024;

/// Read a workspace-relative file referenced by a `FileSource::Url`.
///
/// The reference must stay inside `base_dir`: absolute paths and `..`
/// components are rejected, as are `http://` / `https://` URLs (remote
/// downloads are not supported in V2). Files larger than
/// [`MAX_WORKSPACE_FILE_BYTES`] are rejected. All failures surface as
/// [`ProviderAdapterError::Configuration`] so callers can decide how to
/// project them.
pub fn read_workspace_file(base_dir: &Path, path: &str) -> Result<Vec<u8>, ProviderAdapterError> {
    if path.starts_with("http://") || path.starts_with("https://") {
        return Err(ProviderAdapterError::configuration(format!(
            "remote URLs are not supported in V2 (file block url `{path}`); \
             copy the file into the workspace and reference it by relative path"
        )));
    }
    if Path::new(path).is_absolute() {
        return Err(ProviderAdapterError::configuration(format!(
            "file block url must be a workspace-relative path, got absolute path `{path}`"
        )));
    }
    let mut parts: Vec<String> = Vec::new();
    for component in Path::new(path).components() {
        match component {
            Component::Normal(part) => parts.push(part.to_string_lossy().into_owned()),
            Component::CurDir => {}
            Component::ParentDir => {
                return Err(ProviderAdapterError::configuration(format!(
                    "file block url must stay inside the workspace \
                     (path traversal rejected: `{path}`)"
                )));
            }
            _ => {
                return Err(ProviderAdapterError::configuration(format!(
                    "file block url must be a workspace-relative path (rejected: `{path}`)"
                )));
            }
        }
    }
    if parts.is_empty() {
        return Err(ProviderAdapterError::configuration(
            "file block url must not be empty",
        ));
    }
    let full = base_dir.join(parts.join("/"));
    let meta = std::fs::metadata(&full).map_err(|error| {
        ProviderAdapterError::configuration(format!(
            "failed to read workspace file `{}`: {error}",
            full.display()
        ))
    })?;
    if meta.len() > MAX_WORKSPACE_FILE_BYTES {
        return Err(ProviderAdapterError::configuration(format!(
            "workspace file `{}` exceeds the {} byte limit",
            full.display(),
            MAX_WORKSPACE_FILE_BYTES
        )));
    }
    std::fs::read(&full).map_err(|error| {
        ProviderAdapterError::configuration(format!(
            "failed to read workspace file `{}`: {error}",
            full.display()
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn temp_dir(prefix: &str) -> std::path::PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock before unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("reimagine-ai-protocol-files-{prefix}-{nonce}"))
    }

    #[test]
    fn reads_file_inside_workspace() {
        let base = temp_dir("read");
        fs::create_dir_all(base.join("refs")).unwrap();
        fs::write(base.join("refs/pic.png"), b"hello").unwrap();
        let bytes = read_workspace_file(&base, "refs/pic.png").expect("ok");
        assert_eq!(bytes, b"hello");
        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn reads_dot_prefixed_relative_paths() {
        let base = temp_dir("dots");
        fs::create_dir_all(base.join("refs")).unwrap();
        fs::write(base.join("refs/pic.png"), b"x").unwrap();
        let bytes = read_workspace_file(&base, "./refs/./pic.png").expect("ok");
        assert_eq!(bytes, b"x");
        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn rejects_parent_traversal() {
        let base = temp_dir("traversal");
        fs::create_dir_all(&base).unwrap();
        let err = read_workspace_file(&base, "../secret.txt").expect_err("must reject");
        assert!(err.to_string().contains("path traversal"), "{err}");
        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn rejects_absolute_paths() {
        let base = temp_dir("absolute");
        fs::create_dir_all(&base).unwrap();
        let err = read_workspace_file(&base, "/etc/passwd").expect_err("must reject");
        assert!(err.to_string().contains("workspace-relative"), "{err}");
        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn rejects_remote_urls() {
        let base = temp_dir("remote");
        fs::create_dir_all(&base).unwrap();
        for url in ["https://example.com/pic.png", "http://example.com/pic.png"] {
            let err = read_workspace_file(&base, url).expect_err("must reject");
            assert!(
                err.to_string()
                    .contains("remote URLs are not supported in V2"),
                "{err}"
            );
        }
        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn rejects_missing_files() {
        let base = temp_dir("missing");
        fs::create_dir_all(&base).unwrap();
        let err = read_workspace_file(&base, "no-such.png").expect_err("must reject");
        assert!(
            err.to_string().contains("failed to read workspace file"),
            "{err}"
        );
        let _ = fs::remove_dir_all(&base);
    }

    #[test]
    fn rejects_oversized_files() {
        let base = temp_dir("oversize");
        fs::create_dir_all(&base).unwrap();
        let big = vec![0u8; MAX_WORKSPACE_FILE_BYTES as usize + 1];
        fs::write(base.join("big.png"), &big).unwrap();
        let err = read_workspace_file(&base, "big.png").expect_err("must reject");
        assert!(err.to_string().contains("exceeds the"), "{err}");
        let _ = fs::remove_dir_all(&base);
    }
}
