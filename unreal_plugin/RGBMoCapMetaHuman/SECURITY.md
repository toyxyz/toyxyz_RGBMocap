# Security and privacy

The app-to-owned-bridge connection uses loopback TCP, a per-launch random nonce,
bounded input and complete-snapshot validation. The nonce is supplied in the
child environment and cleared there after use; it is not a distributed secret.
This is not protection against a compromised local account.

Unreal's built-in UDP Message Bus is a **trusted-network protocol**, not an
encrypted or authenticated public-Internet endpoint. Use local/trusted networks
and restrict UDP Messaging/firewall interfaces to intended peers. Do not expose
it directly to untrusted networks. The plugin evaluates exact source-qualified
subjects and validates matching provider, rig and body/face frame metadata;
these checks prevent accidental mixing, not a hostile LAN impersonation attack.

The selected app people publish bone/rest transforms, lengths, person labels,
short subject identity suffixes and optional ARKit face coefficients. Video,
mesh vertices, model files, source paths and neutral calibration files are not
sent. The plugin's Face switch controls applying received expressions; it does
not disable their publication. App person checkboxes and Target None control
outgoing streams. Face Overlay visibility is a display control.

Project files, local logs and inference/capture archives remain private user
data. Inspect/redact them before sharing. The distributed allowlist omits logs,
credentials, personal paths, private test results, character assets and PDBs.
Embedded DLL strings and the exact extracted ZIP are audited before release;
checksums verify artifact bytes but do not independently establish trust in a
publisher. Native binaries are unsigned; source is included for inspection and
rebuilding. Report security concerns privately to the repository owner rather
than posting credentials or raw private logs in an issue.
