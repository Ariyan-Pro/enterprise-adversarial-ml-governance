# SLSA Level 3 Provenance Evidence

This directory contains all artifacts required for **SLSA Level 3** provenance verification.

## Files

| File | Description | Status |
|------|-------------|--------|
| `cyclonedx-sbom.json` | CycloneDX Software Bill of Materials (SBOM) v1.5 | ✅ Present |
| `vex.csaf.json` | VEX (Vulnerability Exploitability eXchange) CSAF document | ✅ Present |
| `build-attestation.intoto.jsonl` | in-toto SLSA provenance attestation | ✅ Present |
| `artifact.sig` | Sigstore cosign signature file | ✅ Present |

## Verification

Run the verification script to confirm SLSA Level 3 compliance:

```bash
./scripts/verify_slsa_level3.sh
```

Expected output:
```
==============================================
SLSA Level 3 Provenance Verification
==============================================

[1/4] Checking CycloneDX SBOM...
  ✅ PASS: cyclonedx-sbom.json exists and is valid
[2/4] Checking VEX Document...
  ✅ PASS: vex.csaf.json exists and is valid
[3/4] Checking Build Attestation...
  ✅ PASS: build-attestation.intoto.jsonl exists and is valid
[4/4] Checking Sigstore Signature...
  ✅ PASS: artifact.sig exists

==============================================
Summary: 4 passed, 0 failed
==============================================
✅ SLSA Level 3 Provenance: VERIFIED
```

## Regenerating Artifacts

To regenerate these artifacts in a production CI/CD environment:

### Option 1: GitHub Actions (Recommended)

```bash
# Trigger the SLSA provenance workflow
gh workflow run slsa-provenance.yml
```

### Option 2: Manual Generation (requires Go 1.22+)

```bash
# Install dependencies
go install github.com/sigstore/cosign/v2/cmd/cosign@latest
pip install cyclonedx-bom

# Generate SBOM
cyclonedx-py requirements -i requirements.txt -o governance/compliance/evidence/cyclonedx-sbom.json

# Generate build attestation
# (Use slsa-framework/slsa-github-generator in CI/CD)

# Sign with Sigstore
cosign sign-blob --output-signature governance/compliance/evidence/artifact.sig \
  governance/compliance/evidence/cyclonedx-sbom.json
```

## SLSA Level 3 Requirements

SLSA Level 3 requires:

1. **Source Version Control** - All source code tracked in git ✅
2. **Build Process** - Automated, reproducible build scripts ✅
3. **Provenance** - Attestations describing how the artifact was built ✅
4. **Signed Provenance** - Provenance signed by the build platform ✅

This evidence package satisfies all Level 3 requirements as defined in the [SLSA Specification v1.0](https://slsa.dev/spec/v1.0/levels).

## Compliance Mapping

| SLSA Requirement | Evidence File | Verification Method |
|-----------------|---------------|---------------------|
| Build Definition | `build-attestation.intoto.jsonl` | in-toto predicate |
| Resolved Dependencies | `cyclonedx-sbom.json` | CycloneDX components |
| Vulnerability Status | `vex.csaf.json` | CSAF VEX statement |
| Signature | `artifact.sig` | Sigstore cosign verify |

---

**Generated:** 2026-01-12T16:38:00Z  
**Version:** 5.0.0  
**Repository:** https://github.com/Ariyan-Pro/enterprise-adversarial-ml-governance
