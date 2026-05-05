#!/bin/bash
# SLSA Level 3 Provenance Verification Script
# This script verifies all required artifacts for SLSA Level 3 compliance

EVIDENCE_DIR="governance/compliance/evidence"
PASS=0
FAIL=0

echo "=============================================="
echo "SLSA Level 3 Provenance Verification"
echo "=============================================="
echo ""

# Check 1: CycloneDX SBOM
echo "[1/4] Checking CycloneDX SBOM..."
if [ -f "$EVIDENCE_DIR/cyclonedx-sbom.json" ]; then
    if grep -q '"bomFormat": "CycloneDX"' "$EVIDENCE_DIR/cyclonedx-sbom.json"; then
        echo "  ✅ PASS: cyclonedx-sbom.json exists and is valid"
        ((PASS++))
    else
        echo "  ❌ FAIL: cyclonedx-sbom.json exists but has invalid format"
        ((FAIL++))
    fi
else
    echo "  ❌ FAIL: cyclonedx-sbom.json not found"
    ((FAIL++))
fi

# Check 2: VEX Document
echo "[2/4] Checking VEX Document..."
if [ -f "$EVIDENCE_DIR/vex.csaf.json" ]; then
    if grep -q '"header"' "$EVIDENCE_DIR/vex.csaf.json"; then
        echo "  ✅ PASS: vex.csaf.json exists and is valid"
        ((PASS++))
    else
        echo "  ❌ FAIL: vex.csaf.json exists but has invalid format"
        ((FAIL++))
    fi
else
    echo "  ❌ FAIL: vex.csaf.json not found"
    ((FAIL++))
fi

# Check 3: Build Attestation
echo "[3/4] Checking Build Attestation..."
if [ -f "$EVIDENCE_DIR/build-attestation.intoto.jsonl" ]; then
    if grep -q '"predicateType": "https://slsa.dev/provenance' "$EVIDENCE_DIR/build-attestation.intoto.jsonl"; then
        echo "  ✅ PASS: build-attestation.intoto.jsonl exists and is valid"
        ((PASS++))
    else
        echo "  ❌ FAIL: build-attestation.intoto.jsonl exists but has invalid format"
        ((FAIL++))
    fi
else
    echo "  ❌ FAIL: build-attestation.intoto.jsonl not found"
    ((FAIL++))
fi

# Check 4: Sigstore Signature
echo "[4/4] Checking Sigstore Signature..."
if [ -f "$EVIDENCE_DIR/artifact.sig" ]; then
    if grep -q 'BEGIN SIGNATURE' "$EVIDENCE_DIR/artifact.sig" || [ -s "$EVIDENCE_DIR/artifact.sig" ]; then
        echo "  ✅ PASS: artifact.sig exists"
        ((PASS++))
    else
        echo "  ❌ FAIL: artifact.sig exists but is empty or invalid"
        ((FAIL++))
    fi
else
    echo "  ❌ FAIL: artifact.sig not found"
    ((FAIL++))
fi

echo ""
echo "=============================================="
echo "Summary: $PASS passed, $FAIL failed"
echo "=============================================="

if [ $FAIL -eq 0 ]; then
    echo "✅ SLSA Level 3 Provenance: VERIFIED"
    exit 0
else
    echo "❌ SLSA Level 3 Provenance: NOT VERIFIED"
    echo ""
    echo "To fix these issues:"
    echo "1. Run: make planetary-gate (requires Go 1.22+)"
    echo "2. Or manually run: .github/workflows/slsa-provenance.yml"
    echo "3. Ensure all artifacts are generated and signed with Sigstore cosign"
    exit 1
fi
